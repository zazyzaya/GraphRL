import torch
import random 
import numpy as np

from copy import deepcopy
from tqdm import tqdm
from joblib import Parallel, delayed
from abc import ABC, abstractmethod
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules import Bilinear, Linear, ModuleList, Sequential, Sigmoid, ReLU, Tanh
from torch_geometric.utils import degree, to_dense_adj
from scipy.sparse import csr_matrix

from gensim.models import Word2Vec

class Q_Network(torch.nn.Module):
    def __init__(self, state_feats, action_feats, hidden=16, layers=1):
        super().__init__()
        
        self.in_dim = state_feats + action_feats
        self.hidden = hidden
        
        self.lin = Sequential(Linear(self.in_dim, self.hidden), Tanh(), 
                              Linear(self.hidden, self.hidden//10), Tanh())
        
        self.out = Linear(self.hidden//10, 1)
        
    def forward(self, s, a):
        x = torch.cat((s,a),dim=1)
        
        x = self.lin(x)
        return self.out(x)
    
    def reset_parameters(self):
        self.lin = Sequential(Linear(self.in_dim, self.hidden), Tanh(), 
                              Linear(self.hidden, self.hidden//10), Tanh())
        
        self.out = Linear(self.hidden//10, 1)

'''
Generates random walks based on some user specified reward function
'''
class Q_Walker(ABC):
    def __init__(self, data, state_feats=None, action_feats=None,
                 gamma=0.99, epsilon=lambda x: 0.5, episode_len=10,
                 num_walks=10, hidden=64, network=None, edata=None,
                 frozen=True):
        if state_feats == None:
            state_feats=data.num_nodes
        if action_feats == None:
            action_feats=data.num_nodes
           
        self.state_feats=state_feats
        self.action_feats=action_feats 
            
        # Just one layer for now. Technically this problem is supposed to be 
        # linearly seperable. But for more complex reward functions it may 
        # be beneficial to toss in another layer
        if network == None:
            self.qNet = Q_Network(self.state_feats, self.action_feats, hidden=hidden)
        else:
            self.qNet = network
            
        self.use_frozen_q = frozen
        
        if frozen:
            self.qNet_theta_negative = deepcopy(self.qNet)
            self.qNet_theta_negative.reset_parameters()
        
        self.data = data
        
        self.episode_cnt = 1
        self.step = 0
        self.epsilon = epsilon
        self.gamma = gamma
        self.episode_len = episode_len
        self.num_walks = num_walks
        
        if edata == None:
            edata = torch.full(data.edge_index[0].size(), 1, dtype=torch.long)
        
        # CSR matrices make it easier to calculate neighbors
        self.csr = csr_matrix((
            edata,
            (data.edge_index[0].numpy(), data.edge_index[1].numpy())), 
            shape=(data.num_nodes, data.num_nodes)
        )
        
        # Dense adj matrices make concurrent episode generation possible
        # may not be worth the memory overhead?
        # self.dense_adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)
        
    def parameters(self):
        return self.qNet.parameters()
    
    '''
    Must calculate some reward from the state, action and next state
    given the Q function 
    '''
    @abstractmethod
    def reward(self,s,a,s_prime,nid):
        pass
    
    '''
    A few built-in reward functions to play with
    '''
    def min_similarity_reward(self, s,a,s_prime,nid):
        return s.logical_xor(s_prime).sum()
    
    def max_similarity_reward(self, s,a,s_prime,nid):
        return s.logical_and(s_prime).sum()
    
    def max_degree_reward(self, s,a,s_prime,nid):
        return (self.data.edge_index[0,:] == nid).sum()
    
    def min_degree_reward(self, s,a,s_prime,nid):
        return 1 / (self.data.edge_index[0,:] == nid).sum()
    
    '''
    Given a state, and a chosen action, returns the next state, s' that
    the module will transition to upon taking that action.
    
    By default, just returns the one-hot encoding of the next node to explore
    '''
    def state_transition(self, s,a=None):
        return self.state_transition_one_hot(s,a)
    
    def state_transition_node_feats(self, s,a=None):
        if a == None:
            return self.data.x[s]
        
        return self.data.x[a]
    
    def state_transition_one_hot(self, s,a=None):
        ret = torch.zeros(self.data.num_nodes, dtype=torch.float)
        
        if a == None:
            ret[s]=1
            return ret
        else:
            ret[a]=1
            return ret
    
    
    def state_transition_combine(self, s,a=None):
        if a == None:
            return self.data.x[s]
        
        return (self.data.x[a].logical_or(s)).float()
    
    '''
    Given a list of neighbors, encode them. As of right now, 
    just uses one-hot
    '''
    def encode_actions(self, acts, nid=None):
        actions = torch.zeros((len(acts), self.action_feats))
        for a in range(len(acts)):
            actions[a][acts[a]] = 1
            
        return actions  
    
    '''
    Can override the above method with this one to use node feats
    as actions
    '''
    def encode_actions_node_feats(self, actions):
        return self.data.x[torch.tensor(actions, dtype=torch.long)]                  
    
    '''
    Takes a node feature, and puts it at the a'th index,
    then runs it through the q net. 
    
    Can also accept a list of actions and run the Q(s,a) for 
    each of them (useful for finding Q(s,a)_max(a) )
    
    TODO make this work with more than one state at a time?
    '''
    def Q(self,s,a,theta_negative=False):    
        if len(s.size()) == 1:
            s = s.unsqueeze(dim=0)
        if s.size()[0] != a.size()[0]:
            s = s.expand(a.size()[0], s.size()[1])
          
        # So loss function doesn't explode
        if theta_negative and self.use_frozen_q:
            return self.qNet_theta_negative(s,a)
        else:  
            return self.qNet(s,a)
        
    '''
    Want to keep the network Q* uses stationary for faster
    convergence. Only update it every few epochs
    '''
    def reparameterize_Q(self):
        if self.use_frozen_q:
            self.qNet_theta_negative = deepcopy(self.qNet)
    
    '''
    Estimate for updating Q policy
    '''
    def Q_star(self, s,a,s_prime,nid):
        r = self.reward(s,a,s_prime,nid)
        next_r = self.gamma * self.policy(s_prime, return_value=True, nid=a)
        
        return r + next_r
    
    '''
    Chooses next action with optional e-greedy policy 
    '''
    def policy(self, s, nid, egreedy=True, return_value=False, weighted_rand=False):    
        neighbors = self.csr[nid].indices
        
        # Can't do anything about orphaned nodes
        if len(neighbors) == 0:
            return nid
        
        actions = self.encode_actions(neighbors)
        
        if not return_value and egreedy and random.random() < self.epsilon(self.episode_cnt):
            action = np.random.choice(neighbors)    
        else:
            value_predictions = self.Q(s,actions,theta_negative=True)
            
            # Use this to make biased selection for random walk
            if weighted_rand:
                return neighbors[torch.multinomial(F.relu(value_predictions).T + 1e-10, 1)[-1]]
            
            # We can also use this function to calculate max(a) Q(s,a)
            if return_value:
                return value_predictions.max().item()
            
            action = value_predictions.argmax().item()
            action = neighbors[action]
           
        # Action is the NID of the next node to visit 
        return action
    
    '''
    One unit of work for paralell execution 
    '''
    def episode_task(self, nid):
        s = self.state_transition(nid)
        states = []
        actions = []
        rewards = []
            
        for _ in range(self.episode_len):    
            a = self.policy(s, nid)
            next_s = self.state_transition(s,a=a)
            r = self.Q_star(s,a,next_s,nid)
            
            states.append(s)
            actions.append(a)
            rewards.append(r)
            
            nid = a
            s = next_s
            
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards
        }
        
    '''
    Generates walk guided by policy
    '''
    def policy_walk(self, nid, egreedy=True):
        walks = []    
            
        for __ in range(self.num_walks):
            walk = [nid]
            s = self.state_transition(nid)
            
            for _ in range(self.episode_len-1):        
                a = self.policy(s, walk[-1], egreedy=egreedy)
                s = self.state_transition(s,a=a)
                
                # String so w2v can use it
                walk.append(a)
            walks.append([str(w) for w in walk])
            
        return walks
    
    def policy_weighted_walk(self, nid, **kwargs):
        walks = []    
            
        for __ in range(self.num_walks):
            walk = [nid]
            s = self.state_transition(nid)
            
            for _ in range(self.episode_len-1):        
                a = self.policy(s, walk[-1], egreedy=False, weighted_rand=True)
                s = self.state_transition(s,a=a)
                
                # String so w2v can use it
                walk.append(a)
            walks.append([str(w) for w in walk])
            
        return walks
    
    '''
    Generates random walk (for comparison)
    kwargs appears so it can have the same function header as 
    self.policy_walk for testing
    '''
    def random_walk(self, nid, **kwargs):
        walks = []
        
        for __ in range(self.num_walks):
            walk = [nid]
            
            for _ in range(self.episode_len-1):
                neighbors = self.csr[walk[-1]].indices
                n = np.random.choice(neighbors)
                
                walk.append(n)
                 
            walks.append([str(w) for w in walk])
            
        return walks 
    
    
    def _combine_dicts(self, dicts, key):
        l = []
        [l.append(s) for d in dicts for s in d[key]]
        return l
    
    '''
    Only use if state vector isn't dependant on previous states
    Faster way to calculate observed rewards.
    
    Given a batch of states, generates random actions and rewards for
    those actions. Returns matrix of s,a,q*(s,a)
    
    TODO 
    '''
    def generate_rewards(self, batch=[]):
        if len(batch) == 0:
            s = torch.tensor(range(self.data.num_nodes))
        else:
            s = torch.tensor(batch)
            
        a = torch.multinomial(self.dense_adj[s], num_samples=1)
        r = self.q_star(s)
        
    
    '''
    Runs one episode from each node and updates weights
    This needs to be seriously optimised
    '''
    def episode(self, batch=[], workers=-1, quiet=False):
        if len(batch) == 0:
            batch = list(range(self.data.num_nodes))
        
        # Don't bother with gradients while we generate walks. Saves some time
        with torch.no_grad():
            episodes = Parallel(n_jobs=workers, prefer='threads')(
                delayed(self.episode_task)(
                    nid
                )
                for nid in tqdm(batch, desc='Episodes completed', disable=quiet)
            )

        states = self._combine_dicts(episodes, 'states')
        actions = self._combine_dicts(episodes, 'actions')
        rewards = self._combine_dicts(episodes, 'rewards')
            
        states = torch.stack(states)
        actions = self.encode_actions(actions)
        rewards = torch.Tensor([rewards]).T
        
        return states, actions, rewards
        
        
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  
from sklearn.multiclass import OneVsRestClassifier as OVR
from sklearn.linear_model import LogisticRegression as LR
class RW_Encoder():
    def __init__(self, walker):
        self.walker = walker 
        
    def generate_walks(self, batch=[], workers=-1, random=False, 
                       egreedy=True, quiet=False, include_labels=False,
                       weighted=False):
        if random:
            walk_fn = self.walker.random_walk
        elif weighted: 
            walk_fn = self.walker.policy_weighted_walk
        else:
            walk_fn = self.walker.policy_walk
        
        with torch.no_grad():
            all_walks = Parallel(n_jobs=workers, prefer='threads')(
                delayed(walk_fn)(
                    nid,
                    egreedy=egreedy
                )
                for nid in tqdm(batch, desc='Walks generated', disable=quiet)
            )
            
        walks = []
        if not include_labels:
            for node_walks in all_walks:
                walks += node_walks
            return walks
        
        # Asumes return is a list of dicts w labels
        else:
            y = torch.zeros((self.walker.num_walks * len(batch), 
                             self.walker.data.y.size()[1]))
            last_idx = 0
            
            for node_walk in all_walks:
                y[last_idx:last_idx+self.walker.num_walks] = node_walk['label']
                walks += node_walk['walks']
                last_idx += self.walker.num_walks
                
            return walks, y
                
    
    
    def encode_nodes(self, batch=[], walks=None, random=False, w2v_params={}, weighted=False):
        # Make sure required params exist in w2v_params 
        required = dict(size=64, sg=1, workers=16)
        for k,v in required.items():
            if k not in w2v_params:
                w2v_params[k] = v    
        
        if walks == None:
            walks = self.generate_walks(batch=batch, workers=w2v_params['workers'], random=random, weighted=weighted)
            
        model = Word2Vec(walks, **w2v_params)
        
        idx_order = torch.tensor([int(i) for i in model.wv.index2entity], dtype=torch.long)
        y = self.walker.data.y[idx_order].numpy()
        X = model.wv.vectors
        
        return X, y
    
    def compare_to_random(self, batch, w2v_params={}, multiclass=False):
        # Test against policy weighted walks
        wX, wy = self.encode_nodes(batch=batch, weighted=True, w2v_params=w2v_params)
        
        # Generate policy guided walks
        pX, py = self.encode_nodes(batch=batch, w2v_params=w2v_params)
        
        # Test against random walk embeddings
        rX, ry = self.encode_nodes(batch=batch, random=True, w2v_params=w2v_params)

        if multiclass:
            estimator = lambda : OVR(LR(), n_jobs=16)
            y_trans = lambda y : y 
        else:
            estimator = lambda : LR(n_jobs=16)
            y_trans = lambda y : y.argmax(axis=1)

        lr = estimator()
        Xtr, Xte, ytr, yte = train_test_split(pX, y_trans(py))
        lr.fit(Xtr, ytr)
        yprime = lr.predict(Xte)
        
        print(yprime)
        print("Policy guided:")
        print(classification_report(yprime, yte))
        
        lr = estimator()
        Xtr, Xte, ytr, yte = train_test_split(wX, y_trans(wy))
        lr = OVR(LR(), n_jobs=16)
        lr.fit(Xtr, ytr)
        yprime = lr.predict(Xte)
        print("Policy weighted:")
        print(classification_report(yprime, yte))
        
        lr = estimator()
        Xtr, Xte, ytr, yte = train_test_split(rX, y_trans(ry))
        lr = OVR(LR(), n_jobs=16)
        lr.fit(Xtr, ytr)
        yprime = lr.predict(Xte)
        print("Random walk:")
        print(classification_report(yprime, yte))
            
    
# Example 
class Q_Walk_Simplified(Q_Walker):
    def __init__(self, data, gamma=0.99, epsilon=lambda x: 0.5, episode_len=10,
                 num_walks=10, hidden=64, one_hot=False, network=None):
        
        self.one_hot = one_hot
        
        # Don't bother with node features at all 
        # Make sure to update state/action transition functions accordingly
        if one_hot:
            super().__init__(data, gamma=gamma, epsilon=epsilon, episode_len=episode_len, 
                            num_walks=num_walks, hidden=hidden, network=network)  
        # Set state and action feats to be dim of node features
        else:
            super().__init__(data, gamma=gamma, epsilon=epsilon, episode_len=episode_len, 
                            num_walks=num_walks, hidden=hidden, state_feats=data.x.size()[1],
                            action_feats=data.x.size()[1], network=network)     
    
    
    def reward(self, s,a,s_prime,nid):
        return self.max_degree_reward(s,a,s_prime,nid)
    
    def state_transition(self, s,a=None):
        if not self.one_hot:
            return self.state_transition_node_feats(s,a)
        else:
            return super().state_transition(s,a)
    
    def encode_actions(self, actions):
        if not self.one_hot: 
            return self.encode_actions_node_feats(actions)
        else:
            return super().encode_actions(actions)
        

def train_loop(Agent, data, sample_size=50, epochs=200, clip=None, 
                  reparam=40, lr=1e-4, verbose=1, early_stopping=10):
    
    # While training, look at each node in isolation with a random action
    nw = Agent.num_walks
    wl = Agent.episode_len
    eps = Agent.epsilon
    
    Agent.num_walks = 1
    Agent.episode_len = 1
    Agent.epsilon = lambda x : 0
    
    # Get rid of nodes without neighbors
    non_orphans = (degree(data.edge_index[0], num_nodes=data.num_nodes) != 0).nonzero()
    non_orphans = non_orphans.T.numpy()[0]
    
    # Just need to learn the reward for individual nodes and the best neighbor 
    # So, just train on every node in the network
    tot_loss = float('inf')
    is_early_stopping = False
    reparammed = False
    opt = torch.optim.Adam(Agent.parameters(), lr=1e-2)
    for e in range(epochs):
        if tot_loss < reparam:
            # Don't want to stop right after a param update
            # Make sure the model knows to fit to the updated one as well as it 
            # was fit earlier before halting
            if is_early_stopping:
                if reparammed:
                    print("Early stopping")
                    break
                else:
                    reparammed = True
            
            print("Updating Q(-theta)")
            Agent.reparameterize_Q()
            
            # Dont reparam again until at least loss is that low
            reparam = tot_loss
        
        if sample_size:    
            b = np.array_split(non_orphans, non_orphans.shape[0]//sample_size)
        else:
            b = [non_orphans]
        
        tot_loss = 0
        steps = 0
        for batch in b:
            s,a,r = Agent.episode(batch=batch, workers=8, quiet=True)
            opt.zero_grad()
            loss = F.mse_loss(Agent.Q(s,a), r)
            loss.backward()
            
            if clip:
                torch.nn.utils.clip_grad_norm_(Agent.parameters(), clip)
            
            print("\t[%d-%d]: %0.4f" % (e,steps,loss.item()))
            
            opt.step()
            tot_loss += loss.item()
            steps += 1
    
        tot_loss = tot_loss/steps
        print("[%d]: %0.4f" % (e, tot_loss))
        
        if tot_loss <= early_stopping:
            print("Preparing for early stopping")
            is_early_stopping = True
        
    # Set parameters back when returning agent
    Agent.num_walks = nw
    Agent.episode_len = wl
    Agent.epsion = eps
    
    return non_orphans

def example(sample_size=50, epochs=200, clip=None, reparam=40,
            gamma=0.99, nw=10, wl=5):
    import load_graphs as lg
    data = lg.load_cora()
    
    # Set up a basic agent 
    Agent = Q_Walk_Simplified(data, episode_len=wl, num_walks=nw, 
                           epsilon=lambda x : 0.95, gamma=0.999,
                           hidden=1000, one_hot=True)

    Encoder = RW_Encoder(Agent)
    
    non_orphans = train_loop(Agent, data, epochs=epochs, sample_size=sample_size, 
                             reparam=reparam, clip=clip)    
    
    Encoder.compare_to_random(non_orphans)
    
        
if __name__ == '__main__':
    example(epochs=100, sample_size=400)