import torch
import random 
import numpy as np

from torch_scatter import scatter_mean
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
                 num_walks=10, hidden=64, network=None, edata=False,
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
        
        self.episode_cnt = 0
        self.step = 0
        self.epsilon = epsilon
        self.gamma = gamma
        self.episode_len = episode_len
        self.num_walks = num_walks
        
        self.edata = edata
        self.csr = None
        
        # Dense adj matrices make concurrent episode generation possible
        # may not be worth the memory overhead?
        self.dense_adj = None
        
        # If there's a one-to-one mapping between states and nodes we can do
        # pseudo-tabular Q learning 
        self.Q_matrix = None
        
    def _get_csr(self):
        if self.csr == None:
            if self.edata == False:
                edata = torch.full(self.data.edge_index[0].size(), 1, dtype=torch.long)
            else:
                edata = self.data.edge_data
        
            # CSR matrices make it easier to calculate neighbors
            self.csr = csr_matrix((
                edata,
                (self.data.edge_index[0].numpy(), self.data.edge_index[1].numpy())), 
                shape=(self.data.num_nodes, self.data.num_nodes)
            )
            
        return self.csr
        
    def _get_dense_adj(self):
        if self.dense_adj == None:
            self.dense_adj = to_dense_adj(
                self.data.edge_index, 
                max_num_nodes=self.data.num_nodes
            )[-1].float()
            
            # Add self loops to prevent errors in rand walks
            for i in range(self.data.num_nodes):
                self.dense_adj[i,i] = 1
            
        return self.dense_adj
        
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
        return s.logical_xor(s_prime).sum(dim=1, keepdim=True).float()
    
    def max_similarity_reward(self, s,a,s_prime,nid):
        return s.logical_and(s_prime).sum(dim=1, keepdim=True).float()
    
    def max_degree_reward(self, s,a,s_prime,nid):
        return degree(self.data.edge_index[0])[a].unsqueeze(dim=-1)
    
    def min_degree_reward(self, s,a,s_prime,nid):
        return 1 / (self.max_degree_reward(s,a,s_prime,nid))
    
    def euclidean_dist_reward(self, s,a,s_prime,nid):
        return torch.mean((s-s_prime)**2, dim=1, keepdim=True)
    
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
        ret = torch.eye(self.data.num_nodes, dtype=torch.float)
        
        if a == None:
            return ret[s]
        else:
            return ret[a]
    
    
    def state_transition_combine(self, s,a=None):
        if a == None:
            return self.data.x[s]
        
        return (self.data.x[a].logical_or(s)).float()
    
    '''
    Given a list of neighbors, encode them. As of right now, 
    just uses one-hot
    '''
    def encode_actions(self, acts, nid=None):   
        # It really should be a tensor, but in case i forgot to replace
        # code somewhere, allow a list too 
        if type(acts) == torch.Tensor:
            return torch.eye(self.data.num_nodes)[acts.long()]
        
        return torch.eye(self.data.num_nodes)[torch.tensor(acts, dtype=torch.long)]
    
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
        neighbors = self._get_csr()[nid].indices
        
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
    
    def _populate_q_table(self, batch, workers=16):
        # One unit of work for parallel exe
        def pqt_task(nid):
            neighbors = torch.tensor(self._get_csr()[nid].indices, dtype=torch.long, requires_grad=False)
            self.Q_matrix[nid][neighbors] = self.Q(
                self.state_transition(nid),
                self.encode_actions(neighbors)
            ).squeeze()
            
        Parallel(n_jobs=workers, prefer='threads')(
            delayed(pqt_task)(
                nid
            )
            for nid in tqdm(batch, desc='Q Table Rows Written')
        )
    
    def _get_q_table(self, batch, workers=16, hard_reset=False):
        if self.Q_matrix == None or hard_reset:
            self.Q_matrix = torch.zeros(
                (self.data.num_nodes, self.data.num_nodes),
                requires_grad=False
            )
            
            with torch.no_grad():
                self._populate_q_table(batch, workers=workers)
            
        return self.Q_matrix  
    
    '''
    Generate walks in parallel; much faster than Agent.episode
    however, requires that there is a 1 to 1 mapping between nodes
    and states. 
    
    E.g. if states are affected by previous nodes in the path, this
    will not work. However, if states are just encodings of vertexes 
    the agent is on at time t, this is the more efficient way to generate
    walks for word2vec
    '''
    def fast_walks(self, nids, silent=False, strings=True, strategy='random'):
        if strategy in ['egreedy', 'weighted']:
            Q = self._get_q_table(nids)
        if len(nids.size()) == 1:
            nids = nids.unsqueeze(dim=-1)
        
        walks = []
        node = nids
        for _ in tqdm(range(self.num_walks), desc='Walks generated', disable=silent):
            walk = [nids]
            with torch.no_grad():
                for _ in range(self.episode_len):
                    if strategy=='egreedy' and random.random() < self.epsilon(None):
                        node = Q[node.squeeze()].argmax(dim=1, keepdim=True)
                    elif strategy=='weighted':
                        # Since all scores are relatively similar, make the 
                        # difference more pronounced 
                        prob_distro = Q[node.squeeze()]
                        prob_distro = prob_distro - prob_distro[prob_distro > 0].min()
                        
                        node = torch.multinomial(F.relu(prob_distro)+1e-10, num_samples=1)
                    else:
                        node = torch.multinomial(self._get_dense_adj()[node.squeeze()], num_samples=1)
                    
                    walk.append(node)
            
            walk = torch.cat(walk, dim=1)
            if strings:
                walks += [[str(item.item()) for item in one_walk] for one_walk in walk]
            else:
                walks.append(walk)
        
        if not strings:
            walks = torch.cat(walks, dim=0)
        
        return walks 
    
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
                neighbors = self._get_csr()[walk[-1]].indices
                n = np.random.choice(neighbors)
                
                walk.append(n)
                 
            walks.append([str(w) for w in walk])
            
        return walks 
    
    
    def _combine_dicts(self, dicts, key):
        l = []
        [l.append(s) for d in dicts for s in d[key]]
        return l
    
    '''
    Approximate state-action values by generating a random
    walk, and returning the sum of the discounted future
    rewards resulting from that walk
    '''
    def fast_episode_generation(self, batch=[], wl=None, nw=None):
        # Simulate nw random walks of len wl
        if wl==None:
            wl = self.episode_len
        if nw==None:
            nw = self.num_walks
        
        # Copy the random walks nw times (repeat is memory efficient, doesn't
        # actually use nw * t memory, it just makes new pointers or something)
        if len(batch) == 0:
            batch = torch.tensor([range(self.data.num_nodes)])
        else:
            batch = torch.tensor(batch)

        # Builds or retrieves the adj matrix            
        adj = self._get_dense_adj()
        
        # Take one random step to learn from
        state = self.state_transition(batch)
        nids = torch.multinomial(adj[batch], num_samples=1).squeeze()
        action = self.encode_actions(nids)
        
        # Starting state is next step
        s = self.state_transition(state, a=nids).repeat((nw,1))
        a = action.repeat((nw,1))
        nids = nids.repeat(nw)
        idx = torch.tensor(range(batch.size()[0])).repeat(nw)
        
        # And add reward for initial random step
        rewards = [self.reward(state.repeat((nw,1)), nids, s, batch.repeat(nw))]
          
        for _ in range(wl):
            # Randomly sample an action. By using the adj matrix as the 
            # prob distro, we guarantee that a neighbor is always selected
            # without having to use the csr matrix which is ragged and not 
            # good for matrix ops    
            a = torch.multinomial(adj[nids], num_samples=1).squeeze()
            
            s_prime = self.state_transition(s,a=a)
            rewards.append(self.reward(s,a,s_prime,nids))
            
            nids = a
            a = self.encode_actions(nids)
            s = s_prime
        
        rewards = torch.cat(rewards, dim=1)    
        
        discount = torch.tensor([self.gamma ** i for i in range(wl+1)])
        rewards *= discount
        rewards = rewards.sum(dim=1, keepdim=True)
        
        # Then find average reward of random walks
        rewards = scatter_mean(rewards.T, idx).T
        return state, action, rewards
        
    
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
    
    def generate_walks_fast(self, batch=[], strategy='weighted', 
                            silent=True, strings=False, encode=False,
                            w2v_params=dict()):
        if batch == []:
            batch = range(self.walker.data.num_nodes)
        
        if type(batch) != torch.Tensor:
            batch = torch.tensor(batch)
            
        if not silent:
            print("Generating %s walks" % strategy)
        
        with torch.no_grad():
            if not encode:
                return self.walker.fast_walks(
                    batch,
                    strategy=strategy,
                    silent=silent,
                    strings=strings
                )
            
            walks = self.walker.fast_walks(
                    batch,
                    strategy=strategy,
                    silent=silent,
                    strings=True
            )
        
        return self.encode_nodes(batch=batch, walks=walks, w2v_params=w2v_params, quiet=silent)
    
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
                
    
    
    def encode_nodes(self, batch=[], walks=None, random=False, w2v_params={}, weighted=False, quiet=False):
        # Make sure required params exist in w2v_params 
        required = dict(size=128, sg=1, workers=16)
        for k,v in required.items():
            if k not in w2v_params:
                w2v_params[k] = v    
        
        if walks == None:
            walks = self.generate_walks(batch=batch, workers=w2v_params['workers'], random=random, weighted=weighted)
            
        if not quiet:
            print(walks[0])
            
        model = Word2Vec(walks, **w2v_params)
        
        idx_order = torch.tensor([int(i) for i in model.wv.index2entity], dtype=torch.long)
        X = torch.zeros((self.walker.data.x.size()[0], w2v_params['size']), dtype=torch.float)
        
        # Put embeddings back in order
        X[idx_order] = torch.tensor(model.wv.vectors)
        y = self.walker.data.y[:batch.max()+1, :]     
        
        # It seems like training more models improves later models'
        # accuracy? Putting this in to absolutely make sure runs
        # are indipendant
        del model   
        
        # Only select nodes with embeddings
        X = X[batch]
        y = y[batch] 
        
        return X, y
    
    def generate_mixed_walks(self, batch, mix_ratio=0.6, encode=True, 
                             silent=True, strategy='weighted'):
        nw = self.walker.num_walks 
        
        assert mix_ratio >= 0 and mix_ratio <= 1, "Mix ratio must be IR in [0,1]"
        
        if type(batch) != torch.Tensor:
            batch = torch.tensor(batch)
        
        if mix_ratio > 0:
            self.walker.num_walks = int(nw*mix_ratio)
            rand_walks = self.walker.fast_walks(
                batch,
                strategy='random',
                silent=silent,
                strings=True
            )
        else:
            rand_walks = []
        
        if mix_ratio != 1:
            self.walker.num_walks = int(nw*(1-mix_ratio))
            policy_walks = self.walker.fast_walks(
                batch,
                strategy=strategy,
                silent=silent,
                strings=True
            )
        else:
            policy_walks = []
        
        self.walker.num_walks = nw
        return self.encode_nodes(batch, walks=policy_walks+rand_walks, quiet=True)
    
    def get_accuracy_report(self,X,y,multiclass=False,test_size=0.1):
        if multiclass:
            estimator = lambda : OVR(LR(), n_jobs=16)
            y_trans = lambda y : y 
        else:
            estimator = lambda : LR(n_jobs=16, max_iter=1000)
            y_trans = lambda y : y.argmax(axis=1)
         
        strat = y_trans(y) if not multiclass else None   
        lr = estimator()
        Xtr, Xte, ytr, yte = train_test_split(
            X, y_trans(y), 
            stratify=strat, 
            test_size=test_size,
            random_state=1337
        )
        
        lr.fit(Xtr, ytr)
        yprime = lr.predict(Xte)
        
        return classification_report(yprime, yte, output_dict=True)
        
    
    def compare_to_random(self, batch, w2v_params={}, multiclass=False, fast_walks=False):
        if type(batch) != torch.Tensor and fast_walks:
            batch = torch.tensor(batch)
        
        # Generate policy guided walks
        print("Generating policy guided walks")
        if fast_walks:
            walks = self.walker.fast_walks(batch, egreedy=True, weighted_rand=False)
        else:
            walks = None 
            
        pX, py = self.encode_nodes(batch=batch, w2v_params=w2v_params, walks=walks)
        
        # Test against policy weighted walks
        print("Generating policy weighted walks")
        if fast_walks:
            walks = self.walker.fast_walks(batch, egreedy=False, weighted_rand=True)
        else:
            walks = None 
            
        wX, wy = self.encode_nodes(batch=batch, weighted=True, 
                                   w2v_params=w2v_params, walks=walks)
        
        # Test against random walk embeddings
        print("Generating random walks")
        if type(batch) != torch.Tensor:
            batch = torch.tensor(batch)
            
        walks = self.walker.fast_walks(batch, egreedy=False, weighted_rand=False)
        rX, ry = self.encode_nodes(batch=batch, random=True, w2v_params=w2v_params, walks=walks)

        if multiclass:
            estimator = lambda : OVR(LR(), n_jobs=16)
            y_trans = lambda y : y 
        else:
            estimator = lambda : LR(n_jobs=16, max_iter=1000)
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
class Q_Walk_Example(Q_Walker):
    def __init__(self, data, gamma=0.99, epsilon=lambda x: 0.5, episode_len=10,
                 num_walks=10, hidden=64, one_hot=False, network=None, frozen=True):
        
        self.one_hot = one_hot
        
        # Don't bother with node features at all 
        # Make sure to update state/action transition functions accordingly
        if one_hot:
            super().__init__(data, gamma=gamma, epsilon=epsilon, episode_len=episode_len, 
                            num_walks=num_walks, hidden=hidden, network=network, frozen=frozen)  
        # Set state and action feats to be dim of node features
        else:
            super().__init__(data, gamma=gamma, epsilon=epsilon, episode_len=episode_len, 
                            num_walks=num_walks, hidden=hidden, state_feats=data.x.size()[1],
                            action_feats=data.x.size()[1], network=network, frozen=frozen)     
    
    
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
        
import time 
def fast_train_loop(Agent, sample_size=None, clip=None, lr=1e-4, verbose=1, 
                    early_stopping=0.05, epochs=800, nw=None, wl=None, 
                    minibatch_bootstrap=False, strategy=None, train_eps=0.8):
    non_orphans = (degree(Agent.data.edge_index[0], num_nodes=Agent.data.num_nodes) > 1).nonzero()
    non_orphans = non_orphans.T.numpy()[0]
    
    original_eps = Agent.epsilon
    Agent.epsilon = lambda x : train_eps

    opt = torch.optim.Adam(Agent.parameters(), lr=1e-3, weight_decay=1e-4)
    for e in range(epochs):
        start = time.time()
        if sample_size:  
            np.random.shuffle(non_orphans)  
            b = np.array_split(non_orphans, non_orphans.shape[0]//sample_size)
        else:
            b = [non_orphans]
            
        steps = 0
        tot_loss = 0
        opt.zero_grad()
        for batch in b:
            if strategy:
                s,a,r = Agent.fast_episode_generation(batch=batch, nw=nw, wl=wl, strategy=strategy)
            # Allow backwards compatability w RL_Walker 
            else:
                s,a,r = Agent.fast_episode_generation(batch=batch, nw=nw, wl=wl)
                
            loss = F.mse_loss(Agent.Q(s,a), r)
            loss.backward()
            
            if clip:
                torch.nn.utils.clip_grad_norm_(Agent.parameters(), clip)
            
            if sample_size and verbose > 1:
                print("\t[%d-%d]: %0.5f" % (e,steps,loss.item()))
            
            avg_g = r.mean()
            tot_loss += loss.item()
            steps += 1
    
        opt.step()
        
        if sample_size:
            tot_loss = tot_loss/steps
        
        print("[%d]: loss: %0.5f \t avg G_t: %0.5f \t(%0.4f s.)" % (e, tot_loss, avg_g, time.time()-start))
        
        if tot_loss <= early_stopping:
            print("Early stopping")
            break
        
        if minibatch_bootstrap and sample_size:
            sample_size = int(sample_size * 1.5) if sample_size < non_orphans.shape[0] // 1.5 else None
        
    Agent.epsilon = original_eps
    s,a,r = Agent.fast_episode_generation(batch=non_orphans, nw=1)
    print(Agent.Q(s,a)[:10])
    return non_orphans

def train_loop(Agent, sample_size=50, clip=None, decreasing_param=False,
                  reparam=40, lr=1e-4, verbose=1, early_stopping=10,
                  training_wl=1, gamma_depth=None):
    
    # While training, look at each node in isolation with a random action
    nw = Agent.num_walks
    wl = Agent.episode_len
    eps = Agent.epsilon
    gamma = Agent.gamma 
    
    Agent.num_walks = 1
    
    # If the history of a walk is encoded into the state make this higher
    Agent.episode_len = training_wl 
    Agent.epsilon = lambda x : 0
    
    # Get rid of nodes without neighbors
    non_orphans = (degree(Agent.data.edge_index[0], num_nodes=Agent.data.num_nodes) != 0).nonzero()
    non_orphans = non_orphans.T.numpy()[0]
    
    # Just need to learn the reward for individual nodes and the best neighbor 
    # So, just train on every node in the network
    tot_loss = float('inf')
    is_early_stopping = False
    reparammed = False
    
    # Don't use discount factor until we have a pretty good estimate of the 
    # immidiate reward
    Agent.gamma = 0
    gamma_active = False
    distance_learned = 1
    gamma_depth = gamma_depth if gamma_depth else wl
    
    e = 0
    opt = torch.optim.Adam(Agent.parameters(), lr=1e-2)
    while True: 
        # Let the agent learn the immidiate rewards first
        if is_early_stopping and not gamma_active:
            Agent.gamma = gamma
            gamma_active = True
            
            # Also turn lr way down to help learn slower
            #for p in opt.param_groups:
            #    p['lr'] = 1e-4
        
        if tot_loss < reparam:
            # Don't want to stop right after a param update
            # Make sure the model knows to fit to the updated one as well as it 
            # was fit earlier before halting
            if is_early_stopping and distance_learned >= gamma_depth:
                if reparammed:
                    print("Early stopping")
                    break
                else:
                    reparammed = True
            
            if Agent.use_frozen_q:
                print("Updating Q(-theta) [gamma^%d]" % distance_learned)
                Agent.reparameterize_Q()
            
            # Dont reparam again until at least loss is that low
            if decreasing_param:
                reparam = tot_loss
                
            # We can think of every time we reparam after the base 
            # estimates are learned as increasing n in the discrete Bellman 
            # equation: R(s,a) + Sum_{i=0}^n gamma^i * Q(s_i, s_{i+1})
            # Thus, it only makes sense to allow it to reparam as many times 
            # as the walk length is (see early stopping condition). Any longer is
            # mostly a waste of time
            if gamma_active:
                distance_learned += 1
        
        if sample_size:    
            b = np.array_split(non_orphans, non_orphans.shape[0]//sample_size)
        else:
            b = [non_orphans]
        
        tot_loss = 0
        steps = 0
        for batch in b:
            s,a,r = Agent.episode(batch=batch, workers=8, quiet=True if verbose <= 1 else False)
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
            if not gamma_active:
                print("Activating discount factor")
            is_early_stopping = True
            
        e += 1
        
    # Set parameters back when returning agent
    Agent.num_walks = nw
    Agent.episode_len = wl
    Agent.epsion = eps
    
    return non_orphans

def example(sample_size=50, clip=None, reparam=40,
            gamma=0.99, nw=10, wl=10):
    import load_graphs as lg
    data = lg.load_cora()
    
    # Set up a basic agent 
    Agent = Q_Walk_Example(data, episode_len=wl, num_walks=nw, 
                           epsilon=lambda x : 0.95, gamma=0.999,
                           hidden=1000, one_hot=True, frozen=False)

    Encoder = RW_Encoder(Agent)
    
    non_orphans = fast_train_loop(Agent, sample_size=sample_size, clip=clip)    
    
    Encoder.compare_to_random(non_orphans)
    
        
if __name__ == '__main__':
    example(sample_size=None)