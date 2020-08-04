import torch
import random 
import numpy as np

from tqdm import tqdm
from joblib import Parallel, delayed
from abc import ABC, abstractmethod
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules import Bilinear, Linear
from torch_geometric.utils import degree
from scipy.sparse import csr_matrix

from gensim.models import Word2Vec

class Q_Network(torch.nn.Module):
    def __init__(self, state_feats, action_feats, hidden=16):
        super().__init__()
        
        self.lin = Linear(state_feats + action_feats, hidden)
        self.out = Linear(hidden, 1)
        
    def forward(self, s, a):
        x = torch.cat((s,a),dim=1)
        
        x = self.lin(x)
        x = self.out(F.relu(x))
        return F.relu(x)

'''
Generates random walks based on some user specified reward function
'''
class Q_Walker(ABC):
    def __init__(self, data, state_feats=None, action_feats=None,
                 gamma=0.99, epsilon=lambda x: 0.5, episode_len=10,
                 num_walks=10, hidden=64, network=None, edata=None):
        if state_feats == None:
            state_feats=data.x.size()[0]
        if action_feats == None:
            action_feats=data.x.size()[0]
           
        self.state_feats=state_feats
        self.action_feats=action_feats 
            
        # Just one layer for now. Technically this problem is supposed to be 
        # linearly seperable. But for more complex reward functions it may 
        # be beneficial to toss in another layer
        if network == None:
            self.qNet = Q_Network(self.state_feats, self.action_feats, hidden=hidden)
        else:
            self.qNet = network
            
        self.data = data
        
        self.episode_cnt = 1
        self.step = 0
        self.epsilon = epsilon
        self.gamma = gamma
        self.episode_len = episode_len
        self.num_walks = num_walks
        
        if edata == None:
            edata = torch.full(data.edge_index[0].size(), 1, dtype=torch.int)
        
        # CSR matrices make it easier to calculate neighbors
        self.csr = csr_matrix((
            edata,
            (data.edge_index[0].numpy(), data.edge_index[1].numpy())
        ))
        
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
    Given a state, and a chosen action, returns the next state, s' that
    the module will transition to upon taking that action.
    
    By default, just returns the one-hot encoding of the next node to explore
    '''
    def state_transition(self,s,a=None):
        one_hot_state = torch.zeros((self.state_feats), dtype=torch.float)
        
        if a == None:
            one_hot_state[s] = 1
        else:
            one_hot_state[a] = 1
            
        return one_hot_state
    
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
    Takes a node feature, and puts it at the a'th index,
    then runs it through the q net. 
    
    Can also accept a list of actions and run the Q(s,a) for 
    each of them (useful for finding Q(s,a)_max(a) )
    
    TODO make this work with more than one state at a time?
    '''
    def Q(self,s,a):    
        if len(s.size()) == 1:
            s = s.unsqueeze(dim=0)
        if s.size()[0] != a.size()[0]:
            s = s.expand(a.size()[0], s.size()[1])
            
        return self.qNet(s,a)
    
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
    def policy(self, s, nid, egreedy=True, return_value=False):    
        neighbors = self.csr[nid].indices
        
        # Can't do anything about orphaned nodes
        if len(neighbors) == 0:
            return nid
        
        actions = self.encode_actions(neighbors)
        
        if not return_value and egreedy and random.random() < self.epsilon(self.episode_cnt):
            action = np.random.choice(neighbors)    
        else:
            value_predictions = self.Q(s,actions)
            
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
            
        for _ in range(self.episode_len-1):    
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
                walk.append(nid)
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
    Runs one episode from each node and updates weights
    This needs to be seriously optimised
    '''
    def episode(self, batch=[], workers=-1, quiet=False):
        if len(batch) == 0:
            batch = list(range(self.data.x.size()[0]))
        
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
from sklearn.tree import DecisionTreeClassifier as DTree      
class RW_Encoder():
    def __init__(self, walker):
        self.walker = walker 
        
    def generate_walks(self, batch=[], workers=-1, random=False, 
                       egreedy=True, quiet=False, include_labels=False):
        if random:
            walk_fn = self.walker.random_walk
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
                
    
    
    def encode_nodes(self, batch=[], walks=None, random=False, w2v_params={}):
        # Make sure required params exist in w2v_params 
        required = dict(size=64, sg=1, workers=16)
        for k,v in required.items():
            if k not in w2v_params:
                w2v_params[k] = v    
        
        if walks == None:
            walks = self.generate_walks(batch=batch, workers=w2v_params['workers'], random=random)
            
        model = Word2Vec(walks, **w2v_params)
        
        idx_order = torch.tensor([int(i) for i in model.wv.index2entity], dtype=torch.long)
        y = self.walker.data.y[idx_order]
        X = model.wv.vectors
        
        return X, y
    
    def compare_to_random(self, batch):
        # Generate policy guided walks
        pX, py = self.encode_nodes(batch=batch)
        
        # Test against random walk embeddings
        rX, ry = self.encode_nodes(batch=batch, random=True)

        Xtr, Xte, ytr, yte = train_test_split(pX, py)
        dt = DTree()
        dt.fit(Xtr, ytr)
        yprime = dt.predict(Xte)
        print("Policy guided:")
        print(classification_report(yprime, yte))
        
        Xtr, Xte, ytr, yte = train_test_split(rX, ry)
        dt = DTree()
        dt.fit(Xtr, ytr)
        yprime = dt.predict(Xte)
        print("Random walk:")
        print(classification_report(yprime, yte))
            
    
# Example 
class Q_Walk_Feat_Similarity(Q_Walker):
    def __init__(self, data, gamma=0.99, epsilon=lambda x: 0.5, episode_len=10,
                 num_walks=10, hidden=64):
        
        # Set state and action feats to be dim of node features
        super().__init__(data, state_feats=data.x.size()[1], action_feats=data.x.size()[1], 
                         gamma=gamma, epsilon=epsilon, episode_len=episode_len, 
                         num_walks=num_walks, hidden=hidden)        
    
    # Want to minimize the similarity of nodes walked to
    def reward(self, s,a,s_prime,nid):
        return s.logical_xor(s_prime).sum()
    
    def state_transition(self, s,a=None):
        if a == None:
            return self.data.x[s]
        
        return self.data.x[a]
    
    def encode_actions(self, actions):
        return self.data.x[torch.tensor(actions, dtype=torch.long)]        

def example(sample_size=50, epochs=200):
    import load_cora as lc
    data = lc.load_data()
    
    # Get rid of nodes without neighbors
    non_orphans = (degree(data.edge_index[0], num_nodes=data.x.size()[0]) != 0).nonzero()
    non_orphans = non_orphans.T.numpy()[0]
    
    # For more complex policies, it's important to set gamma higher so future rewards 
    # are taken into account for the overall reward value. Here, we set gamma to 0 because
    # future steps don't really matter for the example this is basically a TD(1) agent 
    # with the set of rules we gave it
    Agent = Q_Walk_Feat_Similarity(data, episode_len=5, num_walks=100, epsilon=lambda x : 0.95, gamma=0.5)
    Encoder = RW_Encoder(Agent)
    
    # While training, set it to randomly walk to generate a diverse
    # set of states to learn from 
    Agent.epsilon = lambda x : 0
    Agent.gamma = 0.8
    
    opt = torch.optim.Adam(Agent.parameters(), lr=1e-2)
    for e in range(epochs):
        b = np.random.choice(non_orphans, (sample_size), replace=False)
        
        s,a,r = Agent.episode(batch=b, workers=8, quiet=True)
        opt.zero_grad()
        loss = F.mse_loss(Agent.Q(s,a), r)
        loss.backward()
        opt.step()
        
        print("[%d]: %0.4f" % (e, loss.item()))
        
    
    Agent.num_walks = 10
    Encoder.compare_to_random(non_orphans)
    
        
if __name__ == '__main__':
    example(epochs=100)