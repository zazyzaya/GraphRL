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
    def __init__(self, state_feats, max_actions, hidden=16):
        super().__init__()
        
        self.in_dim = state_feats
        self.hidden = hidden
        self.max_actions = max_actions
        
        self.lin = Sequential(Linear(self.in_dim, self.hidden), Tanh())
        self.out = Linear(self.hidden, self.max_actions)
        
    def forward(self, s):
        x = self.lin(s)
        return self.out(x)
    
    def reset_parameters(self):
        self.lin = Sequential(Linear(self.in_dim, self.hidden), Tanh())
        self.out = Linear(self.hidden, self.max_actions)

'''
Generates random walks based on some user specified reward function
'''
class Q_Walker():
    def __init__(self, data, gamma=0.99, epsilon=lambda x: 0.5, 
                episode_len=10, num_walks=10, hidden=64, network=None,
                beta=1):

        self.beta = beta
        self.state_feats=data.num_nodes
        self.hidden = hidden 

         # Is there really not a better way to do this?
        self.max_actions = max(
            degree(data.edge_index[0]).max().long().item(),
            degree(data.edge_index[1]).max().long().item()
        )
            
        # Just one layer for now. Technically this problem is supposed to be 
        # linearly seperable. But for more complex reward functions it may 
        # be beneficial to toss in another layer
        if network == None:
            self.qNet = Q_Network(self.state_feats, self.max_actions, hidden=hidden)
        else:
            self.qNet = network
        
        self.data = data
        self.episode_cnt = 0
        self.step = 0
        self.epsilon = epsilon
        self.gamma = gamma
        self.episode_len = episode_len
        self.num_walks = num_walks
        self.csr = None
        self.dense_adj = None

        self.cs = torch.nn.CosineSimilarity()

        # Used for fast tensor walks 
        self.action_map = None

    '''
    Used to build action map (compressed adj matrix)
    '''
    def _get_csr(self):
        if self.csr == None:
            edata = torch.full(self.data.edge_index[0].size(), 1, dtype=torch.long)
        
            # CSR matrices make it easier to calculate neighbors
            self.csr = csr_matrix((
                edata,
                (self.data.edge_index[0].numpy(), self.data.edge_index[1].numpy())), 
                shape=(self.data.num_nodes, self.data.num_nodes)
            )
            
        return self.csr

    '''  
    Used for Sorenson index reward function
    '''
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

    '''
    Some preprocessing tools for the graph. 
    To undo this, call repair_edge_index
    '''
    def remove_direction(self):
        self.ei_len = self.data.edge_index.size()[1]
        new_ei = torch.cat(
            [
                self.data.edge_index, 
                self.data.edge_index[torch.tensor([1,0]), :]
            ], dim=1)
        
        # Remove duped self loops
        dupes = new_ei[0,:]==new_ei[1,:]
        dupes[:self.ei_len] = False 
        new_ei = new_ei[:, ~dupes]

        self.data.edge_index = new_ei
        
    def repair_edge_index(self):
        self.data.edge_index = self.data.edge_index[
            :, 
            :self.ei_len
        ]
        
    '''
    Call this after all preprocessing and before training. Builds out the 
    -1-padded compressed adj list (not sparse) where each row corresponds to
    the node with that index, and each member of the row is a neighbor's nid 
    right-padded with -1 
    '''   
    def update_action_map(self):
        self.max_actions = max(
            degree(self.data.edge_index[0]).max().long().item(),
            degree(self.data.edge_index[1]).max().long().item()
        )
        
        self.action_map = torch.full(
            (self.data.num_nodes, self.max_actions),
            -1,
            dtype=torch.float
        )
        
        for i in range(self._get_csr().shape[0]):
            for j,idx in enumerate(self.csr[i].indices):
                self.action_map[i][j] = idx

        self.qNet.max_actions = self.max_actions
        self.qNet.reset_parameters()
        
    def parameters(self):
        return self.qNet.parameters()               
    
    '''
    Assumes a is a mask s.t. a == [0,0,1,0,...,0] corresponds to taking action 
    action_map[nid(s),2]
    '''
    def Q(self, s, a):
        est_rewards = self.qNet(s)
        return (est_rewards * a).sum(dim=1, keepdim=True)
    
    '''
    Return policy estimate for all neighbors (or 0 if no action at that index)
    '''
    def value_estimation(self, s, nids):
        with torch.no_grad():
            est = (self.qNet(s) * torch.clamp(self.action_map[nids]+1, max=1))
            
            # Discourage self loops
            est[self.action_map[nids] == nids.unsqueeze(-1)] = 0
        return est
        
    '''
    Given a tensor of states, with a tensor of nids corresponding to them, 
    return the next nid to walk to if following user defined policy 
    in ['egreedy', 'weighted', 'perfect'], else uses random
    '''        
    def policy(self, s, nids, strategy='weighted', return_nids=False, 
                return_both=False, value_estimates=None):
        
        if type(value_estimates) == type(None):
            value_estimates = self.value_estimation(s, nids)
        
        # If weighted_rand, it's pretty simple, just select the neighbor based on
        # the values the Q Net assigns each possible action 
        if strategy=='weighted':
            ret = torch.multinomial(
                F.relu(value_estimates) + torch.clamp(self.action_map[nids]+1, max=1)*1e-10, 
                num_samples=1
            ).squeeze()
            
        elif strategy=='egreedy':
            # Add just a bit to the neighbors so if all values are <=0 one will still be
            # selected to walk on 
            max_vals = torch.argmax(
                F.relu(value_estimates) + torch.clamp(self.action_map[nids]+1, max=1)*1e-10, 
                dim=1
            )
            
            # Then randomly select e% of the nodes to pick a random action for 
            rand_nodes = torch.rand((nids.size()[0],)) > self.epsilon(self.episode_cnt)
            rand_nids = nids[rand_nodes]
            
            # Make sure we only select neighbors as actions, and not the 
            # rows denoted -1 (for no neighbor in that slot)
            max_vals[rand_nodes] = torch.multinomial(
                torch.clamp(self.action_map[rand_nids]+1, max=1),
                num_samples=1
            ).squeeze()
                        
            ret = max_vals
            test = self.action_map[nids, ret]
            if test[test == -1].sum() > 0:
                print("Uh oh")
             
        # The same as e-greedy set e to 1.0. Useful for finding Q*(s,a)
        elif strategy=='perfect':
            ret = torch.argmax(
                F.relu(value_estimates) + torch.clamp(self.action_map[nids]+1, max=1)*1e-10, 
                dim=1
            )
            
        # If not recognized, assume random is the policy   
        else:
            ret = torch.multinomial(
                torch.clamp(self.action_map[nids]+1, max=1),
                num_samples=1
            ).squeeze()
            
        
        # If we need the node ids (for generating random walks)
        if return_nids:
            return self.action_map[nids, ret].long()
        
        # Otherwise, we can just use the pure action encodings (or both)
        else:
            encoded = torch.zeros((nids.size()[0], self.max_actions))
            encoded[range(nids.size()[0]), ret] = 1
            
            if not return_both:
                return encoded
            
            return self.action_map[nids, ret].long(), encoded
    
    '''
    Generate walks in parallel following learned policy
    '''
    def generate_walks(self, nids, strategy='random', silent=False, strings=True):
        walks = []
        og_nids = nids

        for _ in tqdm(range(self.num_walks), desc='Walks generated', disable=silent):
            nids = og_nids
            walk = [nids]
            s = self.state_transition(nids)

            print("Testing: delete me (rl_mod.py line 266)")
            print(nids[:10])
            
            with torch.no_grad():
                for _ in range(self.episode_len):
                    a_nids = self.policy(s,nids,strategy=strategy,return_nids=True)
                    s = self.state_transition(s,a=a_nids)
                    walk.append(a_nids)
                    nids = a_nids
            
            walk = torch.stack(walk, dim=1)
            if strings:
                walks += [[str(item.item()) for item in one_walk] for one_walk in walk]
            else:
                walks.append(walk)
        
        if not strings:
            walks = torch.cat(walks, dim=0)
        
        return walks 
    
    '''
    Generate an episode using a user defined strategy in ['random', 'weighted', 'egreedy', 'perfect']
    Builds several episodes to average the reward such that we find the total discounted
    expected value of the reward
    '''
    def generate_episode(self, batch=[], wl=None, nw=None, strategy='random'):
        self.episode_cnt += 1
        
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
        
        # Take one random step to learn from
        state = self.state_transition(batch)
        nids,action = self.policy(state,batch,strategy=strategy,return_both=True)
        
        # Starting state is next step
        s = self.state_transition(state, a=nids).repeat((nw,1))
        a = action.repeat((nw,1))
        nids = nids.repeat(nw)
        idx = torch.tensor(range(batch.size()[0])).repeat(nw)
        
        # And add reward for initial random step
        rewards = [self.reward(state.repeat((nw,1)), nids, s, batch.repeat(nw))]
          
        for _ in range(wl):
            # Select an action using the policy  
            a = self.policy(s,nids,strategy=strategy,return_nids=True)
            
            s_prime = self.state_transition(s,a=a)
            rewards.append(self.reward(s,a,s_prime,nids))
            
            nids = a
            s = s_prime
        
        rewards = torch.cat(rewards, dim=1)    
        
        discount = torch.tensor([[self.gamma ** i for i in range(wl+1)]])
        rewards = torch.mm(rewards, discount.T)
                
        # Then find average reward of random walks
        rewards = scatter_mean(rewards.T, idx).T
        return state, action, rewards
        
    '''
    Given a tensor of states, and a tensor of actions, returns the next states, s' that
    the module will transition to upon taking that action.
    
    By default, just returns the id of the next node to explore
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
    Must calculate some reward from the state, action and next state
    given the Q function. Recommend overwriting this with a user
    defined function

    s: tensor of |V|x1 states
    a: tensor of |V|x1 actions (index of action map)
    s_prime: tensor of |V|x1 states that will be transitioned to
    nid: node ids of members of s
    '''
    def reward(self,s,a,s_prime,nid):
        return self.struct_and_sim_reward(s,a,s_prime,nid)
    
    '''
    A few built-in reward functions to play with
    '''
    def min_similarity_reward(self, s,a,s_prime,nid):
        return s.logical_xor(s_prime).sum(dim=1, keepdim=True).float()
    
    def min_sim_reward(self, s,a,s_prime,nid):
        return 1-self.cs(self.data.x[nid], self.data.x[a]).unsqueeze(-1)    
    
    def max_sim_reward(self, s,a,s_prime,nid):
        sim = self.cs(self.data.x[nid],self.data.x[a]).unsqueeze(-1)
        
        # Punish returning walking toward "yourself"
        sim[sim==1] = 0
        
        return sim 

    '''
    While this does work, the numpy version that iterates
    through each row is slightly faster. Oh well
    '''
    def shared_neighbors_tensors(self,src,dst):
        dst[dst == -1] = -2
        
        # You've just gotta trust me on this one. This works, I promise
        num_shared = (
                src == dst.unsqueeze(-1).transpose(0,1)
            ).transpose(1,0).sum(axis=1).sum(axis=1,keepdim=True)
    
        return num_shared
    
    '''
    Most memory efficient way of doing this, but not as fast
    as the dense adj method
    '''
    def shared_neighbors_np(self,src,dst):
        dst[dst == -1] = -2
        
        return torch.tensor(
            [
                [
                    np.intersect1d(
                        src[i].numpy(),
                        dst[i].numpy()
                    ).shape[0]
                for i in range(src.size()[0])]
            ]
        ).T
        
    '''
    Fastest, but obvi most memory intensive way to accomplish
    this. For large graphs this will be intractable
    '''
    def shared_neighbors_dense_adj(self,src,dst):
        adj = self._get_dense_adj()
        return adj[src.long()].logical_and(adj[dst.long()]).sum(dim=1,keepdim=True)
    
    '''
    Calculates Sørenson Index:

        | N(u) && N(v) |
        ----------------
        |N(u)|  + |N(v)|
    
    Useful for well-clustered graphs and balances
    for outliers 
    '''
    def sorensen_reward(self, s, a, s_prime, nid):
        src = self.action_map[nid]
        dst = self.action_map[a]
        
        # So two non-edges don't count as a shared edge
        #dst[dst == -1] = -2
        
        ret = self.shared_neighbors_dense_adj(
            nid,a
        ).float().true_divide(
            (src >= 0).sum(axis=1, keepdim=True) + 
            (dst >= 0).sum(axis=1, keepdim=True)
        )
        
        ret[nid == a] = 0
        return ret

    def struct_and_sim_reward(self, s, a, s_prime, nid):
        # Sim reward is how similar node feats are
        # Sorenson index is how similar node structures are
        #return self.sorensen_reward(s,a,s_prime,nid)+self.max_sim_reward(s,a,s_prime,nid)
        
        if self.beta > 0:
            struct_r = 1-self.sorensen_reward(s,a,s_prime,nid)
        # Don't calculate if no need. Kind of expensive
        else:
            struct_r = torch.zeros((s.size()[0],1))
            
        if self.beta != 1:
            feat_r = self.max_sim_reward(s,a,s_prime,nid)
        else:
            feat_r = torch.zeros((s.size()[0],1))
            
        # Cubed to try to push close values further apart while keeping the sign 
        # the same (even though it's always positive... just in case)
        return torch.pow(
            (self.beta * struct_r) + ((1-self.beta) * feat_r), 
            1
        ) 

'''
Built-in training method for Q_walker
'''  
import time 
def train_loop(Agent, sample_size=None, lr=1e-3, verbose=1, 
                    early_stopping=0.05, epochs=800, nw=None, wl=None, 
                    strategy='egreedy', train_eps=0.8):
    # First, filter out the orphaned nodes we cant walk on 
    non_orphans = (degree(Agent.data.edge_index[0], num_nodes=Agent.data.num_nodes) > 1).nonzero()
    non_orphans = non_orphans.T.numpy()[0]
    
    # Save the original epsilon, and train using train_eps
    original_eps = Agent.epsilon
    Agent.epsilon = lambda x : train_eps

    opt = torch.optim.Adam(Agent.parameters(), lr=lr, weight_decay=1e-4)

    for e in range(epochs):
        start = time.time()
        np.random.shuffle(non_orphans)  

        if sample_size:  
            b = np.array_split(non_orphans, non_orphans.shape[0]//sample_size)
        else:
            b = [non_orphans]
            
        steps = 0
        tot_loss = 0
        opt.zero_grad()
        for batch in b:
            # Genreate a set of state, actions and discounted sum of rewards
            s,a,r = Agent.generate_episode(batch=batch, nw=nw, wl=wl, strategy=strategy)
            
            loss = F.mse_loss(Agent.Q(s,a), r)
            loss.backward()
            
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
    
    if verbose > 1:
        s,a,r = Agent.fast_episode_generation(batch=non_orphans, nw=1)
        print(Agent.Q(s,a)[:10])

    Agent.epsilon = original_eps
    return non_orphans

    
'''
Wrapper class for Q_walker to help convert walks to Word2Vec vectors
'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  
from sklearn.multiclass import OneVsRestClassifier as OVR
from sklearn.linear_model import LogisticRegression as LR
class RW_Encoder():
    def __init__(self, walker):
        self.walker = walker 
    
    def generate_walks(self, batch=[], strategy='weighted', 
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
                return self.walker.generate_walks(
                    batch,
                    strategy=strategy,
                    silent=silent,
                    strings=strings
                )
            
            walks = self.walker.generate_walks(
                    batch,
                    strategy=strategy,
                    silent=silent,
                    strings=True
            )
        
        return self.encode_nodes(walks, batch=batch, w2v_params=w2v_params, quiet=silent)
    
    def encode_nodes(self, walks, batch=[], w2v_params={}, quiet=False):
        # Make sure required params exist in w2v_params 
        required = dict(size=128, sg=1, workers=16)
        for k,v in required.items():
            if k not in w2v_params:
                w2v_params[k] = v    
           
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
        
