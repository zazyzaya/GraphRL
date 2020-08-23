import torch 
import random 

from tqdm import tqdm
from rl_module import Q_Walker
from torch_scatter import scatter_mean
from torch_geometric.utils import degree
from torch.nn import Sequential, Linear, Tanh
from torch.nn import functional as F

class Q_Network_No_Action(torch.nn.Module):
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

class Q_Walker_Improved(Q_Walker):
    def __init__(self, data, state_feats=None, gamma=0.99, epsilon=lambda x: 0.5, 
                 episode_len=10, num_walks=10, hidden=64, network=None, edata=False,
                 frozen=True):
        
        # Is there really not a better way to do this?
        self.max_actions = max(
            degree(data.edge_index[0]).max().long().item(),
            degree(data.edge_index[1]).max().long().item()
        )
        
        if not state_feats:
            state_feats = data.num_nodes
        
        if not network:
            network = Q_Network_No_Action(state_feats, self.max_actions, hidden)
        
        super().__init__(data, state_feats=state_feats, action_feats=0, 
                        gamma=gamma, epsilon=epsilon, episode_len=episode_len,
                        num_walks=num_walks, hidden=hidden, network=network, 
                        edata=edata, frozen=True)
        
        self.action_map = None
        
    '''
    Call this after all preprocessing and before training
    '''   
    def update_action_map(self):
        self.action_map = torch.full(
            (self.data.num_nodes, self.max_actions),
            -1,
            dtype=torch.float
        )
        
        for i in range(self._get_csr().shape[0]):
            for j,idx in enumerate(self.csr[i].indices):
                self.action_map[i][j] = idx
        
    '''
    Assumes a is a mask s.t. a == [0,0,1,...] corresponds to taking action 
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
        return est
        
    '''
    Given a tensor of states, with a tensor of nids corresponding to them, 
    return the next nid to walk to if following weighted rand or e greedy policy
    '''        
    def policy(self, s, nids, strategy='weighted', return_nids=False, return_both=False):
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
             
        # If neither, assume random is the policy   
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
    Generate walks in parallel; much faster than Agent.episode
    '''
    def fast_walks(self, nids, strategy='random', silent=False, strings=True):
        walks = []
        node = nids
        for _ in tqdm(range(self.num_walks), desc='Walks generated', disable=silent):
            walk = [nids]
            s = self.state_transition(nids)
            
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
    Generate an episode using a user defined strategy in ['random', 'weighted', 'policy']
    Builds several episodes to average the reward such that we find the total discounted
    expected value of the reward
    '''
    def fast_episode_generation(self, batch=[], wl=None, nw=None, strategy='random'):
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
        
        discount = torch.tensor([self.gamma ** i for i in range(wl+1)])
        rewards *= discount
        rewards = rewards.sum(dim=1, keepdim=True)
        
        # Then find average reward of random walks
        rewards = scatter_mean(rewards.T, idx).T
        return state, action, rewards
            
            
class Q_Walk_Simplified(Q_Walker_Improved):
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