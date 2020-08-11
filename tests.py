import torch 
import load_graphs as lg
import pickle as pkl
import torch.nn.functional as F

from torch_geometric.utils import degree
from rl_module import Q_Walk_Simplified, RW_Encoder, train_loop 

class QW_Cora(Q_Walk_Simplified):
    def __init__(self, data, gamma=0.99, epsilon=lambda x: 0.5, episode_len=10,
                 num_walks=10, hidden=64, one_hot=False, network=None):
        super().__init__(data, gamma=gamma, epsilon=epsilon, episode_len=episode_len,
                         num_walks=num_walks, hidden=hidden, one_hot=one_hot, network=network)
        
        self.max_degree = degree(data.edge_index[0]).max()
        
    def reward(self, s,a,s_prime,nid):
        return super().min_degree_reward(s,a,s_prime,nid)
    

def cora(sample_size=50, clip=None, reparam=40,
            gamma=0.99, nw=10, wl=5, epsilon=0.95):
    print("Testing the CORA dataset")
    data = lg.load_cora()
    
    # Set up a basic agent 
    Agent = QW_Cora(data, episode_len=wl, num_walks=nw, 
                           epsilon=lambda x : epsilon, gamma=gamma,
                           hidden=1028, one_hot=True)

    Encoder = RW_Encoder(Agent)
    
    non_orphans = train_loop(Agent, sample_size=sample_size, early_stopping=0.01,
                             reparam=0.05, clip=clip, decreasing_param=False,
                             verbose=0, gamma_depth=10)    
    
    Encoder.compare_to_random(non_orphans, w2v_params={'size': 128})
    return Agent
  

def test_cora():  
    # Using original n2v parameters
    Agent = cora(sample_size=800, gamma=0.9999, epsilon=0.5, reparam=600,
                nw=10, wl=80)

test_cora()

'''
Need to seriously improve how single state-action-reward tuples
are generated before this is feasible

class QW_Blog(Q_Walk_Simplified):
    def __init__(self, data, gamma=0.99, epsilon=lambda x: 0.5, episode_len=10,
                 num_walks=10, hidden=64, one_hot=False, network=None):
        super().__init__(data, gamma=gamma, epsilon=epsilon, episode_len=episode_len,
                         num_walks=num_walks, hidden=hidden, one_hot=one_hot, network=network)
        
        self.max_nodes = 9400 # For some reason the degree fn breaks so just hard coding it
        
    def reward(self, s,a,s_prime,nid):
        return torch.log(
            torch.tensor([self.max_nodes / self.csr[a].indices.shape[0]])
        )

def blog(sample_size=50, clip=None, reparam=40,
            gamma=0.99, nw=10, wl=5, epsilon=0.95):
    print("Testing the Blog Catalogue dataset")
    data = lg.load_blog()
    
    # Set up a basic agent 
    Agent = QW_Blog(data, episode_len=wl, num_walks=nw, 
                           epsilon=lambda x : epsilon, gamma=gamma,
                           hidden=1028, one_hot=True)

    Encoder = RW_Encoder(Agent)
    
    non_orphans = train_loop(Agent, sample_size=sample_size, early_stopping=0.05,
                             reparam=0.05, clip=clip, decreasing_param=False,
                             gamma_depth=10)    
    
    Encoder.compare_to_random(non_orphans, w2v_params={'size': 128}, multiclass=True)
    return Agent

def test_blog():
    # Using original n2v parameters
    Agent = blog(sample_size=100, gamma=0.99, epsilon=0.5, reparam=0.05,
                 nw=10, wl=80)
    
test_blog()
'''