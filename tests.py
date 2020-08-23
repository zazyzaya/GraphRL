import torch 
import load_graphs as lg
import pickle as pkl
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score 
from sklearn.multiclass import OneVsRestClassifier as OVR
from sklearn.linear_model import LogisticRegression as LR
from torch_geometric.utils import degree
from rl_module import Q_Walk_Example, RW_Encoder, train_loop, fast_train_loop

class QW_Cora(Q_Walk_Example):
    def __init__(self, data, gamma=0.99, epsilon=lambda x: 0.5, episode_len=10,
                 num_walks=10, hidden=64, one_hot=False, network=None):
        super().__init__(data, gamma=gamma, epsilon=epsilon, episode_len=episode_len,
                         num_walks=num_walks, hidden=hidden, one_hot=one_hot, network=network)
        
        self.max_degree = degree(data.edge_index[0]).max()
        self.cs = torch.nn.CosineSimilarity()
        
    
    def reward(self, s,a,s_prime,nid):
        return self.cs(self.data.x[nid],self.data.x[a]).unsqueeze(-1)
    

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
    
    Encoder.compare_to_random(non_orphans, w2v_params={'size': 128}, fast_walks=True)
    return Agent
  
from sklearn.decomposition import PCA
def preprocess(X):
    decomp = PCA(n_components=256, random_state=1337)
    return torch.tensor(decomp.fit_transform(X.numpy()))

def cora_fast(wl=5, nw=10, gamma=0.99, eps=0.95, early_stopping=0.1):
    print("Testing the CORA dataset")
    data = lg.load_cora()
    
    print("Running PCA on features")
    data.x = preprocess(data.x)
    
    # Set up a basic agent 
    Agent = QW_Cora(
        data, episode_len=wl, num_walks=nw, 
        epsilon=lambda x : eps, gamma=gamma,
        hidden=1028, one_hot=True
    )

    Encoder = RW_Encoder(Agent)
    
    non_orphans = fast_train_loop(
        Agent, 
        verbose=0, 
        early_stopping=early_stopping, 
        epochs=200,
        nw=3
    )    
    
    Encoder.compare_to_random(
        non_orphans, 
        w2v_params={'size': 128}, 
        fast_walks=True
    )
    
    return Agent

def test_cora():  
    Agent = cora_fast(
        gamma=0.99999, 
        eps=0.75, 
        nw=10, 
        wl=10,
        early_stopping=0.01
    )

class Be_Quiet():
    def __init__(self):
        pass
    def flush(self, **kwargs):
        pass
    def write(self, *args):
        pass
    
if __name__ == '__main__':
    test_cora()

import sys 
def test_epsilon_cora():
    data = lg.load_cora()
    
    # Set up a basic agent 
    Agent = QW_Cora(
        data, episode_len=10, num_walks=10, 
        epsilon=lambda x : 0, gamma=0.99,
        hidden=1028, one_hot=True
    )

    Encoder = RW_Encoder(Agent)
    
    non_orphans = fast_train_loop(
        Agent, 
        verbose=0, 
        early_stopping=0.05, 
        epochs=200
    )    
    
    og = sys.stdout 
    
    non_orphans = torch.tensor(non_orphans)
    estimator = lambda : LR(n_jobs=16, max_iter=1000)
    y_trans = lambda y : y.argmax(axis=1)
    
    for i in range(0,105,5):
        Agent.epsilon = lambda x : i/100
        
        sys.stdout = Be_Quiet()
        walks = Agent.fast_walks(non_orphans, silent=True)
        
        X,y = Encoder.encode_nodes(
            batch=non_orphans, 
            walks=walks,
            w2v_params={'size': 128}
        )
        
        lr = estimator()
        Xtr, Xte, ytr, yte = train_test_split(X, y_trans(y))
        lr.fit(Xtr, ytr)
        yprime = lr.predict(Xte)    
    
        acc = accuracy_score(yte, yprime)
        
        sys.stdout = og
        print('*' * int(acc * 50), end='\t')
        print('(%0.3f) Epsilon: %d' % (acc, i))

#test_epsilon_cora()

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