import torch 
import load_graphs as lg
import pickle as pkl
import torch.nn.functional as F
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score 
from sklearn.multiclass import OneVsRestClassifier as OVR
from sklearn.linear_model import LogisticRegression as LR
from torch_geometric.utils import degree
from rl_module import RW_Encoder, train_loop, fast_train_loop
from rl_module_improved import Q_Walk_Simplified

class QW_Cora(Q_Walk_Simplified):
    def __init__(self, data, gamma=0.99, epsilon=lambda x: 0.5, episode_len=10,
                 num_walks=10, hidden=64, one_hot=False, network=None, beta=1):
        super().__init__(data, gamma=gamma, epsilon=epsilon, episode_len=episode_len,
                         num_walks=num_walks, hidden=hidden, one_hot=one_hot, network=network)

        self.beta = beta
    
    def reward(self, s,a,s_prime,nid):
        # Sim reward is how similar node feats are
        # Sorenson index is how similar node structures are
        #return self.sorensen_reward(s,a,s_prime,nid)+self.max_sim_reward(s,a,s_prime,nid)
        
        if self.beta > 0:
            struct_r = self.sorensen_reward(s,a,s_prime,nid)
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
            3
        )

default_agent_params = dict(
    episode_len=10,
    num_walks=5,
    epsilon=lambda x : 0.95,
    gamma=1-1e-3,
    hidden=2048,
    one_hot=True,
    beta=0.25
)

train_settings = dict(
    verbose=1,
    early_stopping=0.001,
    nw=5,
    wl=20,
    epochs=200,
    sample_size=None,
    lr=1e-5
)

def generic_test(data, trials, num_tests, max_eps, agent_params=default_agent_params,
                 train_settings=train_settings):
    
    print("Running PCA on node features")
    data.x = preprocess(data.x)
    
    Agent = QW_Cora(data, **agent_params)
    Agent.remove_direction()
    Agent.update_action_map()
    
    Encoder = RW_Encoder(Agent)
    
    non_orphans = fast_train_loop(Agent, strategy='egreedy', **train_settings)
    
    print(Agent.qNet(Agent.state_transition(torch.tensor([[0]])))
          * torch.clamp(Agent.action_map[0]+1, max=1))  
    
    print(" nw=%d\n wl=%d\n beta=%0.2f\n gamma=%f" % (
            train_settings['nw'], 
            train_settings['wl'], 
            agent_params['beta'], 
            agent_params['gamma']
        )
    )
    for i in range(0,nw+1,nw//num_tests):
        print()
        eps = (i/nw) * max_eps
        Agent.epsilon = lambda x : eps/100
        all_stats = {'acc':[], 'prec':[], 'rec':[], 'f1':[]}
        for _ in tqdm(range(trials), desc="%d%% random walks" % (100-eps)):
            #X,y = Encoder.generate_mixed_walks(non_orphans, mix_ratio=i/nw)
            X,y = Encoder.generate_walks_fast(non_orphans, strategy='egreedy', silent=True, encode=True)
            stats = Encoder.get_accuracy_report(X,y,test_size=0.25)
            
            all_stats['acc'].append(stats['accuracy'])
            all_stats['prec'].append(stats['weighted avg']['precision'])
            all_stats['rec'].append(stats['weighted avg']['recall'])
            all_stats['f1'].append(stats['weighted avg']['f1-score'])
            
        df = pd.DataFrame(all_stats)
        df = pd.DataFrame([df.mean(), df.sem()])
        df.index = ['mean', 'stderr']
        print(df)
    
    return Agent

def citeseer(gamma=1-1e-3, nw=10, wl=7, epsilon=0.99, trials=10, 
         num_tests=5, max_eps=20, beta=0.25):
    
    print("Testing the Citeseer dataset")
    data = lg.load_citeseer()
    
    agent_params = dict(
        episode_len=wl,
        num_walks=nw,
        epsilon=lambda x : epsilon,
        gamma=gamma,
        hidden=2048,
        one_hot=True,
        beta=beta
    )
    
    global train_settings
    train_settings['nw'] = min(nw, 5)
    train_settings['wl'] = min(wl, 20)
    
    return generic_test(
        data, 
        trials, 
        num_tests,
        max_eps,
        agent_params=agent_params, 
        train_settings=train_settings
    )

def cora(gamma=1-1e-3, nw=10, wl=7, epsilon=0.99, trials=10, 
         num_tests=5, max_eps=20, beta=0.25):
    
    print("Testing the CORA dataset")
    data = lg.load_cora()
    
    agent_params = dict(
        episode_len=wl,
        num_walks=nw,
        epsilon=lambda x : epsilon,
        gamma=gamma,
        hidden=2048,
        one_hot=True,
        beta=beta
    )
    
    train_settings = train_settings
    train_settings['nw'] = min(nw, 5)
    train_settings['wl'] = min(wl, 20)
    
    return generic_test(
        data, 
        trials, 
        num_tests,
        max_eps,
        agent_params=agent_params, 
        train_settings=train_settings
    ) 

from sklearn.decomposition import PCA
def preprocess(X):
    decomp = PCA(n_components=256, random_state=1337)
    return torch.tensor(decomp.fit_transform(X.numpy()))

class Be_Quiet():
    def __init__(self):
        pass
    def flush(self, **kwargs):
        pass
    def write(self, *args):
        pass

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

if __name__ == '__main__':
    citeseer()