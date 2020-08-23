import sys 
import torch 
import numpy as np
import load_graphs as lg 

from tests import Be_Quiet
from rl_module_improved import Q_Walk_Simplified
from rl_module import RW_Encoder, fast_train_loop
from link_prediction import evaluate, generate_negative_samples, partition_data

class QW_Cora(Q_Walk_Simplified):
    def __init__(self, data, gamma=0.99, epsilon=lambda x: 0.5, episode_len=10,
                 num_walks=10, hidden=64, one_hot=False, network=None):
        super().__init__(data, gamma=gamma, epsilon=epsilon, episode_len=episode_len,
                         num_walks=num_walks, hidden=hidden, one_hot=one_hot, network=network)

        self.cs = torch.nn.CosineSimilarity()
    
    def remove_direction(self):
        self.data.edge_index = torch.cat(
            [
                self.data.edge_index, 
                self.data.edge_index[torch.tensor([1,0]), :]
            ], dim=1)
        
        pass
        
    def repair_edge_index(self):
        self.data.edge_index = self.data.edge_index[
            :, 
            :self.data.edge_index.size()[1]//2
        ]
        
    def reward(self, s,a,s_prime,nid):
        return self.max_sim_reward(s,a,s_prime,nid)
    
    def min_sim_reward(self, s,a,s_prime,nid):
        return 1-self.cs(self.data.x[nid], self.data.x[a]).unsqueeze(-1)    
    
    def max_sim_reward(self, s,a,s_prime,nid):
        sim = self.cs(self.data.x[nid],self.data.x[a]).unsqueeze(-1)
        
        # Punish returning walking toward "yourself"
        sim[sim==1] = 0
        
        return sim 
    

from sklearn.decomposition import PCA
def preprocess(X):
    decomp = PCA(n_components=32, random_state=1337)
    return torch.tensor(decomp.fit_transform(X.numpy()))    

def set_up_cora(wl=40, nw=10, epsilon=0.5, gamma=0.8, 
                early_stopping=0.01, epochs=50):
    data = lg.load_cora()
    
    print("Splitting edges into train and test sets")
    partition_data(data, percent_hidden=0.10)
    
    print("Running PCA on node features")
    data.x = preprocess(data.x)
    
    # Set up a basic agent 
    Agent = QW_Cora(data, episode_len=wl, num_walks=nw, 
                           epsilon=lambda x : epsilon, gamma=gamma,
                           hidden=1028, one_hot=True)
    
    # New experiment: train on bidirectional graph
    Agent.remove_direction()
    all_edges = Agent.data.edge_index
    
    # Note by repeating train_mask, agent still won't learn about (u,v)
    # if (v,u) is masked. The edges it learns on are the same, they are 
    # just allowed to be bidirectional for random walks
    Agent.data.edge_index = data.edge_index[:, data.train_mask.repeat(2)]
    
    # Make sure edges are hidden when querying for neighbors
    Agent.update_action_map()
    
    Encoder = RW_Encoder(Agent)
    non_orphans = fast_train_loop(
        Agent,
        verbose=1,
        early_stopping=early_stopping,
        nw=min(nw, 5),
        wl=min(wl, 20),
        epochs=epochs,
        sample_size=None,
        minibatch_bootstrap=False
    )
    
    data.edge_index = all_edges
    return Agent, Encoder, non_orphans

def set_up_citeseer(wl=10, nw=10, epsilon=0.5, gamma=0.99, early_stopping=0.03,
                    epochs=50):
    data = lg.load_citeseer()
    
    print("Splitting edges into train and test sets")
    partition_data(data, percent_hidden=0.10)
    
    print("Running PCA on node features")
    data.x = preprocess(data.x)
    
    # Set up a basic agent 
    Agent = QW_Cora(data, episode_len=wl, num_walks=nw, 
                           epsilon=lambda x : epsilon, gamma=gamma,
                           hidden=1028, one_hot=True)
    
    # New experiment: train on bidirectional graph
    Agent.remove_direction()
    all_edges = Agent.data.edge_index
    
    # Note by repeating train_mask, agent still won't learn about (u,v)
    # if (v,u) is masked. The edges it learns on are the same, they are 
    # just allowed to be bidirectional for random walks
    Agent.data.edge_index = data.edge_index[:, data.train_mask.repeat(2)]
    Agent.update_action_map()
    
    Encoder = RW_Encoder(Agent)
    non_orphans = fast_train_loop(
        Agent,
        verbose=1,
        early_stopping=early_stopping,
        nw=min(nw, 5),
        wl=min(wl, 10),
        epochs=epochs,
        sample_size=128,
        minibatch_bootstrap=False
    )
    
    data.edge_index = all_edges
    return Agent, Encoder, non_orphans

def test_agent(trials=1, set_up_fn=set_up_cora):
    Agent, Encoder, non_orphans = set_up_fn()
    
    w2v = {'window': 10, 'size': 128}
    
    rw = []
    pww = []
    pgw = []
    
    for _ in range(trials):
        rw.append(Encoder.generate_walks_fast(
            batch=non_orphans, 
            strategy='random', 
            encode=True, 
            w2v_params=w2v,
            silent=False
        )[0])
        
        pww.append(Encoder.generate_walks_fast(
            batch=non_orphans, 
            strategy='weighted', 
            encode=True, 
            w2v_params=w2v,
            silent=False
        )[0])
        pgw.append(Encoder.generate_walks_fast(
            batch=non_orphans, 
            strategy='egreedy', 
            encode=True, 
            w2v_params=w2v,
            silent=False
        )[0])
    
    # Make edges directed again
    Agent.repair_edge_index()
    data = Agent.data
    
    print("Generating negative samples")
    neg = generate_negative_samples(data, data.test_mask.size()[0])
    pos = data.edge_index[:, data.test_mask].T
    
    # Quickly find standard error as did 
    stderr = lambda x : x.std()/(trials ** 0.5)
    
    print()
    print("Running link prediction (avg of %d indipendant runs)" % trials)
    for bop in ['hadamard', '   l1   ', '   l2   ']:#, '   avg  ']:
        print("-"* 10 + ' ' + bop + ' ' + '-'*10)
        
        # Remove spaces lazilly added for formatting
        bop = bop.strip()
        
        rw_score = np.array([evaluate(w, pos, neg, bin_op=bop) for w in rw])
        print("\tRandom walks:\t%0.4f (+/-) %0.03f" % (rw_score.mean(), stderr(rw_score)))
        
        pww_score = np.array([evaluate(w, pos, neg, bin_op=bop) for w in pww])
        print("\tPolicy walks:\t%0.4f (+/-) %0.03f" % (pww_score.mean(), stderr(pww_score)))
        
        pgw_score = np.array([evaluate(w, pos, neg, bin_op=bop) for w in pgw])
        print("\tE-greedy walks:\t%0.4f (+/-) %0.03f" % (pgw_score.mean(), stderr(pgw_score)))
    

test_agent(trials=5)    

def test_epsilon(wl=40, nw=10, gamma=0.75, early_stopping=0.05):
    Agent, Encoder, non_orphans = set_up_cora(
        wl=wl, nw=nw, gamma=gamma, early_stopping=early_stopping
    )
    
    print("Generating negative samples")
    Agent.repair_edge_index()
    data = Agent.data
     
    neg = generate_negative_samples(data, data.train_mask.size()[0])
    pos = data.edge_index[:, data.test_mask].T
    
    print()
    print("Testing different epsilons")
    og = sys.stdout 
    for eps in range(0, 105, 5):
        sys.stdout = Be_Quiet()
        
        eps = eps / 100 
        Agent.epsilon = lambda x : eps
        walks, _ = Encoder.generate_walks_fast(
            batch=non_orphans, 
            strategy='egreedy',
            encode=True,
            silent=True
        )
        
        sys.stdout = og 
        print("-"*10 + ' Epsilon: %0.2f ' % eps + "-"*10)
        for op in ['hadamard', 'l1\t', 'l2\t']:
            score = evaluate(walks, pos, neg, bin_op=op.strip())
            print(op + ": %0.4f" % score) 

#test_epsilon()