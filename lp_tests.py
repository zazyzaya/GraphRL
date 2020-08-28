import sys 
import torch 
import numpy as np
import load_graphs as lg 

from gcn_experiments import get_gcn_embeddings
from tests import Be_Quiet
from rl_module_improved import Q_Walk_Simplified
from rl_module import RW_Encoder, fast_train_loop
from link_prediction import evaluate, generate_negative_samples, partition_data

class QW_Cora(Q_Walk_Simplified):
    def __init__(self, data, gamma=0.99, epsilon=lambda x: 0.5, episode_len=10,
                 num_walks=10, hidden=64, one_hot=False, network=None, beta=1):
        super().__init__(data, gamma=gamma, epsilon=epsilon, episode_len=episode_len,
                         num_walks=num_walks, hidden=hidden, one_hot=one_hot, network=network)
        self.beta = beta
        
    def reward(self, s,a,s_prime,nid):
        # Sim reward is how similar node feats are
        # Sorenson index is how similar node structures are
        if self.beta > 0:
            struct_r = self.sorensen_reward(s,a,s_prime,nid)
        # Don't calculate if no need. Kind of expensive
        else:
            struct_r = torch.zeros((s.size()[0],1))
            
        if self.beta != 1:
            feat_r = self.max_sim_reward(s,a,s_prime,nid)
        else:
            feat_r = torch.zeros((s.size()[0],1))
            
        return (self.beta * struct_r) + ((1-self.beta) * feat_r)

from sklearn.decomposition import PCA
def preprocess(X):
    decomp = PCA(n_components=256, random_state=1337)
    return torch.tensor(decomp.fit_transform(X.numpy()))    

def train_agent(Agent, kwargs):
    print("Training on random walks")
    non_orphans = fast_train_loop(
        Agent,
        **kwargs
    )
    
    print("Training on weighted walks")
    fast_train_loop(
        Agent,
        strategy='weighted',
        **kwargs
    )
    
    return non_orphans

def set_up_cora(wl=40, nw=10, epsilon=0.6, gamma=0.9999, 
                early_stopping=0.01, epochs=50, beta=1, 
                retrain=False):
    
    data = lg.load_cora()
    
    print("Splitting edges into train and test sets")
    partition_data(data, percent_hidden=0.10)
    
    print("Running PCA on node features")
    data.x = preprocess(data.x)
    
    # Set up a basic agent 
    Agent = QW_Cora(data, episode_len=wl, num_walks=nw, 
                           epsilon=lambda x : epsilon, gamma=gamma,
                           hidden=512, one_hot=True, beta=beta)
    
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
    
    train_settings = dict(
        verbose=1,
        early_stopping=early_stopping,
        nw=min(nw, 5),
        wl=min(wl, 20),
        epochs=epochs//2,
        sample_size=None,
        minibatch_bootstrap=False
    )
    
    non_orphans = train_agent(Agent, train_settings)
    
    data.edge_index = all_edges
    return Agent, Encoder, non_orphans

def set_up_citeseer(wl=10, nw=10, epsilon=0.5, gamma=0.99, early_stopping=0.03,
                    epochs=50, beta=1):
    data = lg.load_citeseer()
    
    print("Splitting edges into train and test sets")
    partition_data(data, percent_hidden=0.10)
    
    print("Running PCA on node features")
    data.x = preprocess(data.x)
    
    # Set up a basic agent 
    Agent = QW_Cora(data, episode_len=wl, num_walks=nw, 
                           epsilon=lambda x : epsilon, gamma=gamma,
                           hidden=1028, one_hot=True, beta=beta)
    
    # New experiment: train on bidirectional graph
    Agent.remove_direction()
    all_edges = Agent.data.edge_index
    
    # Note by repeating train_mask, agent still won't learn about (u,v)
    # if (v,u) is masked. The edges it learns on are the same, they are 
    # just allowed to be bidirectional for random walks
    Agent.data.edge_index = data.edge_index[:, data.train_mask.repeat(2)]
    Agent.update_action_map()
    
    kwargs = dict(
        verbose=1,
        early_stopping=early_stopping,
        nw=min(nw, 5),
        wl=min(wl, 20),
        epochs=epochs//2,
        sample_size=None,
        minibatch_bootstrap=False
    )
    
    Encoder = RW_Encoder(Agent)
    train_agent(Agent, kwargs)
    
    data.edge_index = all_edges
    return Agent, Encoder, non_orphans

def test_gcn_agent(trials=1, set_up_fn=set_up_cora, include_random=False):
    
    Agent, Encoder, non_orphans = set_up_fn(
        wl=10, nw=40, 
        epsilon=0.6, 
        epochs=50, 
        gamma=0.9999
    )
    
    rw = []
    pww = []
    pgw = []
    
    max_samples=2**18
    epochs=25
    
    gcn_settings=dict(hidden=128, out=128)
    
    encoder_kwargs=dict(
        encode=False,
        strings=False,
        silent=False
    )
    embedder_kwargs=dict(
        as_numpy=False,
        max_samples=max_samples,
        epochs=epochs,
        gcn_kwargs=gcn_settings
    )
    
    # Remask the edges before giving it to GCN
    all_edges = Agent.data.edge_index
    Agent.data.edge_index = Agent.data.edge_index[:, Agent.data.train_mask.repeat(2)]
    
    for _ in range(trials):
        if include_random:
            rw.append(
                get_gcn_embeddings(
                    Agent.data,
                    Encoder.generate_walks_fast(
                        batch=non_orphans, 
                        strategy='random', 
                        **encoder_kwargs
                    ),
                    text='random',
                    **embedder_kwargs
                )
            )
            
        pww.append(
            get_gcn_embeddings(
                Agent.data,
                Encoder.generate_walks_fast(
                    batch=non_orphans, 
                    strategy='weighted', 
                    **encoder_kwargs
                ),
                text='weighted',
                **embedder_kwargs
            )
        )
        pgw.append(
            get_gcn_embeddings(
                Agent.data,
                Encoder.generate_walks_fast(
                    batch=non_orphans, 
                    strategy='egreedy', 
                    **encoder_kwargs
                ),
                text='e-greedy',
                **embedder_kwargs
            )
        )
    
    # Make edges directed again
    Agent.data.edge_index = all_edges
    Agent.repair_edge_index()
    data = Agent.data
    
    print("Generating negative samples")
    neg = generate_negative_samples(data, data.test_mask.size()[0])
    pos = data.edge_index[:, data.test_mask].T
    
    # Quickly find standard error as did original paper
    stderr = lambda x : x.std()/(trials ** 0.5)
    
    print()
    print("Running link prediction (avg of %d indipendant runs)" % trials)
    for bop in ['hadamard', '   l1   ', '   l2   ']:#, '   avg  ']:
        print("-"* 10 + ' ' + bop + ' ' + '-'*10)
        
        # Remove spaces lazilly added for formatting
        bop = bop.strip()
        
        if include_random:
            rw_score = np.array([evaluate(w, pos, neg, bin_op=bop) for w in rw])
            print("\tRandom walks:\t%0.4f (+/-) %0.03f" % (rw_score.mean(), stderr(rw_score)))
            
        pww_score = np.array([evaluate(w, pos, neg, bin_op=bop) for w in pww])
        print("\tPolicy walks:\t%0.4f (+/-) %0.03f" % (pww_score.mean(), stderr(pww_score)))
        
        pgw_score = np.array([evaluate(w, pos, neg, bin_op=bop) for w in pgw])
        print("\tE-greedy walks:\t%0.4f (+/-) %0.03f" % (pgw_score.mean(), stderr(pgw_score)))
     
#test_gcn_agent(trials=1, include_random=True)       

def test_link_pred(trials=1, set_up_fn=set_up_cora, include_random=False):    
    Agent, Encoder, non_orphans = set_up_fn()
    
    w2v = {'window': 10, 'size': 128, 'iter':5}
    
    rw = []
    pww = []
    pgw = []
    
    for _ in range(trials):
        if include_random:
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
    
    # Quickly find standard error as did original paper
    stderr = lambda x : x.std()/(trials ** 0.5)
    
    print()
    print("Running link prediction (avg of %d indipendant runs)" % trials)
    for bop in ['hadamard', '   l1   ', '   l2   ']:#, '   avg  ']:
        print("-"* 10 + ' ' + bop + ' ' + '-'*10)
        
        # Remove spaces lazilly added for formatting
        bop = bop.strip()
        
        if include_random:
            rw_score = np.array([evaluate(w, pos, neg, bin_op=bop) for w in rw])
            print("\tRandom walks:\t%0.4f (+/-) %0.03f" % (rw_score.mean(), stderr(rw_score)))
            
        pww_score = np.array([evaluate(w, pos, neg, bin_op=bop) for w in pww])
        print("\tPolicy walks:\t%0.4f (+/-) %0.03f" % (pww_score.mean(), stderr(pww_score)))
        
        
        pgw_score = np.array([evaluate(w, pos, neg, bin_op=bop) for w in pgw])
        print("\tE-greedy walks:\t%0.4f (+/-) %0.03f" % (pgw_score.mean(), stderr(pgw_score)))
        
        print()
        if include_random:
            print(rw_score)
        print(pww_score)
        print(pgw_score)

#test_link_pred(trials=3,include_random=True)

def test_epsilon(wl=40, nw=10, gamma=0.99, early_stopping=0.05, trials=5):
    Agent, Encoder, non_orphans, _ = set_up_cora(
        wl=wl, nw=nw, gamma=gamma, 
        early_stopping=early_stopping
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
        
        walks = []
        eps = eps / 100 
        Agent.epsilon = lambda x : eps
        
        for _ in range(trials):
            walks.append(Encoder.generate_walks_fast(
                batch=non_orphans, 
                strategy='egreedy',
                encode=True,
                silent=True
            )[0])
        
        stderr = lambda x : x.std()/(trials ** 0.5)
        
        sys.stdout = og 
        print("-"*10 + ' Epsilon: %0.2f ' % eps + "-"*10)
        for op in ['hadamard', 'l1\t', 'l2\t']:
            score = np.array([evaluate(walk, pos, neg, bin_op=op.strip()) for walk in walks])
            print(op + ": %0.4f (+/-) %0.3f" % (score.mean(), stderr(score))) 
            

def test_mixture(wl=40, nw=10, gamma=0.9999, early_stopping=0.05, trials=5):
    Agent, Encoder, non_orphans= set_up_cora(
        wl=wl, nw=nw, gamma=gamma, 
        early_stopping=early_stopping,
        epochs=100, beta=0.5
    )
    
    print("Generating negative samples")
    Agent.repair_edge_index()
    data = Agent.data
     
    neg = generate_negative_samples(data, data.train_mask.size()[0])
    pos = data.edge_index[:, data.test_mask].T
    
    print()
    print("Testing different proportions of random walks")
    og = sys.stdout
    for prop_rand in range(0,nw+1,2):
        rw = prop_rand
        pw = nw-prop_rand
        
        embeddings = []
        sys.stdout = Be_Quiet()
        for _ in range(trials):
            if rw == 0:
                walks = []
            else:
                Agent.num_walks = rw
                walks = Encoder.generate_walks_fast(
                    batch=non_orphans, 
                    strategy='random',
                    encode=False,
                    strings=True,
                    silent=True
                )
            
            if pw > 0:
                Agent.num_walks = pw
                walks = Encoder.generate_walks_fast(
                    batch=non_orphans, 
                    strategy='weighted',
                    encode=False,
                    strings=True,
                    silent=True
                )
                
            embeddings.append(
                Encoder.encode_nodes(
                    batch=non_orphans,
                    walks=walks
                )[0]
            )
                
        
        stderr = lambda x : x.std()/(trials ** 0.5)
        sys.stdout = og
        print("%d random walks : %d policy walks" % (rw,pw))
        for op in ['hadamard', 'l1\t', 'l2\t']:
            score = np.array([evaluate(walk, pos, neg, bin_op=op.strip()) for walk in embeddings])
            print("\t" + op + ": %0.4f (+/-) %0.3f" % (score.mean(), stderr(score))) 

test_mixture(trials=3)