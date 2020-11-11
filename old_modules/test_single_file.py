import torch 
import load_graphs as lg
import pickle as pkl
import torch.nn.functional as F
import pandas as pd

# Suppresses sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from random import randint
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score 
from sklearn.multiclass import OneVsRestClassifier as OVR
from sklearn.linear_model import LogisticRegression as LR
from torch_geometric.utils import degree
from rl_module_single_file import Q_Walker, RW_Encoder, train_loop

default_agent_params = dict(
    episode_len=20,
    num_walks=5,
    epsilon=lambda x : 0.95,
    gamma=1-1e-3,
    hidden=2048,
    beta=0.25
)

train_settings = dict(
    verbose=1,
    early_stopping=0.001,
    nw=5,
    wl=10,
    epochs=50,
    sample_size=None,
    lr=1e-3,
    train_eps=0.75
)

# Same as n2v paper
w2v_params = dict(
    size=128,
    negative=10,
    window=10
)

from sklearn.decomposition import PCA
def pca(X, dim=256):
    n_components = min(dim, X.size()[1])
    decomp = PCA(n_components=n_components, random_state=1337)
    return torch.tensor(decomp.fit_transform(X.numpy()))

def generic_test(data, trials, num_tests, max_eps, agent_params=default_agent_params,
                 train_settings=train_settings, undirected=True, preprocess=True,
                 multiclass=False):
    
    if preprocess:
        print("Running PCA on node features")
        data.x = pca(data.x)
    
    Agent = Q_Walker(data, **agent_params)
    if undirected:
        Agent.remove_direction()
    
    Agent.update_action_map()
    
    Encoder = RW_Encoder(Agent)

    # First train it with user settings
    non_orphans = train_loop(Agent, strategy='egreedy', **train_settings)
    
    # Get some idea abt what predictions look like for neighbors of random node
    rnd = non_orphans[randint(0,non_orphans.shape[0]-1)]
    print(Agent.value_estimation(Agent.state_transition(torch.tensor([[rnd]])), torch.tensor([[rnd]])))

    # Then, train it a little more assuming perfect policy found (no more exploration)
    train_settings['epochs'] = 25 * train_settings['nw']
    train_settings['nw'] = 1 # No need to repeat the same thing if its deterministic now
    train_loop(Agent, strategy='perfect', **train_settings)
    
    # Get some idea abt what predictions look like for neighbors of random node
    print(Agent.value_estimation(Agent.state_transition(torch.tensor([[rnd]])), torch.tensor([[rnd]])))
    
    print(" train eps=%0.2f\n nw=%d\n wl=%d\n beta=%0.2f\n gamma=%f" % (
            train_settings['train_eps'],
            Agent.num_walks, 
            Agent.episode_len, 
            agent_params['beta'], 
            agent_params['gamma']
        )
    )
    
    for i in range(0,num_tests+1):
        print()
        eps = (i/num_tests) * max_eps
        Agent.epsilon = lambda x : eps/100
        all_stats = {'prec':[], 'rec':[], 'f1':[]}
        
        if not multiclass:
            all_stats['acc'] = []
            
        for _ in tqdm(range(trials), desc="%d%% random walks" % (100-eps)):
            if i != num_tests:
                X,y = Encoder.generate_walks(
                    non_orphans, strategy='egreedy', 
                    silent=True, encode=True,
                    w2v_params=w2v_params
                )
            
            # Last iteration also test weighted
            else:
                X,y = Encoder.generate_walks(
                    non_orphans, strategy='weighted', 
                    silent=True, encode=True,
                    w2v_params=w2v_params
                )

            stats = Encoder.get_accuracy_report(X,y,test_size=0.25,multiclass=multiclass)
            
            if not multiclass:
                all_stats['acc'].append(stats['accuracy'])
                
            all_stats['prec'].append(stats['weighted avg']['precision'])
            all_stats['rec'].append(stats['weighted avg']['recall'])
            all_stats['f1'].append(stats['weighted avg']['f1-score'])

        eps = eps if i != num_tests else 'Weighted' 
        print('======= ' + str(eps) + ' =======')
        df = pd.DataFrame(all_stats)
        df = pd.DataFrame([df.mean(), df.sem()])
        df.index = ['mean', 'stderr']
        print(df)
    
    return Agent

def citeseer(gamma=0.99, nw=5, wl=20, epsilon=0.5, trials=5, 
         num_tests=7, max_eps=35, beta=1):
    
    print("Testing the Citeseer dataset")
    data = lg.load_citeseer()
    
    agent_params = dict(
        episode_len=wl,
        num_walks=nw,
        epsilon=lambda x : epsilon,
        gamma=gamma,
        hidden=2048,
        beta=beta
    )
    
    global train_settings  
    train_settings['train_eps'] = epsilon  
    train_settings['nw'] = 2
    train_settings['epochs'] = 75
    return generic_test(
        data, 
        trials, 
        num_tests,
        max_eps,
        agent_params=agent_params, 
        train_settings=train_settings,
        undirected=True
    )

citeseer()