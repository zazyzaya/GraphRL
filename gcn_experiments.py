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
from torch_geometric.nn import GCNConv
from scipy.sparse import csr_matrix

from gensim.models import Word2Vec

import load_graphs as lg
from rl_module import *
from rl_module_improved import Q_Walk_Simplified

# Simple GCN class which generates 16-dim node embedding
class GCN(torch.nn.Module):
    def __init__(self, data):
        super(GCN, self).__init__()
        self.data = data 
        self.conv1 = GCNConv(data.x.size()[1], 128)
        self.conv2 = GCNConv(128,64)

    def forward(self):
        x, edge_index = self.data.x, self.data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        #return F.log_softmax(x, dim=1)
        return x

# Function that takes node sequences and returns additional data for unsupervised training
# Specifically, it returns a set of nodes that co-occur on random walks (context_pairs)
# As well as a set of nodes that do not occur on random walks (neg_nodes)
# TODO still need to make sure the random sampled node is not in context_pairs
def gen_aux_train_data(walks):
    # Go through each walk and generte context pairs
    context_pairs = []
    neg_nodes = []
    for w in tqdm(walks):
        for n in w:
            for n2 in w:
                if n != n2:
                    context_pairs.append([int(n),int(n2)])

    # Build negative samples (just 1 for now)
    # for each node in our context pairs generate a series of negative samples
    num_neg_samples = 5
    for n in tqdm(context_pairs):
        neg_samples = []
        for _ in range(num_neg_samples):
            n2 = np.random.choice(non_orphans)
            #TODO make sure negative sample isnt in positive samples...
            #while [n,n2] in context_pairs or [n2,n] in context_pairs:
            #    n2 = np.random.choice(non_orphans)
            neg_samples.append(int(n2))
        neg_nodes.append(neg_samples)
    print("Num context pairs: %d" % len(context_pairs))
    print("Num neg nodes: %d" % len(neg_nodes))
    context_pairs = np.array(context_pairs)
    neg_nodes = np.array(neg_nodes)
    return context_pairs, neg_nodes

def gen_aux_train_data_tensors(walks):
    print("Generating pos samples")
    context_pairs = []
    for i in range(walks.size()[1]):
        for j in range(walks.size()[1]):
            if i == j:
                continue
            context_pairs.append(walks[:, torch.tensor([i,j])])
    
    context_pairs = torch.cat(context_pairs, dim=0)
    
    # Encode tuples for fast lookup to make sure negative samples
    # aren't present. Convert from tuple to unique int and store as a set
    exp = torch.tensor([walks.max().item(), 1])
    encode = lambda x : (x*exp).sum(axis=1)
    encoded_pairs = set([encode.item() for encode in (context_pairs*exp).sum(axis=1)])
    
    print("Generating neg samples")
    num_neg_samples = 5
    neg_samples = torch.tensor(
        np.random.choice(walks.flatten().unique().numpy(), size=(context_pairs.size()[0]*num_neg_samples,2))
    )
    
    print("Removing false-neg samples")
    non_dupes = []
    for enc in tqdm(encode(neg_samples)):
        non_dupes.append(enc.item() not in encoded_pairs)
        
    neg_samples = neg_samples[non_dupes, :]
    
    print("Sampling the correct num of neg samples")
    neg_samples = neg_samples[:context_pairs.size()[0], :]
    
    return context_pairs.numpy(), neg_samples.numpy()

# Skip-gram like Unsupervised loss based on similarity of context pairs and dissimilarity of neg samples
def GCN_unsup_loss(embeds, context_pairs, neg_samples):
    input_embeds = embeds[context_pairs[:,0]]
    context_embeds = embeds[context_pairs[:,1]]

    # neg embeds is avg of all neg samps
    neg_embeds = embeds[neg_samples]
    num_neg_samples = neg_embeds.size()[1]
    neg_embeds_avg = torch.true_divide(neg_embeds.sum(axis=1),num_neg_samples)

    # Compute affinity between input nodes & context nodes (nodes that co-occur on random walks)
    aff = input_embeds * context_embeds
    # Compute affinity between input nodes & avg of negative nodes (nodes that do not occur on random walks)
    neg_aff = input_embeds * neg_embeds_avg
    # Cross entropy loss 
    true_xent = torch.nn.BCEWithLogitsLoss()(aff, torch.ones_like(aff))
    negative_xent = torch.nn.BCEWithLogitsLoss()(neg_aff, torch.zeros_like(neg_aff))
    loss = torch.sum(true_xent + negative_xent)
    return loss

def GCN_train(epochs, model, optimizer, context_pairs, neg_nodes):
    for e in range(epochs):
        model.train()
        optimizer.zero_grad()
        # Get node embeddings
        forward_pass = model.forward()
        loss = GCN_unsup_loss(forward_pass, context_pairs, neg_nodes)
        l = loss.item()
        print("Epoch %d: Loss: %2f" % (e, l))
        loss.backward()
        optimizer.step()

# For RL walks
class QW_Cora(Q_Walk_Simplified):
    def __init__(self, data, gamma=0.99, epsilon=lambda x: 0.5, episode_len=10,
                 num_walks=10, hidden=64, one_hot=False, network=None):
        super().__init__(data, gamma=gamma, epsilon=epsilon, episode_len=episode_len,
                         num_walks=num_walks, hidden=hidden, one_hot=one_hot, network=network)

        self.cs = torch.nn.CosineSimilarity()
        
    def reward(self, s,a,s_prime,nid):
        return self.cs(self.data.x[nid],self.data.x[a]).unsqueeze(-1)

from sklearn.decomposition import PCA
def preprocess(X):
    decomp = PCA(n_components=256, random_state=1337)
    return torch.tensor(decomp.fit_transform(X.numpy()))

def train_RL_walker(data, gamma=0.99,eps=0.75,nw=10,wl=5,
                    early_stopping=0.03,epochs=50):
    # Preprocessing data
    print("Preprocessing data")
    data.x = preprocess(data.x)
    
    # Set up a basic agent 
    Agent = QW_Cora(
        data, episode_len=wl, num_walks=nw,
        epsilon=lambda x : eps, gamma=gamma,
        hidden=1028, one_hot=True
    )

    Encoder = RW_Encoder(Agent)
    Agent.update_action_map()

    non_orphans = fast_train_loop(
        Agent,
        verbose=1,
        early_stopping=early_stopping,
        nw=nw,
        epochs=epochs//2
    )
    
    fast_train_loop(
        Agent,
        verbose=1,
        early_stopping=early_stopping,
        nw=nw,
        epochs=epochs//2,
        strategy='weighted'
    )

    return Encoder, non_orphans

def get_walks(Encoder, non_orphans, nw=10, wl=5, strategy='random'):
    # Generate some policy walks and random walks for comparison:
    print("Generating %s walks..." % strategy)
    return Encoder.generate_walks_fast(
        batch=non_orphans, 
        strings=False, 
        strategy=strategy
    )

def get_gcn_embeddings(data, walks, text='random', epochs=50):
    print("Generating aux data for unsupervised GCN training w/ %s walks..." % text)
    context_pairs, neg_nodes = gen_aux_train_data_tensors(walks)
    
    # Build the model
    model = GCN(data)
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=5e-4)
    ], lr=0.001) 

    # Train it
    print("Training GCN w/ %s walks..." % text)
    GCN_train(epochs, model, optimizer, context_pairs, neg_nodes)
    
    model.eval()
    embeds = model.forward().detach().numpy()
    
    return embeds

# And now we can evaluate
# Now want to use embeddings to predict label
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier as DTree
from sklearn.multiclass import OneVsRestClassifier as OVR
from sklearn.linear_model import LogisticRegression as LR
def test_node_classification(embeds, walk_type, data):
    estimator = lambda : LR(n_jobs=16, max_iter=1000)
    y_trans = lambda y : y.argmax(axis=1)

    # Forward pass to get node embeds
    lr = estimator()
    Xtr, Xte, ytr, yte = train_test_split(embeds, y_trans(data.y))
    lr.fit(Xtr, ytr)
    yprime = lr.predict(Xte)
    print("GCN with %s result:" % walk_type)
    print(classification_report(yprime, yte))
    

def node_class_cora():
    # Load data
    print("Loading cora data...")
    data = lg.load_cora()
    
    # Set up a basic agent 
    print("Training RL walker...")
    Encoder, non_orphans = train_RL_walker(data,epochs=25)
    
    rw = get_gcn_embeddings(
        data, 
        get_walks(Encoder, non_orphans, strategy='random'),
        epochs=25
    )
    rl = get_gcn_embeddings(
        data, 
        get_walks(Encoder, non_orphans, strategy='weighted'),
        text='weighted',
        epochs=25
    )
    
    test_node_classification(rw, 'Random Walk', data)
    test_node_classification(rl, 'Policy Guided', data)
    
if __name__ == '__main__':
    node_class_cora()