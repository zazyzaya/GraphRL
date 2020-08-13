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

# Simple GCN class which generates 16-dim node embedding
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(data.x.size()[1], 32)
        self.conv2 = GCNConv(32, 16)

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.weights
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
        forward_pass = model_rw.forward()
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

        self.max_degree = degree(data.edge_index[0]).max()

    def reward(self, s,a,s_prime,nid):
        return super().min_similarity_reward(s,a,s_prime,nid)


def train_RL_walker(gamma=0.99,eps=0.75,nw=10,wl=5,early_stopping=0.01):
    # Set up a basic agent 
    Agent = QW_Cora(
        data, episode_len=wl, num_walks=nw,
        epsilon=lambda x : eps, gamma=gamma,
        hidden=1028, one_hot=True
    )

    Encoder = RW_Encoder(Agent)

    non_orphans = fast_train_loop(
        Agent,
        verbose=1,
        early_stopping=early_stopping,
        nw=nw,
        epochs=5
    )

    return Encoder, non_orphans



# Load data
print("Loading cora data...")
data = lg.load_cora()

# Number of epochs.. Used for GCN training
epochs=50

# Set up a basic agent 
print("Training RL walker...")
Encoder, non_orphans = train_RL_walker(nw=10,wl=5)

# Generate some policy walks and random walks for comparison:
print("Generating policy-based walks...")
policy_walks = Encoder.generate_walks(batch=non_orphans, workers=4)
print("Generating random walks...")
random_walks = Encoder.generate_walks(batch=non_orphans, random=True, workers=4)

# lets start with the random walks
print("Generating aux data for unsupervised GCN training w/ random walks...")
context_pairs, neg_nodes = gen_aux_train_data(random_walks)
# Build the model
model_rw = GCN()
optimizer_rw = torch.optim.Adam([
    dict(params=model_rw.conv1.parameters(), weight_decay=5e-4),
    dict(params=model_rw.conv2.parameters(), weight_decay=0)
], lr=0.01)  # Only perform weight-decay on first convolution.

# Train it
print("Training GCN w/ random walks...")
GCN_train(epochs, model_rw, optimizer_rw, context_pairs, neg_nodes)

# Now the RL version
print("Generating aux data for unsupervised GCN training w/ policy walks...")
context_pairs, neg_nodes = gen_aux_train_data(policy_walks)
model_rl = GCN()
optimizer_rl = torch.optim.Adam([
    dict(params=model_rl.conv1.parameters(), weight_decay=5e-4),
    dict(params=model_rl.conv2.parameters(), weight_decay=5e-4)
], lr=0.01) 

print("Training GCN w/ policy walks...")
GCN_train(epochs, model_rl, optimizer_rl, context_pairs, neg_nodes)

# And now we can evaluate
# Now want to use embeddings to predict label
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier as DTree
from sklearn.multiclass import OneVsRestClassifier as OVR
from sklearn.linear_model import LogisticRegression as LR
estimator = lambda : LR(n_jobs=16)
y_trans = lambda y : y.argmax(axis=1)
model_rw.eval()
model_rl.eval()

# Forward pass to get node embeds
embeds_rw = model_rw.forward().detach().numpy()
lr = estimator()
Xtr, Xte, ytr, yte = train_test_split(embeds_rw, y_trans(data.y))
lr.fit(Xtr, ytr)
yprime = lr.predict(Xte)
print("GCN with Random Walks result:")
print(classification_report(yprime, yte))

# Repeat for RL
embeds_rl = model_rl.forward().detach().numpy()
lr = estimator()
Xtr, Xte, ytr, yte = train_test_split(embeds_rl, y_trans(data.y))
lr.fit(Xtr, ytr)
yprime = lr.predict(Xte)
print("GCN with Reinforcement Learning walks:")
print(classification_report(yprime, yte))
