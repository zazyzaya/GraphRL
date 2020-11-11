import torch 
import pandas as pd 

from torch_geometric.data import Data 
from torch_geometric.utils import add_remaining_self_loops

from torch_geometric.datasets import Reddit

# File locations
DATA = '/mnt/raid0_24TB/datasets/'
CORA = DATA + 'cora/'
BLOG = DATA + 'n2v_benchmarks/BlogCatalog-dataset/'
CITE = DATA + 'citeseer/'
PPI = DATA + 'n2v_benchmarks/Homo_sapiens.mat'

def load_cora():
    edges = pd.read_csv(CORA + 'cora_cites.csv')
    data = pd.read_csv(CORA + 'cora_content.csv')
    
    id_to_node = dict([(row['paper_id'], idx) for idx,row in data.iterrows()])
    class_to_int = dict([(c, i) for i,c in enumerate(set(data['label']))])
    
    # COO matrix of edges converted to node ids to match the 
    # feature tensor
    citing = [id_to_node[e] for e in edges['citing_paper_id']]
    cited = [id_to_node[e] for e in edges['cited_paper_id']]
    
    # Undirected since there are so many orphans otherwise
    ei = torch.tensor([
        citing,# + cited,
        cited,# + citing
    ])
    
    ei = add_remaining_self_loops(ei)[0]
    
    # Don't need paper id's or class in node attr vectors
    X = torch.tensor(
        data.iloc[:, 1:-1].values,
        dtype=torch.float
    )
    
    y = torch.zeros(X.size()[0], len(class_to_int))
    i = 0
    for c in data['label']:
        y[i][class_to_int[c]] = 1
        i += 1
    
    weights = y.sum(dim=0)
    weights = weights.max() / weights
    
    return Data(
        x=X,
        edge_index=ei,
        y=y,
        weights=weights,
        num_nodes=X.size()[0]
    )
    
def load_blog():
    edges = pd.read_csv(BLOG + 'edges.csv', names=['src', 'dst'])
    groups = pd.read_csv(BLOG + 'group-edges.csv', names=['usr', 'group'])
    
    src = edges['src'].to_list()
    dst = edges['dst'].to_list()
    
    # Nid's start at 1 so subtract 1 from everything
    ei = torch.tensor([ src + dst, 
                        dst + src], dtype=torch.long)
    ei = ei - 1
    ei = add_remaining_self_loops(ei)[0]
    
    # Takes max nid from src and dst, then uses max of those two 
    num_nodes = edges.max().max()
    
    # Group id's also start at 1 
    y = torch.zeros((num_nodes, groups['group'].max()), dtype=torch.long)
    y[groups['usr']-1, groups['group']-1] = 1
    
    return Data(
        edge_index=ei,
        y=y,
        num_nodes=num_nodes
    )
    
def load_citeseer():
    # The features file consists of a 1-hot vector 
    # mapped to each node, and the last col is the 
    # class (as a string)
    feats = pd.read_csv(
        CITE + 'citeseer.content',
        delimiter='\t',
        header=None
    )
    
    # Only care about nodes with features; all others don't even bother
    # including edges to nodes that have no features, we can't learn from them
    node_map = dict((feats.iloc[i, 0], i) for i in range(len(feats.index)))
    
    # Slice out col 0 (node id), and col -1 (class label)
    x = torch.tensor(
        feats.iloc[:, 1:-1].to_numpy(),
        dtype=torch.float
    )
    
    # Convert strings to one-hot vectors
    y_str = feats.iloc[:, -1].to_list()
    unique_labels = list(set(y_str))
    y_map = { unique_labels[i]:i for i in range(len(unique_labels)) }
    
    # Build one-hot encoding
    y = torch.zeros((len(y_str), len(y_map)))
    for i,ys in enumerate(y_str):
        y[i][y_map[ys]] = 1
    
    # Map each node string to a unique int
    edges = pd.read_csv(
        CITE + 'citeseer.cites', 
        names=['src', 'dst'],
        delimiter='\t'
    )
    
    src = edges['src'].to_list()
    dst = edges['dst'].to_list()
    
    src_filtered = []
    dst_filtered = []
    
    # Remove edges that connect to nodes without features
    for i in range(len(src)):
        if src[i] in node_map and dst[i] in node_map:
            src_filtered.append(src[i])
            dst_filtered.append(dst[i])
    
    # Free up memory, because I'm just that fancy  
    del src
    del dst 
            
    ei = torch.tensor(
        [
            [node_map[s] for s in src_filtered],
            [node_map[d] for d in dst_filtered]
        ],
        dtype=torch.long
    )
    
    ei = add_remaining_self_loops(ei)[0]

    return Data(
        edge_index=ei,
        x=x,
        y=y
    )

'''
def remove_classes_less_than(data,percent=0.05):
    class_cnt = (data.y.unique().unsqueeze(-1) == data.y).sum(dim=1).float()
    class_cnt = class_cnt.true_divide(class_cnt.sum())
    remove = class_cnt <= percent
    
    to_remove = remove[data.y].nonzero()
    delete0 = (data.edge_index[0] == to_remove).sum(dim=0)
    delete1 = (data.edge_index[1] == to_remove).sum(dim=0)
    delete = delete0.logical_or(delete1)
    
    # Nodes in classes too small to matter are removed from 
    # edge list. They will be filtered out before training
    # as they are now orphans
    data.edge_index = data.edge_index[:,~delete]
'''
    
from scipy.io import loadmat
from torch_geometric.utils import from_scipy_sparse_matrix
def load_ppi():
    mat = loadmat(PPI)
    ei = from_scipy_sparse_matrix(mat['network'])[0]
    y = torch.tensor(mat['group'].todense(), dtype=torch.long)
    X = torch.eye(y.size()[0])
    
    return Data(x=X, y=y, edge_index=ei)
    

def load_reddit():
    print("Loading reddit...")
    root='/tmp/reddit'
    d = Reddit(root=root).data
    return d
