import torch 
import pandas as pd 

from torch_geometric.data import Data 

# File locations
DATA = '/mnt/raid0_24TB/datasets/'
CORA = DATA + 'cora/'
BLOG = DATA + 'n2v_benchmarks/BlogCatalog-dataset/'

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
        citing + cited,
        cited + citing
    ])
    
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
    
    # Nid's start at 1 so subtract 1 from everything
    ei = torch.tensor([edges['src'], edges['dst']], dtype=torch.long)
    ei = ei - 1
    
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

load_blog()