import torch 
import pandas as pd
from torch_geometric.data import Data

tensor = torch.tensor
DATA = '/mnt/raid0_24TB/datasets/cora/'

# It's early days yet, but wouldn't it be hilarious if the paper were called: 
# "You go GRLfRND: Graph Reinforcement Learning for Relational Network Datamining"

def load_data():
    edges = pd.read_csv(DATA + 'cora_cites.csv')
    data = pd.read_csv(DATA + 'cora_content.csv')
    
    id_to_node = dict([(row['paper_id'], idx) for idx,row in data.iterrows()])
    class_to_int = dict([(c, i) for i,c in enumerate(set(data['label']))])
    
    # COO matrix of edges converted to node ids to match the 
    # feature tensor
    citing = [id_to_node[e] for e in edges['citing_paper_id']]
    cited = [id_to_node[e] for e in edges['cited_paper_id']]
    
    # Undirected since there are so many orphans otherwise
    ei = tensor([
        citing + cited,
        cited + citing
    ])
    
    # Don't need paper id's or class in node attr vectors
    X = tensor(
        data.iloc[:, 1:-1].values,
        dtype=torch.float
    )
    
    y = torch.zeros(X.size()[0], len(class_to_int))
    i = 0
    for c in data['label']:
        y[i][class_to_int[c]] = 1
        i += 1
    '''
    y = tensor([
        [class_to_int[c] for c in data['label']]
    ]).T
    '''
    
    weights = y.sum(dim=0)
    weights = weights.max() / weights
    
    return Data(
        x=X,
        edge_index=ei,
        y=y,
        weights=weights
    )
    
if __name__ == '__main__':
    data = load_data()