import torch 

from torch.utils import to_dense_adj
from abc import ABC, abstractmethod

class Policy_Network(ABC):
    def __init__(self, data, network, wl=80, nw=10, gamma=0.99, eps=0.95):
        self.data = data
        self.network = network 
        
        if 'dense_adj' in data:
            self.dense_adj = data.dense_adj 
        else:
            self.dense_adj = to_dense_adj(
                self.data.edge_index, 
                max_num_nodes=self.data.num_nodes
            )
        
    def 