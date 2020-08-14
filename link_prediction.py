import torch 
import load_graphs as lg

from torch_geometric.utils import to_dense_adj

def partition_data(data, percent_hidden=0.15):
    ei = data.edge_index
    num_edges = data.edge_index.size()[1]
    test_set_size = int(num_edges * percent_hidden)
    train_set_size = num_edges - test_set_size
    
    indices = torch.randperm(num_edges)
    
    train_mask = torch.zeros(num_edges, dtype=torch.bool)
    test_mask = torch.zeros(num_edges, dtype=torch.bool)
    
    # First just partition the edges so we hide some from the 
    # model as it learns
    train_mask[indices[:train_set_size]] = 1
    test_mask[indices[train_set_size:]] = 1
    
    # But we also need to make sure there aren't any nodes in 
    # the test set we haven't trained on 
    tr = ei[:,train_mask]
    te = ei[:,test_mask]
    
    # All columns containing values in test but not train will be 0 after 
    # this operation
    intersect = tr.view(train_set_size*2,1) == te.view(test_set_size*2)
    intersect = intersect.sum(dim=0)
    intersect = intersect.view((2,test_set_size))
    intersect = intersect.prod(dim=0)
    
    # Then remove them from the test mask (and add them back into train
    # so as not to waste data)
    test_mask[test_mask.nonzero()[intersect == 0]] = 0
    train_mask = ~test_mask
    
    print(
        "Moved %d/%d edges containing isolated nodes to the training set" % 
        (
            (intersect==0).nonzero().size()[0], 
            train_set_size
        )
    )
    
    print(
        "Train set size:\t%d\nTest set size: \t%d" %
        (
            test_mask.sum().item(), 
            train_mask.sum().item()
        )
    )
    
    data.test_mask = test_mask
    data.train_mask = train_mask
    
def generate_negative_samples(data, num_samples):
    # Build inverse adj matrix to find non-edges
    a = to_dense_adj(data.edge_index)[-1].bool()
    non_edges = (~a).nonzero()
    
    # Filter out orphans/nodes without embeddings
    known = data.edge_index[data.train_mask].flatten().unique()
    allowed_edges = (known == non_edges.flatten()).sum(dim=0)
    allowed_edges = allowed_edges.view(non_edges.size()[0], 2).prod(dim=1).bool()
    non_edges = non_edges[allowed_edges]
    
    # Randomly sample from the allowed edge pool
    return non_edges[torch.randperm(non_edges.size()[0])[:num_samples], :]
    
data = lg.load_cora()

print(generate_negative_samples(, 10))