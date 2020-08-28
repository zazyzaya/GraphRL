import torch 
import load_graphs as lg

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import to_dense_adj, add_remaining_self_loops

def partition_data(data, percent_hidden=0.15):
    # First, add self loops 
    ei = data.edge_index
    ei = add_remaining_self_loops(ei)[0]
    
    data.edge_index = ei
    
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
    
    # Also need to keep self-loops for training later on
    test_mask[ei[0] == ei[1]] = 0
    
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
            train_mask.sum().item(),
            test_mask.sum().item()
        )
    )
    
    data.test_mask = test_mask
    data.train_mask = train_mask
    
def generate_negative_samples(data, num_samples):
    # Build inverse adj matrix to find non-edges
    a = to_dense_adj(data.edge_index[:, data.train_mask])[-1].bool()
    
    # Determine what nodes we'll have embeddings for
    known = data.edge_index[:, data.train_mask].flatten().unique()
    not_known = torch.full((a.size()[0],), 1, dtype=torch.bool)
    not_known[known] = 0
    
    # Mark all untrained nodes as having edges so they won't show
    # up in the inverted adj matrix 
    a[not_known, :] = 1
    a[:, not_known] = 1
    
    # That way, when we invert the matrix, the non-edges will only be
    # between nodes that we for sure have embeddings for
    non_edges = (~a).nonzero()    
    
    # Randomly sample from the allowed edge pool
    return non_edges[torch.randperm(non_edges.size()[0])[:num_samples], :]
    

def evaluate(embeddings, pos_samples, neg_samples, bin_op='hadamard', test_size=0.15):    
    X_src_pos = embeddings[pos_samples[:,0], :]
    X_dst_pos = embeddings[pos_samples[:,1], :]
    
    X_src_neg = embeddings[neg_samples[:,0], :]
    X_dst_neg = embeddings[neg_samples[:,1], :]
    
    y = torch.zeros(
        (pos_samples.size()[0]+neg_samples.size()[0],), 
        dtype=torch.float
    )
    
    y[:pos_samples.size()[0]] = 1
    
    # Set the binary operation based on the argument
    # hadamard is default as it performs best on the original
    # node2vec paper
    if bin_op == 'hadamard':
        bin_op = lambda x : x[0] * x[1]
    elif bin_op == 'l1':
        bin_op = lambda x : torch.abs(x[0] - x[1])
    elif bin_op == 'l2':
        bin_op = lambda x : (x[0] - x[1]) ** 2
    elif bin_op == 'avg': 
        bin_op = lambda x : (x[0] + x[1]).true_divide(2)
    elif type(bin_op) == type(lambda x : 0):
        pass
    else:
        raise ValueError("bin_op must be one of ['hadamard', 'l1', 'l2', 'avg'] or a function")
    
    X_pos = bin_op((X_src_pos, X_dst_pos))
    X_neg = bin_op((X_src_neg, X_dst_neg))
    X = torch.cat([X_pos, X_neg], dim=0)
    
    # Convert to numpy for sklearn
    X = torch.sigmoid(X).numpy() 
    y = y.numpy()
    
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=1337
    )
    
    # Then we train the Logistic Regression unit 
    LR = LogisticRegressionCV(
        cv=10, 
        max_iter=2000, 
        scoring='roc_auc'
    )
    LR.fit(Xtr, ytr)
    
    confidence = LR.predict_proba(Xte)
    positive_column = list(LR.classes_).index(1)
    return roc_auc_score(yte, confidence[:, positive_column])