import torch 
import pandas as pd 

from tqdm import tqdm 
from torch_geometric.data import Data

FNAME = '/mnt/raid0_24TB/datasets/LANL_2015/data_files/auth.txt'

'''
Converts each edge feature to a PMF of what features it normally has
as well as adding a slight smoothing amount so there are no 0 prob features
to allow for KB-divergence measuring later on 
'''
def normalize(edge_feats, epsilon=1e-6):
    edge_feats += epsilon
    return edge_feats / edge_feats.sum(dim=1).unsqueeze(-1)

def get_or_assign_edge(e, emap, num_feats):
    if e in emap:
        return emap[e]
    else: 
        emap[e] = torch.zeros(num_feats, dtype=torch.float)
        return emap[e]

def get_or_assign_node_id(n, nm, idx):
    if n in nm:
        return nm[n]
    else:
        nm[n] = idx[0]
        idx[0] += 1
        return nm[n] 


def build_feat_map(get_reader, end):
    feats = set()
    
    for df in tqdm(get_reader(), desc='Frames processed'):
        feats = feats.union(set(df['logon_type']).union(set(df['auth_type']))) - {'?'}

        if df['timestamp'].iloc[-1] >= end:
            break

    feat_map = dict((k,v) for v,k in enumerate(feats))            
    print(feat_map)
    return feat_map

'''
Only load data between the timestamps [start,end)
'''
def build_data(start=150000, end=151649, out='lanl.dat', feat_map=None):
    get_reader = lambda : pd.read_table(FNAME, delimiter=',', header=0,
                       chunksize=10000, iterator=True)

    if feat_map == None:
        print("Building feat map")
        feat_map = build_feat_map(get_reader, 50000)

    # Use singletons to act as pseudo-pointers        
    node_map = dict()
    node_index = [0]
    
    ed_map = dict()   
    
    build_src = lambda x : x['src_user'] + x['src_computer']
    build_dst = lambda x : x['dst_user'] + x['dst_computer']
    missing = lambda x : x == '?'
    
    hit_end = False
    get_num_feats = True 
    
    for df in tqdm(get_reader(), desc='Frames processed'):
        if int(df['timestamp'].iloc[-1]) < start:
            continue 
        
        for _, row in df.iterrows():
            if int(row['timestamp']) < start:
                continue
            
            if int(row['timestamp']) >= end:
                hit_end = True
                break
            
            s = build_src(row)
            d = build_dst(row)
            
            s = get_or_assign_node_id(s, node_map, node_index)
            d = get_or_assign_node_id(d, node_map, node_index)
            e = get_or_assign_edge((s,d), ed_map, len(feat_map))
            
            if not missing(row['logon_type']):
                e[feat_map[row['logon_type']]] += 1
            if not missing(row['auth_type']):
                e[feat_map[row['auth_type']]] += 1
            
                    
        if hit_end:
            break

    
    print("Encoding edges")
    edges = []
    feats = []
    for k,v in ed_map.items():
        edges.append(k)
        feats.append(v)
        
    edge_index = torch.tensor(edges).T
    edge_attr = torch.stack(feats)
    edge_attr = normalize(edge_attr)
    
    print("Saving %d nodes with %d edges" % (len(node_map), len(ed_map)))
    d = Data(
        edge_index=edge_index, 
        edge_attr=edge_attr, 
        num_nodes=len(node_map),
        node_lookup=node_map
    )
    torch.save(d, out)
    
    return feat_map

def load_data(fname='lanl.dat'):
    return torch.load(fname)
        
if __name__ == '__main__':
    feat_map = build_data(start=100000,end=150000,out='train.dat')
    build_data(feat_map=feat_map, out='test.dat')