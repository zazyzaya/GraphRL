import torch 
import numpy as np

from torch_geometric.utils import degree
from torch.nn import functional as F
from torch.nn import Linear
from rl_module import Q_Walker, RW_Encoder, Q_Network
from load_cora import load_data

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  

class Recursive_Step_Network(torch.nn.Module):
    def __init__(self, feat_dim, out_dim, hidden=16):
        super().__init__()
        
        self.state = Linear(feat_dim, feat_dim)
        self.input = Linear(feat_dim, feat_dim)
        self.guess = Linear(feat_dim, feat_dim)
        self.out = Linear(feat_dim, out_dim)
      
    '''
    Given a sequence of steps in a RW aims to 
    classify a node 
    '''  
    def forward(self, steps, hidden=None):
        if hidden==None:
            hidden = torch.zeros(steps.size()[0], steps.size()[2])
        
        # If just testing one vector
        if len(steps.size()) == 1:
            inpt = self.input(steps)
            hidden = self.state(hidden)
            hidden = torch.tanh(inpt+hidden)
            return F.softmax(self.out(hidden), dim=0)
           
        for i in range(steps.size()[1]): 
            # Calculate next state vector
            inpt = self.input(steps[:,i,:])
            hidden = self.state(hidden)
            hidden = torch.tanh(inpt+hidden)
            
        # Calculate output based on hidden vec
        return F.softmax(self.out(hidden), dim=1)
    
    '''
    Assumes we're encoding a single step 
    '''
    def encode(self, step, hidden=None):
        if hidden == None:
            hidden = torch.zeros(step.size())
        
        inpt = self.input(step)
        hidden = self.state(hidden)
        return torch.tanh(inpt+hidden)
            

'''
Base loss on accuracy of node prediction from random walk 
Assumes reward network is pre-trained on random walks
'''
class Supervised_RL(Q_Walker):
    def __init__(self, data, reward_net, epsilon=lambda x : 0.95, gamma=0.9,
                 episode_len=5, num_walks=25, hidden=64):
        
        fn = data.x.size()[1]
        
        super().__init__(data, epsilon=epsilon, gamma=gamma, episode_len=episode_len,
                         num_walks=num_walks, state_feats=fn, action_feats=fn, hidden=hidden)
        
        self.reward_net = reward_net
        
        self.qNet1 = Q_Network(fn, fn, hidden=hidden)
        self.qNet2 = Q_Network(fn, fn, hidden=hidden)
       
        # After training, we set it to be minimum but for now, just pick
        # randomly, since we train both anyway 
        self.qNet = np.random.choice([self.qNet1, self.qNet2])
        
        self.pick_min_q = True
    
    '''
    State is the walk thusfar with s[0] being the label
    '''
    def state_transition(self, s, a=None):
        if a == None:
            return self.reward_net.encode(self.data.x[s])
    
        else:
            return self.reward_net.encode(self.data.x[a], hidden=s)
    
    def reward(self, s, a, s_prime, nid):
        y_prime = self.reward_net(self.data.x[a], s)
        y = self.data.y[nid]
        
        # High accuracy and short walks are rewarded
        return weighted_mse(y, y_prime, self.data.weights)
    
    def encode_actions(self, actions, nid=None):
        return self.data.x[torch.tensor(actions, dtype=torch.long)]  
    
    '''
    Override to make this a double-DQN
    '''
    def Q_star(self, s, a, s_prime, nid):
        r = self.reward(s,a,s_prime,nid)
        
        tmp = self.pick_min_q
        self.pick_min_q = False
        
        self.qNet = self.qNet1
        q_next1 = self.policy(s_prime, return_value=True, nid=a)
        
        self.qNet = self.qNet2
        q_next2 = self.policy(s_prime, return_value=True, nid=a)
        
        self.pick_min_q = tmp
        
        return r + (self.gamma * min(q_next1, q_next2))
    
    def Q(self, s, a):
        # Since Double-DQNs are supposed to prevent the Q function
        # from overestimating, we pick the minimum 
        if self.pick_min_q:
            self.qNet = self.qNet1
            q1 = super().Q(s,a)
            
            self.qNet = self.qNet2
            q2 = super().Q(s,a)
            
            q1[q2 < q1] = q2[q2 < q1]
            
            return q1
        
        # Otherwise, which network to use will be specified (used for 
        # training)
        else:
            return super().Q(s,a)    
    
    '''
    Change to save feat vecs instead of str nids
    '''
    def random_walk(self, nid, **kwargs):
        walks = []
        
        for __ in range(self.num_walks):
            walk = [nid]
            
            for _ in range(self.episode_len-1):
                neighbors = self.csr[walk[-1]].indices
                n = np.random.choice(neighbors)
                walk.append(n)
                 
            walk = torch.stack([self.data.x[w] for w in walk])
            walks.append(walk)
            
        return {'label': self.data.y[nid], 'walks': walks}
    
    def policy_walk(self, nid, egreedy=True, **kwargs):
        walks = []    
            
        for __ in range(self.num_walks):
            walk = [nid]
            s = self.state_transition(nid)
            
            for _ in range(self.episode_len-1):        
                a = self.policy(s, walk[-1], egreedy=egreedy)
                s = self.state_transition(s,a=a)
                
                # String so w2v can use it
                walk.append(nid)
            walk = torch.stack([self.data.x[w] for w in walk])
            walks.append(walk)
            
        return {'label': self.data.y[nid], 'walks': walks}
        
def weighted_mse(target, output, weights):
    return torch.mean(weights[target.long()] * (output - target) ** 2)
        
def train(classifier_epochs=400, qnet_epochs=800):
    data = load_data()
    
    # A little preprocessing. Remove all orphans and partition into train/test sets
    non_orphans = (degree(data.edge_index[0], num_nodes=data.x.size()[0]) != 0).nonzero()
    non_orphans = non_orphans.T.numpy()[0]
    np.random.shuffle(non_orphans)
    
    partition = int(non_orphans.shape[0] * 0.75)
    
    train_idx = non_orphans[:partition]
    test_idx = non_orphans[partition:]
    
    train_mask = torch.zeros(data.x.size()[0], dtype=torch.bool)
    test_mask = torch.zeros(data.x.size()[0], dtype=torch.bool)
    
    train_mask[train_idx] = 1
    test_mask[test_idx] = 1
    
    Classifier = Recursive_Step_Network(data.x.size()[1], data.y.size()[1])
    QNet = Supervised_RL(data, Classifier)
    
    # First train classifier on random walks
    Walker = RW_Encoder(QNet)
    c_opt = torch.optim.Adam(Classifier.parameters(), lr=1e-3)
    
    num_batches = 5
    batch_size = len(train_idx) // num_batches
    QNet.num_walks = 1
    
    # Doesn't need to train for too long to learn
    for epoch in range(20):
        
        np.random.shuffle(train_idx)
        for batch in range(num_batches):
            b = train_idx[batch*batch_size:(batch+1)*batch_size]
            walks, y = Walker.generate_walks(batch=b, workers=8, random=True, include_labels=True, quiet=True)

            walks = torch.stack(walks)
            c_opt.zero_grad()
            
            y_prime = Classifier(walks)
            loss = weighted_mse(
                y, y_prime, data.weights
            )
                
            loss.backward()
            c_opt.step()
            
            print("[%d-%d] %0.4f" % (epoch, batch, loss.item()))
    
    # Test accuracy on test set
    with torch.no_grad():
        walks,y = Walker.generate_walks(batch=test_idx, workers=8, random=True, include_labels=True, quiet=True)
        y_prime = Classifier(torch.stack(walks)).argmax(dim=1).detach().numpy()
        y = y.argmax(dim=1).numpy()
    
        print(classification_report(y,y_prime))
       
    # Train Double-Q network 
    loss_fn = torch.nn.MSELoss()
    q_opt1 = torch.optim.Adam(QNet.qNet1.parameters(), lr=1e-5, amsgrad=True)
    q_opt2 = torch.optim.Adam(QNet.qNet2.parameters(), lr=1e-5, amsgrad=True)
    
    num_batches = 15
    batch_size = len(train_idx) // num_batches
    
    QNet.epsilon = lambda x : 0
    
    # Generate random walks to learn state-action values
    for epoch in range(10):
        
        np.random.shuffle(train_idx)
        for batch in range(num_batches):
            b = train_idx[batch*batch_size:(batch+1)*batch_size]
            s,a,r = QNet.episode(batch=b, workers=8)

            q_opt1.zero_grad()
            q_opt2.zero_grad()
            
            loss1 = loss_fn(
                QNet.qNet1(s,a),
                r
            )
            
            loss2 = loss_fn(
                QNet.qNet2(s,a),
                r
            )
            
            loss1.backward(retain_graph=True)
            loss2.backward()
            
            q_opt1.step()
            q_opt2.step()
            
            print("[%d-%d] %0.4f" % (epoch, batch, min(loss1.item(), loss2.item())))
            
    QNet.epsilon = lambda x : 0.95
    
    # Test accuracy now on test set
    with torch.no_grad():
        walks,y = Walker.generate_walks(batch=test_idx, workers=8, include_labels=True, quiet=True)
        y_prime = Classifier(torch.stack(walks)).argmax(dim=1).detach().numpy()
        y = y.argmax(dim=1).numpy()
    
        print(classification_report(y,y_prime))
        
train()