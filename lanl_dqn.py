import random 

from rl_module import Q_Walker

'''
Make sure to pass edata into constructor
'''
class LANL_Walker(Q_Walker):
    def reward(self, s,a,s_prime,nid):
        
    
    
    def state_transition(self, s, a=None):
        if a == None:
            return self.data.x[s]
        else:
            return self.data.x[a[0]]
    
    '''
    Chooses next action with optional e-greedy policy 
    Override so action is edge data index not neighbor id
    '''
    def policy(self, s, nid, egreedy=True, return_value=False):    
        neighbors = self.csr[nid].indices
        edges = self.csr[nid].data
        
        # Can't do anything about orphaned nodes
        if len(neighbors) == 0:
            return nid
        
        actions = torch.stack(edges)
        
        if not return_value and egreedy and random.random() < self.epsilon(self.episode_cnt):
            action = np.random.choice(edges)    
        else:
            value_predictions = self.Q(s,actions)
            
            # We can also use this function to calculate max(a) Q(s,a)
            if return_value:
                return value_predictions.max().item()
            
            action = value_predictions.argmax().item()
           
        # Action is the NID of the next node to visit 
        return (neighbors[action], edges[action])
    
    def encode_actions(self, actions, nid=None):
        return torch.stack([a[1] for a in actions])