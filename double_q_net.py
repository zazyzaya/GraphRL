from rl_module import Q_Walker

class Double_Q_Net(Q_Walker):
    def __init__(self, data, network_constructor, state_feats=None, action_feats=None,
                 gamma=0.99, epsilon=lambda x: 0.5, episode_len=10, num_walks=10, 
                 hidden=64, network=None):
        
        super().__init__(data, state_feats=state_feats, action_feats=action_feats,
                         gamma=gamma, epsilon=epsilon, episode_len=episode_len,
                         num_walks=num_walks, hidden=hidden)
        
        del self.qNet
        self.qn1 = network_constructor(state_feats, action_feats, hidden=hidden)
        self.qn2 = network_constructor(state_feats, action_feats, hidden=hidden)
        
        self.qNet = np.random.choice([self.qn1, self.qn2])
        self.pick_min_q = True
        
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
        
    