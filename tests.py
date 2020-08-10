import load_graphs as lg

from rl_module import Q_Walk_Simplified, RW_Encoder, train_loop 

class QW_Cora(Q_Walk_Simplified):
    def __init__(self, data, gamma=0.99, epsilon=lambda x: 0.5, episode_len=10,
                 num_walks=10, hidden=64, one_hot=False, network=None):
        super().__init__(data, gamma=gamma, epsilon=epsilon, episode_len=episode_len,
                         num_walks=num_walks, hidden=hidden, one_hot=one_hot, netowrk=network)
        
    def reward(self, s,a,s_prime,nid):
        return super().min_similarity_reward(s,a,s_prime,nid)

def cora(sample_size=50, epochs=200, clip=None, reparam=40,
            gamma=0.99, nw=10, wl=5):
    print("Testing the CORA dataset")
    data = lg.load_cora()
    
    # Set up a basic agent 
    Agent = Q_Walk_Simplified(data, episode_len=wl, num_walks=nw, 
                           epsilon=lambda x : 0.95, gamma=gamma,
                           hidden=1028, one_hot=False)

    Encoder = RW_Encoder(Agent)
    
    non_orphans = train_loop(Agent, data, epochs=epochs, sample_size=sample_size, 
                             reparam=reparam, clip=clip)    
    
    Encoder.compare_to_random(non_orphans, w2v_params={'size': 128})
    
cora(sample_size=800, epochs=200, gamma=0.9999)