import numpy as np
class LanguageModel():
    '''
    Language Model for the synthetic experiments.
    

    Args:
        N_arms: number of total arms 
        p: the parameter controlling the strength of the prior effect.

    '''
    def __init__(self, N_arms, p = 0.9):
        self.N_arms = N_arms
        self.p = p
        return

    def get_probs(self, arm_current = -1):
        prob_vec = np.array([(1-self.p)/(self.N_arms-1)]*(self.N_arms))
        prob_vec[(arm_current + 1) % self.N_arms] = self.p
        return(prob_vec.copy())


