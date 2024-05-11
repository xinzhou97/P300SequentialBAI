import numpy as np
class Bandits():
    '''
    Bandit environment for the synthetic experiments.
    

    Args:
        N_arms: number of total arms.
	Delta:  sub-optimality gap.
        p: the parameter controlling the strength of the prior effect.

    '''
    def __init__(self, N_arms, Delta = 1, p = 0.9):
        self.N_arms = N_arms
        self.Delta = Delta
        self.p = p
        self.cur_arm = self.N_arms - 1
        self.prob_vec = np.array([self.p]+[(1-self.p)/(self.N_arms-1)]*(self.N_arms-1))
        return
    
    
    def next_instance(self):
        self.cur_arm = (np.where(np.random.multinomial(1, self.prob_vec) == 1)[0][0] + 1 + self.cur_arm) % self.N_arms
        return
    
    def get_reward(self,I, cur_arm):
        res = np.zeros(self.N_arms)
        res[I] = np.random.normal(self.Delta * (I == cur_arm), 1, len(I))
        return(res)
    def reset(self):
        self.cur_arm = self.N_arms - 1
    
    def generate_words(self, N_word):
        '''
        create words (with length "N_word") ".
        '''
        res = np.zeros(N_word)
        self.reset()
        for j in range(N_word):
            self.next_instance()
            res[j] = self.cur_arm
        return(res)
        
    def generate_multiple_words(self, N_iter, N_word):
        '''
        create words (with length "N_word") "N_iter" times.
        output: N_iter * N_word 
        '''
        return(np.array([self.generate_words(N_word) for i in range(N_iter)]))

