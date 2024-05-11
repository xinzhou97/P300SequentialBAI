import os
os.environ['PYTHONHASHSEED']=str(1)
import random as rn
import numpy as np
def reset_random_seeds(seed=1):

    np.random.seed(seed)
    rn.seed(seed)
reset_random_seeds()

SYM_PATH = "./"
batch_size = 1
N_iter=200
N_word = 20

# As the prior specification changes,the agents, language models, and the  bandits will change.

# the agents
class ThompsonSamplingTop2():
    '''
    Class to implement a Thompson sampling policy top 2 in batch case.

    Args:
        N_arms: number of total arms 
        N_top: number of picked top arms
    '''
    def __init__(self, N_arms, N_top, batch_size, delta, tmax=5000):
        self.N_arms = N_arms
        self.N_top = N_top
        self.batch_size = batch_size
        self.delta = delta
        self.tmax = tmax
        return

    def initialise(self, mu, Sigma, sigma, beta=1/2):
        '''
        Initialise the arms

        Args:
            mu: mu in prior distribution
            Sigma:Sigma in prior distribution
            sigma: sd of the noise in the rewards
            beta: prob in TS top2 sampling rule
        '''
        if type(mu) == int or type(mu) == float:
            self.mu = np.array([mu for arm in range(self.N_arms)]) * 1.0
        else:
            self.mu = mu
        self.Sigma = np.array([Sigma for arm in range(self.N_arms)]) * 1.0
        self.Sigma_inv = np.array([1/Sigma for arm in range(self.N_arms)]) * 1.0
        self.sigma = sigma
        self.mu_arg2 = self.Sigma_inv * self.mu
        self.beta = beta
        self.s = np.sqrt(self.Sigma)
        self.m = self.mu
        self.t = 0
        return

    def _calculate_Z(self, index_i):
        ## return Z_{i,j} as a vector given index_i
        res = (self.m[index_i]-self.m)/np.sqrt(self.s[index_i]**2 + self.s**2)
        return(res)


    def select_arm(self):
        """Selects an arm for each round.
                
        Returns:
                An vector corresponding to the index of the selected 
                arm in the current batch
        """

        # initialize arm set for picked and remaining
        I_pick = np.array([])
        I_remaining = np.arange(self.N_arms)

        m_top2 = -np.sort(-self.m)[:2]

        
        ## select arms
        while len(I_pick) < self.batch_size:
            n0 = len(I_remaining)
            m_curr = self.m[I_remaining]
            s_curr = self.s[I_remaining]
            
            I1_ind = np.argmax(np.random.normal(m_curr, s_curr, n0))
            I2_ind = I1_ind
            count = 0
            while I1_ind == I2_ind:
                I2_ind = np.argmax(np.random.normal(m_curr, s_curr, n0))
                count += 1
                if count >= 10000:
                    m_curr_mod = m_curr
                    m_curr_mod[I1_ind] = -np.inf # I2 should not be equal to I1
                    I2_ind = np.argmax(np.random.normal(m_curr_mod, s_curr, n0))
            unif = np.random.uniform(0,1,1)[0]
            if unif < self.beta:
                I_ind = I1_ind
            else:
                I_ind = I2_ind
            I_pick = np.append(I_pick, I_remaining[I_ind])
            I_remaining = np.delete(I_remaining, I_ind)
        I_pick = np.random.permutation(I_pick).astype(int)
        
        return(I_pick)
        
    def update(self, I, R):
        '''
        Args:
            I: pulled arms of current batch (with order in the batch)
            R: rewards= vector for the current batch
        '''
        counts = np.zeros(self.N_arms)
        counts[I] = 1
        self.Sigma_inv = self.Sigma_inv + counts * self.sigma**(-2)
        self.Sigma = np.array([x**(-1) for x in self.Sigma_inv])
        score = R
        score[np.isnan(score)] = 0
        self.mu_arg2 = self.mu_arg2 + score * self.sigma**(-2)
        self.mu = self.Sigma * self.mu_arg2
        self.s = np.sqrt(self.Sigma)
        self.m = self.mu
        self.top = (-self.m).argsort()[:self.N_top]
        self.t += 1
        

        ## stopping rule
        Z_min = np.min([[np.delete(self._calculate_Z(ind), self.top)] 
                                        for ind in self.top])
        

        gamma_t = np.sqrt(2*np.log(np.log(self.t*self.batch_size)/self.delta))
        thresh = gamma_t
        return(Z_min >= thresh or self.t >=self.tmax)

class RandomPolicy():
    def __init__(self, N_arms, N_top, batch_size, delta, tmax=5000):
        self.N_arms = N_arms
        self.N_top = N_top
        self.batch_size = batch_size
        self.delta = delta
        self.tmax=tmax
        return

    def initialise(self, mu, Sigma, sigma, beta=1/2):
        '''
        Initialise the arms

        Args:
            mu: mu in prior distribution
            Sigma:Sigma in prior distribution
            sigma: sd of the noise in the rewards
            beta: prob in TS top2 sampling rule
        '''
        if type(mu) == int or type(mu) == float:
            self.mu = np.array([mu for arm in range(self.N_arms)]) * 1.0
        else:
            self.mu = mu
        self.Sigma = np.array([Sigma for arm in range(self.N_arms)]) * 1.0
        self.Sigma_inv = np.array([1/Sigma for arm in range(self.N_arms)]) * 1.0
        self.sigma = sigma
        self.mu_arg2 = self.Sigma_inv * self.mu
        self.beta = beta
        self.s = np.sqrt(self.Sigma)
        self.m = self.mu
        self.t = 0
        return

    def _calculate_Z(self, index_i):
        ## return Z_{i,j} as a vector given index_i
        res = (self.m[index_i]-self.m)/np.sqrt(self.s[index_i]**2 + self.s**2)
        return(res)


    
    def select_arm(self):
        """Selects an arm for each round.
                
        Returns:
                An vector corresponding to the index of the selected 
                arm in the current batch
        """

        I_pick = np.random.choice(self.N_arms, 1).astype(int)
        return(I_pick)
        
    def update(self, I, R):
        '''
        Args:
            I: pulled arms of current batch (with order in the batch)
            R: rewards= vector for the current batch
        '''
        counts = np.zeros(self.N_arms)
        counts[I] = 1
        self.Sigma_inv = self.Sigma_inv + counts * self.sigma**(-2)
        self.Sigma = np.array([x**(-1) for x in self.Sigma_inv])
        score = R
        score[np.isnan(score)] = 0
        self.mu_arg2 = self.mu_arg2 + score * self.sigma**(-2)
        self.mu = self.Sigma * self.mu_arg2
        self.s = np.sqrt(self.Sigma)
        self.m = self.mu
        self.top = (-self.m).argsort()[:self.N_top]
        self.t += 1
        

        ## stopping rule
        Z_min = np.min([[np.delete(self._calculate_Z(ind), self.top)] 
                                        for ind in self.top])

        gamma_t = np.sqrt(2*np.log(np.log(self.t*self.batch_size)/self.delta))
        thresh = gamma_t
        return(Z_min >= thresh or self.t >= self.tmax)
    
# the language model    
class LanguageModel():
    def __init__(self, N_arms):
        self.N_arms = N_arms
        return
    
    def set_prior_mean_matrix(self,mat):
        self.prior_mean_matrix = mat
        return
    
# the bandits
class Bandits():
    def __init__(self, N_arms, sd_ins = 1):
        self.N_arms = N_arms
        self.cur_arm = np.random.randint(self.N_arms)
        self.sd_ins = sd_ins
        return
    
    def set_prior_mean_matrix(self,mat):
        self.prior_mean_matrix = mat
    
    def next_instance(self):
        mu_prior = self.prior_mean_matrix[self.cur_arm]
        self.cur_mu = np.random.normal(mu_prior, self.sd_ins, self.N_arms)
        self.cur_arm = np.argmax(self.cur_mu)
        return
    
    def get_reward(self,cur_mu, I):
        res = np.zeros(len(cur_mu))
        res[I] = np.random.normal(cur_mu[I], 1, len(I))
        return(res)
    def reset(self):
        self.cur_arm = np.random.randint(self.N_arms)
    
    def generate_words(self, N_word):
        '''
        create words (with length "N_word") ".
        '''
        res = np.zeros((N_word, self.N_arms))
        self.reset()
        for j in range(N_word):
            self.next_instance()
            res[j] = self.cur_mu
        return(res)
        
    def generate_multiple_words(self, N_iter, N_word):
        '''
        create words (with length "N_word") "N_iter" times.
        output: N_iter * N_word * self.N_arms
        '''
        return(np.array([self.generate_words(N_word) for i in range(N_iter)]))



# run the experiments
for J in [5,10]:
    for mu_prior in [4,5]:
        for sd_prior in [0.5,1]:
            for lan_form in [1,2,3]:
                # U matrix, representing the language model
                if lan_form == 1:
                    def get_matrix_prior(i, J):
                        res = np.zeros(J)
                        res[(i+1) % J] = 1
                        return(res)
                    prior_mean_matrix = np.array([get_matrix_prior(i, J) for i in range(J)]) * mu_prior
                elif lan_form == 2:
                    def get_matrix_prior(i, J):
                        res = np.zeros(J)
                        res[(i+1) % J] = 1
                        res[(i+2) % J] = 0.5
                        return(res)    
                    prior_mean_matrix = np.array([get_matrix_prior(i, J) for i in range(J)]) * mu_prior
                elif lan_form == 3:
                    def get_matrix_prior(i, J):
                        res = np.zeros(J)
                        if i < (J-2):
                            res[(i+1) % (J-2)] = 1
                        else:
                            res[i % 2 + (J-2)] = 1
                        return(res)    
                    prior_mean_matrix = np.array([get_matrix_prior(i, J) for i in range(J)]) * mu_prior





                lm = LanguageModel(J)
                lm.set_prior_mean_matrix(prior_mean_matrix)
                bandits = Bandits(J, sd_prior)
                bandits.set_prior_mean_matrix(prior_mean_matrix)



                for alg_id in [1,2,3]:
                    if alg_id == 1:
                        # STTS
                        reset_random_seeds()
                        suc_mat = np.zeros((N_iter,N_word))
                        t_mat = np.zeros((N_iter,N_word))
                        mu_bandits = bandits.generate_multiple_words(N_iter,N_word)
                        sigma = 1
                        for i in range(N_iter):
                            id_go = True
                            TStop2 = ThompsonSamplingTop2(J, 1, batch_size, 0.1/N_word)
                            TStop2.initialise(0,100,sigma)
                            for j in range(N_word):
                                if id_go == False:
                                    continue
                                while True:
                                    I = TStop2.select_arm()
                                    R = bandits.get_reward(mu_bandits[i,j], I)
                                    if TStop2.update(I, R):
                                        break
                                suc_mat[i,j] =  ((TStop2.top) == np.argmax(mu_bandits[i,j]))
                                t_mat[i,j] =  TStop2.t
                                arm_id = TStop2.top[0]
                                TStop2 = ThompsonSamplingTop2(J, 1, batch_size, 0.1/N_word)
                                TStop2.initialise(lm.prior_mean_matrix[arm_id],sd_prior ** 2,sigma)

                        np.save(SYM_PATH + "STTS_"+str(mu_prior)+"_"+str(sd_prior)+"_"+str(lan_form)+"_Alt.npy",
                              np.array([np.mean(np.mean(suc_mat, axis=1) == 1),np.mean(np.sum(t_mat, axis=1)) * batch_size,
                                        suc_mat,t_mat]))
                    elif alg_id == 2:
                        # VTTS
                        reset_random_seeds()
                        suc_mat = np.zeros((N_iter,N_word))
                        t_mat = np.zeros((N_iter,N_word))
                        mu_bandits = bandits.generate_multiple_words(N_iter,N_word)
                        sigma = 1
                        for i in range(N_iter):
                            TStop2 = ThompsonSamplingTop2(J, 1, batch_size, 0.1/N_word)
                            TStop2.initialise(0,100,sigma)
                            for j in range(N_word):
                                while True:
                                    I = TStop2.select_arm()
                                    R = bandits.get_reward(mu_bandits[i,j], I)
                                    if TStop2.update(I, R):
                                        break
                                suc_mat[i,j] =  ((TStop2.top) == np.argmax(mu_bandits[i,j]))
                                t_mat[i,j] =  TStop2.t
                                arm_id = TStop2.top[0]
                                TStop2 = ThompsonSamplingTop2(J, 1, batch_size, 0.1/N_word)
                                TStop2.initialise(0,100,sigma)

                        np.save(SYM_PATH + "VTTS_"+str(mu_prior)+"_"+str(sd_prior)+"_"+str(lan_form)+"_Alt.npy",
                              np.array([np.mean(np.mean(suc_mat, axis=1) == 1),np.mean(np.sum(t_mat, axis=1)) * batch_size,
                                        suc_mat,t_mat]))
                    elif alg_id == 3:
                        # Random policy
                        reset_random_seeds()
                        suc_mat = np.zeros((N_iter,N_word))
                        t_mat = np.zeros((N_iter,N_word))
                        mu_bandits = bandits.generate_multiple_words(N_iter,N_word)
                        sigma = 1
                        for i in range(N_iter):
                            RP = RandomPolicy(J, 1, batch_size, 0.1/N_word)
                            RP.initialise(0,100,sigma)
                            for j in range(N_word):
                                while True:
                                    I = RP.select_arm()
                                    R = bandits.get_reward(mu_bandits[i,j], I)
                                    if RP.update(I, R):
                                        break
                                suc_mat[i,j] =  ((RP.top) == np.argmax(mu_bandits[i,j]))
                                t_mat[i,j] =  RP.t
                                arm_id = RP.top[0]
                                RP = RandomPolicy(J, 1, batch_size, 0.1/N_word)
                                RP.initialise(0,100,sigma)

                        np.save(SYM_PATH + "RP_"+str(mu_prior)+"_"+str(sd_prior)+"_"+str(lan_form)+"_Alt.npy",
                              np.array([np.mean(np.mean(suc_mat, axis=1) == 1),np.mean(np.sum(t_mat, axis=1)) * batch_size,
                                        suc_mat,t_mat]))