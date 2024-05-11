import numpy as np

class ThompsonSamplingTop2_P300():
    '''
    Class to implement a Thompson sampling policy top 2 in the batch case.
    

    Args:
        N_arms: number of total arms 
        N_top: number of picked top arms
        batch_size: the batch size, which is 1 in the experiments.
        delta: 1-delta is the confidence level for each task
        tmax: the maximum number of rounds for each task.
    '''
    def __init__(self, N_arms, N_top, batch_size, delta, tmax=10000):
        self.N_arms = N_arms
        self.N_top = N_top
        self.batch_size = batch_size
        self.delta = delta
        self.tmax = tmax
        return

    def initialise(self, mu0, mu1, Sigma, p0, sigma, beta=1/2):
        '''
        Initialise the arms

        Args:
            mu0: mu0 in prior distribution
            mu1: mu1 in prior distribution
            Sigma:Sigma in prior distribution
            p0: parameter in prior specification
            sigma: sd of the noise in the rewards
            beta: prob in TS top2 sampling rule
        '''
        self.Sigma = np.array([Sigma for arm in range(self.N_arms)]) * 1.0
        self.Sigma_inv = np.array([1/Sigma for arm in range(self.N_arms)]) * 1.0
        self.sigma = sigma
        self.mu0 = mu0
        self.mu1 = mu1
        self.mu0_arg2 = self.Sigma_inv * self.mu0
        self.mu1_arg2 = self.Sigma_inv * self.mu1
        self.beta = beta

        self.s = np.sqrt(self.Sigma)
        self.p0 = p0
        self.m0 = self.mu0
        self.m1 = self.mu1
        self.counts = np.zeros(self.N_arms)
        self.avg = np.zeros(self.N_arms)
        self.s_prior = self.s
        self.mu0_prior = self.mu0
        self.mu1_prior = self.mu1
        self.t = 0
        self.finish = False
        return

    def _calculate_Z(self, index_i):
        ## return Z_{i,j} as a vector given index_i
        m = self.p0 * self.m0 + (1-self.p0) * self.m1
        m_var = self.p0 * (self.m0 ** 2) + (1-self.p0) * (self.m1 ** 2) - m ** 2
        res = (m[index_i]-m)/np.sqrt(self.s[index_i]**2 + self.s**2 + m_var[index_i] + m_var)
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

        
        m_top2 = -np.sort(-(self.p0 * self.m0 + (1-self.p0) * self.m1))[:2]

        
        ## select arms
        while len(I_pick) < self.batch_size:
            n0 = len(I_remaining)
            p0_curr = self.p0[I_remaining]
            m0_curr = self.m0[I_remaining]
            m1_curr = self.m1[I_remaining]
            s_curr = self.s[I_remaining]
            
            samp_p = np.array([np.random.binomial(1,p,1)[0] for p in p0_curr])
            samp = np.random.normal(m0_curr, s_curr, n0) * samp_p + np.random.normal(m1_curr, s_curr, n0) * (1-samp_p)
            
            I1_ind = np.argmax(samp)
            I2_ind = I1_ind
            count = 0
            while I1_ind == I2_ind:
                samp_p = np.array([np.random.binomial(1,p,1)[0] for p in p0_curr])
                samp = np.random.normal(m0_curr, s_curr, n0) * samp_p + np.random.normal(m1_curr, s_curr, n0) * (1-samp_p)           
                I2_ind = np.argmax(samp)

                count += 1
                if count >= 100 or self.finish:          
                    I2_ind = np.argsort(-samp)[1]

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
        self.Sigma_inv[I] = self.Sigma_inv[I] + 1 *  self.sigma**(-2)
        self.Sigma[I] = self.Sigma_inv[I] ** (-1)
        score = R
        score[np.isnan(score)] = 0
        self.mu0_arg2[I] = self.mu0_arg2[I] + score[I] * self.sigma**(-2)
        self.mu0[I] = self.Sigma[I] * self.mu0_arg2[I]
        self.mu1_arg2[I] = self.mu1_arg2[I] + score[I] * self.sigma**(-2)
        self.mu1[I] = self.Sigma[I] * self.mu1_arg2[I]
        self.s[I] = np.sqrt(self.Sigma[I])
        self.m0[I] = self.mu0[I]
        self.m1[I] = self.mu1[I]
        self.top = (-(self.p0 * self.m0 + (1-self.p0) * self.m1)).argsort()[:self.N_top]
        self.t += 1
        
        self.counts += counts
        self.avg[I] = (self.avg[I] * (self.counts[I] - 1) + score[I])/self.counts[I]
        PR = (self.s_prior[I] ** 2 + 1/self.counts[I] * self.sigma**(2)) ** (-1)
        C0 = np.sqrt(PR) * np.exp(-PR/2* ((self.avg[I] - self.mu0[I]) ** 2) )
        C1 = np.sqrt(PR) * np.exp(-PR/2* ((self.avg[I] - self.mu1[I]) ** 2) )
        p0_new = (self.p0[I] * C0) / (self.p0[I] * C0 + (1-self.p0[I]) * C1)
        self.p0[I] = p0_new


        ## stopping rule
        Z_min = np.min([[np.delete(self._calculate_Z(ind), self.top)] 
                                        for ind in self.top])
        

        gamma_t = np.sqrt(2*np.log(np.log(self.t*self.batch_size)/self.delta))
        thresh = gamma_t
        self.finish = (Z_min >= thresh or self.t >=self.tmax)
        return(self.finish)

class RandomPolicy():
    def __init__(self, N_arms, N_top, batch_size, delta, tmax=10000):
        self.N_arms = N_arms
        self.N_top = N_top
        self.batch_size = batch_size
        self.delta = delta
        self.tmax = tmax
        return

    def initialise(self, mu0, mu1, Sigma,p0, sigma, beta=1/2):
        '''
        Initialise the arms

        Args:
            mu: mu in prior distribution
            Sigma:Sigma in prior distribution
            sigma: sd of the noise in the rewards
            beta: prob in TS top2 sampling rule
        '''
        self.Sigma = np.array([Sigma for arm in range(self.N_arms)]) * 1.0
        self.Sigma_inv = np.array([1/Sigma for arm in range(self.N_arms)]) * 1.0
        self.sigma = sigma
        self.mu0 = mu0
        self.mu1 = mu1
        self.mu0_arg2 = self.Sigma_inv * self.mu0
        self.mu1_arg2 = self.Sigma_inv * self.mu1
        self.beta = beta

        self.s = np.sqrt(self.Sigma)
        self.p0 = p0
        self.m0 = self.mu0
        self.m1 = self.mu1
        self.counts = np.zeros(self.N_arms)
        self.avg = np.zeros(self.N_arms)
        self.s_prior = self.s
        self.mu0_prior = self.mu0
        self.mu1_prior = self.mu1
        self.t = 0
        self.finish = False
        return

    def _calculate_Z(self, index_i):
        ## return Z_{i,j} as a vector given index_i
        m = self.p0 * self.m0 + (1-self.p0) * self.m1
        m_var = self.p0 * (self.m0 ** 2) + (1-self.p0) * (self.m1 ** 2) - m ** 2
        res = (m[index_i]-m)/np.sqrt(self.s[index_i]**2 + self.s**2 + m_var[index_i] + m_var)
        return(res)


    
    def select_arm(self):
        """Selects an arm for each round.
                
        Returns:
                An vector corresponding to the index of the selected 
                arm in the current batch
        """

        #I_pick = np.random.permutation(np.arange(self.N_arms)).astype(int)
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
        self.Sigma_inv[I] = self.Sigma_inv[I] + 1 *  self.sigma**(-2)
        self.Sigma[I] = self.Sigma_inv[I] ** (-1)
        score = R
        score[np.isnan(score)] = 0
        self.mu0_arg2[I] = self.mu0_arg2[I] + score[I] * self.sigma**(-2)
        self.mu0[I] = self.Sigma[I] * self.mu0_arg2[I]
        self.mu1_arg2[I] = self.mu1_arg2[I] + score[I] * self.sigma**(-2)
        self.mu1[I] = self.Sigma[I] * self.mu1_arg2[I]
        self.s[I] = np.sqrt(self.Sigma[I])
        self.m0[I] = self.mu0[I]
        self.m1[I] = self.mu1[I]
        self.top = (-(self.p0 * self.m0 + (1-self.p0) * self.m1)).argsort()[:self.N_top]
        self.t += 1
        
        self.counts += counts
        self.avg[I] = (self.avg[I] * (self.counts[I] - 1) + score[I])/self.counts[I]
        PR = (self.s_prior[I] ** 2 + 1/self.counts[I] * self.sigma**(2)) ** (-1)
        C0 = np.sqrt(PR) * np.exp(-PR/2* ((self.avg[I] - self.mu0[I]) ** 2) )
        C1 = np.sqrt(PR) * np.exp(-PR/2* ((self.avg[I] - self.mu1[I]) ** 2) )
        p0_new = (self.p0[I] * C0) / (self.p0[I] * C0 + (1-self.p0[I]) * C1)
        self.p0[I] = p0_new


        ## stopping rule
        Z_min = np.min([[np.delete(self._calculate_Z(ind), self.top)] 
                                        for ind in self.top])
        

        gamma_t = np.sqrt(2*np.log(np.log(self.t*self.batch_size)/self.delta))
        thresh = gamma_t
        self.finish = (Z_min >= thresh or self.t >=self.tmax)
        return(self.finish)


class BatchRacing():
    def __init__(self, N_arms, N_top, batch_size, delta,sigma, tmax=5000):
        self.N_arms = N_arms
        self.N_top = N_top
        self.batch_size = batch_size
        self.delta = delta
        self.sigma = sigma
        self.t = 0
        self.S = np.arange(self.N_arms)
        self.R = np.empty(0)
        self.A = np.empty(0)
        self.Ti = np.zeros(self.N_arms)
        self.mu = np.zeros(self.N_arms)
        self.tot_len = 0
        self.deviation = np.vectorize(self.deviation_naive)
        self.tmax = tmax
        return
    @staticmethod  
    def RoundRobin_P300(A, Ti, b, n):
        a = np.zeros(n)
        ind  = Ti.argsort()[:b]
        a[A[ind]] = 1
        return(a)
    
    @staticmethod  
    def deviation_naive(tao, w, sigma = 1):
        if tao == 0:
            return(np.inf)
        return(4 * sigma *np.sqrt(np.log(np.log2(2*tao)/w)/tao) * 0.25)
    

    def select_arm(self):
        """Selects an arm for each round.
                
        Returns:
                An vector corresponding to the index of the selected 
                arm in the current batch
        """
        a = self.RoundRobin_P300(self.S, self.Ti[self.S], self.batch_size, self.N_arms)
        I_pick = np.arange(self.N_arms)[a>0] 
        return(I_pick)
        
    def update(self, I, R):
        '''
        Args:
            I: pulled arms of current batch (with order in the batch)
            R: rewards= vector for the current batch
        '''
        score = R
        self.Ti[I] += 1
        kt = self.N_top - len(self.A)
        self.mu[I] = (self.mu[I] * (self.Ti[I]-1) + score[I]) / self.Ti[I]
        self.top = np.argmax(self.mu)
        deviation_S = self.deviation(self.Ti[self.S], np.sqrt(self.delta/(6*self.N_arms)),self.sigma)
        L_Ti = self.mu[self.S] - deviation_S 
        U_Ti = self.mu[self.S] + deviation_S
        ind_add = np.where(L_Ti > (-np.sort(-U_Ti))[kt])
        ind_rem = np.where(U_Ti < (-np.sort(-L_Ti))[kt-1])
        self.A = np.append(self.A, self.S[ind_add])
        self.R = np.append(self.R,self.S[ind_rem])
        self.S = np.delete(self.S, np.append(ind_add, ind_rem))
        self.t += 1
        self.tot_len += len(I)
        if self.N_top == 1 and self.t >= self.tmax:
            self.A = np.array([np.argmax(self.mu)])
            return(True)
        return(len(self.S) == 0)
