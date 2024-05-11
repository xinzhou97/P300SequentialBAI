import os
os.environ['PYTHONHASHSEED']=str(1)
import random as rn
import numpy as np
def reset_random_seeds(seed=1):

    np.random.seed(seed)
    rn.seed(seed)
reset_random_seeds()

from agent_SBAI import *
from Bandit_env import *
from LanguageModel import *

N_iter= 200 # number of iterations
N_word = 20 # number of taotal typing words
sigma = 1 # standard error of the noise in the rewards
Delta=2 #sub-optimality gap
SYM_PATH = "./"
t_list = np.arange(5,101,5) # budget list
N_t = len(t_list)

# STTS Oracle

for J in [10,20]:
    for p0 in [1/J,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        lm = LanguageModel(J, p0)
        bandits = Bandits(J, Delta, p0)

        reset_random_seeds()
        batch_size = 1
        words_gen = bandits.generate_multiple_words(N_iter, N_word)
        res_final = np.zeros((N_iter, N_word, N_t))


        for i in range(N_iter):
            reset_random_seeds(i+1000)
            res = np.zeros((N_word, N_t))
            for j in range(N_word):
                if j == 0:
                    cur_arm = -1
                TStop2 = ThompsonSamplingTop2_P300(J, 1, batch_size,  0.1/N_word)
                # use language model to initialize
                TStop2.initialise(np.ones(J) * Delta, np.ones(J) * 0, 0.2, lm.get_probs(cur_arm), sigma)
                
                ind = 0
                while ind < N_t:
                    I = TStop2.select_arm()
                    R = bandits.get_reward(I, words_gen[i,j])
                    outp = TStop2.update(I, R)
                    if TStop2.t in t_list:
                        res[j, ind] = (TStop2.top) == (words_gen[i,j])
                        ind += 1


                # cur_arm is always the true arm so language model is always correct
                cur_arm = int(words_gen[i,j]) 
            res_final[i, :, :] = res.copy()



        np.save(SYM_PATH + "STTSOracle_"+str(J)+"_"+str(Delta)+"_"+str(p0)+"_FB.npy",
              res_final)

# STTS 

for J in [10,20]:
    for p0 in [1/J,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        lm = LanguageModel(J, p0)
        bandits = Bandits(J, Delta, p0)

        reset_random_seeds()
        batch_size = 1
        words_gen = bandits.generate_multiple_words(N_iter, N_word)
        res_final = np.zeros((N_iter, N_word, N_t))


        for i in range(N_iter):
            reset_random_seeds(i+1000)
            res = np.zeros((N_word, N_t))
            for j in range(N_word):
                if j == 0:
                    cur_arm = -1
                TStop2 = ThompsonSamplingTop2_P300(J, 1, batch_size,  0.1/N_word)
                # use language model to initialize
                TStop2.initialise(np.ones(J) * Delta, np.ones(J) * 0, 0.2, lm.get_probs(cur_arm), sigma)
                
                ind = 0
                while ind < N_t:
                    I = TStop2.select_arm()
                    R = bandits.get_reward(I, words_gen[i,j])
                    outp = TStop2.update(I, R)
                    if TStop2.t in t_list:
                        res[j, ind] = (TStop2.top) == (words_gen[i,j])
                        ind += 1


                # cur_arm is the identified arm 
                cur_arm = TStop2.top
            res_final[i, :, :] = res.copy()



        np.save(SYM_PATH + "STTS_"+str(J)+"_"+str(Delta)+"_"+str(p0)+"_FB.npy",
              res_final)

# VTTS 

for J in [10,20]:
    for p0 in [1/J,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        lm = LanguageModel(J, p0)
        bandits = Bandits(J, Delta, p0)

        reset_random_seeds()
        batch_size = 1
        words_gen = bandits.generate_multiple_words(N_iter, N_word)
        res_final = np.zeros((N_iter, N_word, N_t))


        for i in range(N_iter):
            reset_random_seeds(i+1000)
            res = np.zeros((N_word, N_t))
            for j in range(N_word):
                if j == 0:
                    cur_arm = -1
                TStop2 = ThompsonSamplingTop2_P300(J, 1, batch_size,  0.1/N_word)
                # do not use language model to initialize
                TStop2.initialise(np.ones(J) * Delta, np.ones(J) * 0, 0.2, np.ones(J) / J, sigma)
                
                ind = 0
                while ind < N_t:
                    I = TStop2.select_arm()
                    R = bandits.get_reward(I, words_gen[i,j])
                    outp = TStop2.update(I, R)
                    if TStop2.t in t_list:
                        res[j, ind] = (TStop2.top) == (words_gen[i,j])
                        ind += 1


                # cur_arm is the identified arm 
                cur_arm = TStop2.top
            res_final[i, :, :] = res.copy()



        np.save(SYM_PATH + "VTTS_"+str(J)+"_"+str(Delta)+"_"+str(p0)+"_FB.npy",
              res_final)

# Random policy

for J in [10,20]:
    for p0 in [1/J,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        lm = LanguageModel(J, p0)
        bandits = Bandits(J, Delta, p0)

        reset_random_seeds()
        batch_size = 1
        words_gen = bandits.generate_multiple_words(N_iter, N_word)
        res_final = np.zeros((N_iter, N_word, N_t))


        for i in range(N_iter):
            reset_random_seeds(i+1000)
            res = np.zeros((N_word, N_t))
            for j in range(N_word):
                if j == 0:
                    cur_arm = -1
                RP = RandomPolicy(J, 1, batch_size,  0.1/N_word)
                RP.initialise(np.ones(J) * Delta, np.ones(J) * 0, 0.2, np.ones(J) / J, sigma)
                
                ind = 0
                while ind < N_t:
                    I = RP.select_arm()
                    R = bandits.get_reward(I, words_gen[i,j])
                    outp = RP.update(I, R)
                    if RP.t in t_list:
                        res[j, ind] = (RP.top) == (words_gen[i,j])
                        ind += 1


                # cur_arm is the identified arm 
                cur_arm = RP.top
            res_final[i, :, :] = res.copy()



        np.save(SYM_PATH + "RP_"+str(J)+"_"+str(Delta)+"_"+str(p0)+"_FB.npy",
              res_final)

# Batch Racing (BR)

for J in [10,20]:
    for p0 in [1/J,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        lm = LanguageModel(J, p0)
        bandits = Bandits(J, Delta, p0)

        reset_random_seeds()
        batch_size = 1
        words_gen = bandits.generate_multiple_words(N_iter, N_word)
        res_final = np.zeros((N_iter, N_word, N_t))


        for i in range(N_iter):
            reset_random_seeds(i+1000)
            res = np.zeros((N_word, N_t))
            for j in range(N_word):
                if j == 0:
                    cur_arm = -1
                BR = BatchRacing(J, 1, batch_size, 0.1/N_word,1)
                
                ind = 0
                while ind < N_t:
                    I = BR.select_arm()
                    R = bandits.get_reward(I, words_gen[i,j])
                    outp = BR.update(I, R)
                    if BR.t in t_list:
                        res[j, ind] = (BR.top) == (words_gen[i,j])
                        ind += 1

                # cur_arm is the identified arm 
                cur_arm = BR.top
            res_final[i, :, :] = res.copy()



        np.save(SYM_PATH + "BR_"+str(J)+"_"+str(Delta)+"_"+str(p0)+"_FB.npy",
              res_final)