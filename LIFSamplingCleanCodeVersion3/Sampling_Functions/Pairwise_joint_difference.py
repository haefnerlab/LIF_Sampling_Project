#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 00:43:53 2020

@author: achattoraj
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 13:07:23 2018

@author: achattoraj
"""
import itertools
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from itertools import combinations
from Sampling_Functions.Marg_samples import *

def bool2int(x): 
        y = 0 
        for i,j in enumerate(x): 
            y += j<<i 
        return y 


def pairwise_prob_difference_compute(samples,N):
        n_samples = samples.shape[1]
        time_steps = n_samples
        comb = combinations(range(N), 2)
        comb = np.array(list(comb)) 
        prob_pair = np.zeros(comb.shape[0]*4)
        logprob_pair = np.zeros(comb.shape[0]*4)
        prob_diff = -100*np.ones(comb.shape[0]*4)
        logbndprob_diff = -100*np.ones(comb.shape[0]*4)
        p_marg = marg_prob_compute(samples,N)
        eps = 0.05
#        prob_00 = np.array([])
#        prob_01 = np.array([])
#        prob_10 = np.array([])
#        prob_11 = np.array([])
#        
#        logprob_00 = np.array([])
#        logprob_01 = np.array([])
#        logprob_10 = np.array([])
#        logprob_11 = np.array([])
        
        for i in range(comb.shape[0]):
            arr=np.zeros(4)
            pr=np.zeros(4)
            prlim=np.zeros(4)
            logbndpr=np.zeros(4)
            logpr=np.zeros(4)
            t = np.array([samples[comb[i,0],:],samples[comb[i,1],:]])
            arr[0] = np.size(np.where((t[0,:] == 0) & (t[1,:]==0))[0])
            arr[1] = np.size(np.where((t[0,:] == 0) & (t[1,:]==1))[0])
            arr[2] = np.size(np.where((t[0,:] == 1) & (t[1,:]==0))[0])
            arr[3] = np.size(np.where((t[0,:] == 1) & (t[1,:]==1))[0])
            arr = arr/sum(arr)
            
            logpr[0] = np.log(arr[0]+eps) - np.log(1-p_marg[comb[i,0]]+eps)-np.log(1-p_marg[comb[i,1]]+eps)
            logpr[1] = np.log(arr[1]+eps) - np.log(1-p_marg[comb[i,0]]+eps)-np.log(p_marg[comb[i,1]]+eps)
            logpr[2] = np.log(arr[2]+eps) - np.log(p_marg[comb[i,0]]+eps)-np.log(1-p_marg[comb[i,1]]+eps)
            logpr[3] = np.log(arr[3]+eps) - np.log(p_marg[comb[i,0]]+eps)-np.log(p_marg[comb[i,1]]+eps)
            
            
            
            pr[0] = (arr[0]) - (1-p_marg[comb[i,0]])*(1-p_marg[comb[i,1]])
            pr[1] = (arr[1]) - (1-p_marg[comb[i,0]])*(p_marg[comb[i,1]])
            pr[2] = (arr[2]) - (p_marg[comb[i,0]])*(1-p_marg[comb[i,1]])
            pr[3] = (arr[3]) - (p_marg[comb[i,0]])*(p_marg[comb[i,1]])
            
            if p_marg[comb[i,0]]>=0.05 and p_marg[comb[i,1]]>=0.05:
                
                prlim[0] = (arr[0]) - (1-p_marg[comb[i,0]])*(1-p_marg[comb[i,1]])
                prlim[1] = (arr[1]) - (1-p_marg[comb[i,0]])*(p_marg[comb[i,1]])
                prlim[2] = (arr[2]) - (p_marg[comb[i,0]])*(1-p_marg[comb[i,1]])
                prlim[3] = (arr[3]) - (p_marg[comb[i,0]])*(p_marg[comb[i,1]])
                
                prob_diff[i*4:i*4+4] = prlim
                
                logbndpr[0] = np.log(arr[0]+eps) - np.log(1-p_marg[comb[i,0]]+eps)-np.log(1-p_marg[comb[i,1]]+eps)
                logbndpr[1] = np.log(arr[1]+eps) - np.log(1-p_marg[comb[i,0]]+eps)-np.log(p_marg[comb[i,1]]+eps)
                logbndpr[2] = np.log(arr[2]+eps) - np.log(p_marg[comb[i,0]]+eps)-np.log(1-p_marg[comb[i,1]]+eps)
                logbndpr[3] = np.log(arr[3]+eps) - np.log(p_marg[comb[i,0]]+eps)-np.log(p_marg[comb[i,1]]+eps)
                
                logbndprob_diff[i*4:i*4+4] = prlim
            
#            prob_00 = np.append(prob_00,pr[0])
#            prob_01 = np.append(prob_01,pr[1])
#            prob_10 = np.append(prob_10,pr[2])
#            prob_11 = np.append(prob_11,pr[3])
#            
#            logprob_00 = np.append(logprob_00,logpr[0])
#            logprob_01 = np.append(logprob_01,logpr[1])
#            logprob_10 = np.append(logprob_10,logpr[2])
#            logprob_11 = np.append(logprob_11,logpr[3])
            
            
            prob_pair[i*4:i*4+4] = pr 
            logprob_pair[i*4:i*4+4] = logpr 
               
        return prob_pair, prob_diff, logprob_pair, logbndprob_diff#, prob_00,prob_01,prob_10,prob_11,logprob_00,logprob_01,logprob_10,logprob_11
        