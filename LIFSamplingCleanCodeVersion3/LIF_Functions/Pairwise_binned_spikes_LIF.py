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

def bool2int(x): 
        y = 0 
        for i,j in enumerate(x): 
            y += j<<i 
        return y 


def pairwise_prob_compute(samples,N):
        n_samples = samples.shape[1]
        time_steps = n_samples
        comb = combinations(range(N), 2)
        comb = np.array(list(comb)) 
        prob_pair = np.zeros(comb.shape[0]*4)
        for i in range(comb.shape[0]):
            arr=np.zeros(4)
            t = np.array([samples[comb[i,0],:],samples[comb[i,1],:]])
            arr[0] = np.size(np.where((t[0,:] == 0) & (t[1,:]==0))[0])
            arr[1] = np.size(np.where((t[0,:] == 0) & (t[1,:]==1))[0])
            arr[2] = np.size(np.where((t[0,:] == 1) & (t[1,:]==0))[0])
            arr[3] = np.size(np.where((t[0,:] == 1) & (t[1,:]==1))[0])
            arr = arr/sum(arr)
            prob_pair[i*4:i*4+4] = arr         
        return prob_pair