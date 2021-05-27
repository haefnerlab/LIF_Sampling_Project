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


def pairwise_prob_compute(samples,N,return_separated):
        n_samples = samples.shape[1]
        time_steps = n_samples
        comb = combinations(range(N), 2)
        comb = np.array(list(comb)) 
        prob_pair = np.zeros(comb.shape[0]*4)
        prob_00 = np.array([])
        prob_01 = np.array([])
        prob_10 = np.array([])
        prob_11 = np.array([])
        for i in range(comb.shape[0]):
            arr=np.zeros(4)
            t = np.array([samples[comb[i,0],:],samples[comb[i,1],:]])
            arr[0] = np.size(np.where((t[0,:] == 0) & (t[1,:]==0))[0])
            arr[1] = np.size(np.where((t[0,:] == 0) & (t[1,:]==1))[0])
            arr[2] = np.size(np.where((t[0,:] == 1) & (t[1,:]==0))[0])
            arr[3] = np.size(np.where((t[0,:] == 1) & (t[1,:]==1))[0])
            arr = arr/sum(arr)
            
            prob_00 = np.append(prob_00,arr[0])
            prob_01 = np.append(prob_01,arr[1])
            prob_10 = np.append(prob_10,arr[2])
            prob_11 = np.append(prob_11,arr[3])
            prob_pair[i*4:i*4+4] = arr   
            
        if return_separated==0:
            return prob_pair
        else:
            return prob_pair,prob_00,prob_01,prob_10,prob_11