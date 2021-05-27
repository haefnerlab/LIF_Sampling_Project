#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 12:57:37 2018

@author: achattoraj
"""



import numpy as np
def marg_prob_compute(samples,N):
    prob_sample = np.zeros(N)
    n_samples = samples.shape[1]
    time_steps = n_samples
    for i in range(N):
        prob_sample[i] = float(np.sum(samples[i,:]))/time_steps
    return prob_sample









