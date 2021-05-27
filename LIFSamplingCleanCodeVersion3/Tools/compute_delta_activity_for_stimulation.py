#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:37:24 2019

@author: achattoraj
"""
import numpy as np
from numpy import random


def compute_delta_activity(G, selected, N, num_trials, fr_no_stim, fr_stim, intervals):
    sig_corr = []
    for j in range(N):
        mtch = np.dot(G[:,selected],G[:,j])
        sig_corr = np.append(sig_corr,mtch)

    influence = np.zeros((N,num_trials))
    delta_activity = np.zeros((N,num_trials))
    for i in range(N):
        if num_trials>1:
#            influence[i,:] = (fr_stim[:,i] - np.mean(fr_no_stim[:,i])) / np.std(fr_stim[:,i])
            influence[i,:] = (fr_stim[:,i] - np.mean(fr_no_stim[:,i])) / np.sqrt(np.var((fr_stim[:,i] - np.mean(fr_no_stim[:,i]))))

        else:
            influence[i,:] = (fr_stim[:,i] - np.mean(fr_no_stim[:,i]))
    mean_influence = np.mean(influence,1)

    sig_corr = np.delete(sig_corr,selected)
    mean_influence = np.delete(mean_influence,selected)
    temp_indx = np.abs(sig_corr)>0.45
    sig_corr = sig_corr[~temp_indx]
    mean_influence = mean_influence[~temp_indx]
    min_sig = np.min(sig_corr)
    max_sig = np.max(sig_corr)
    binned_mean_influence = np.zeros(intervals-1)
    binned_mid_sig_corr = np.zeros(intervals-1)
    new_sig_corr = np.linspace(min_sig,max_sig,intervals)
    cnt = np.zeros(intervals-1)
    err = np.zeros(intervals-1)
    for i in range(intervals-1):
        indx_true = []
        indx_true = np.where((sig_corr>=new_sig_corr[i]) & (sig_corr<new_sig_corr[i+1]) == True,1,0)
        cnt[i] = sum(indx_true)
        binned_mid_sig_corr[i] = (new_sig_corr[i] + new_sig_corr[i+1])/2
        binned_mean_influence[i] = sum(mean_influence[np.nonzero(indx_true)])/cnt[i]
        err[i] = np.std(mean_influence[np.nonzero(indx_true)])/np.sqrt(cnt[i])

        
    return sig_corr, mean_influence, binned_mid_sig_corr, binned_mean_influence, err
