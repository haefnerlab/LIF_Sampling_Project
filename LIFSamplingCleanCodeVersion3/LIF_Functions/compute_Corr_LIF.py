#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 14:40:38 2018

@author: achattoraj
"""
import numpy as np
import math
from spikes_from_BRIAN import *

def return_Corr(N, s, rng,flag=0,cv_max_indx=0):
    # s = Extract_Spikes(N,duration_ms,sampling_bin_ms,t,spikes)
    corr_for_max_cv=np.array([])
    ind = np.array(range(s.shape[1]/rng))
    ind = ind * rng
    c_r = np.array([])
    mspk = np.zeros((N,ind.size))
    for j in range(N):
        m = np.array([])
        for k in range(ind.size):
            mspk[j,k] = sum(s[j,ind[k]:ind[k]+rng])    
    for m in range(N):
        for n in range(N):
            if m!=n:
                d = np.corrcoef(mspk[m,:],mspk[n,:])[0,1]
                if ~math.isnan(d):
                    c_r = np.append(c_r,d)
                    if flag==1 and m==cv_max_indx:
                        corr_for_max_cv=np.append(corr_for_max_cv,d)
    c_r = c_r[np.logical_not(np.isnan(c_r))]
    corr_for_max_cv = corr_for_max_cv[np.logical_not(np.isnan(corr_for_max_cv))]
    indx=np.where(c_r==1)
    c_r = np.delete(c_r,indx)
    if flag==1:
        return c_r,corr_for_max_cv
    else:
        return c_r