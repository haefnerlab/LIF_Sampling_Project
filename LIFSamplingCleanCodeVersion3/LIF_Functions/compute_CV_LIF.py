#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 15:01:21 2018

@author: achattoraj
"""
from spikes_from_BRIAN import *
import numpy as np

## actual stuff with spike array
#def return_CV(N, s,sampling_bin_ms):
#    cv1 = np.array([])
#    for i in range(N):
#        temp = 0
#        isi = np.array([])
#        time_steps = s.shape[1]#int(duration_ms/sampling_bin_ms)
#        for j in range(time_steps):
#            if s[i,j]==1:
#                isi = np.append(isi,((j+1)*sampling_bin_ms-temp*sampling_bin_ms))
#                temp = j+1
#        if np.mean(isi)>0 and not np.isnan(np.mean(isi)):
#            cv1 = np.append(cv1,np.sqrt(np.var(isi))/np.mean(isi))
#    return cv1

def return_CV(N, times,indices,flag=0):
    cv = np.array([])
    mn = np.zeros(N)
    cv_max=-1
    for o in range(N):
        sp = np.array(times[indices==o])
        interval = []
        for j in range(len(sp)-1):
            interval.append(sp[j+1]-sp[j])
        m = np.mean(interval)
        mn[o] = m
        v = np.sqrt(np.var(interval))
        if m>0 and not np.isnan(v):
            cv = np.append(cv,float(v/m))
            add_cv=np.sqrt(np.var(interval))/np.mean(interval)
            cv = np.append(cv,add_cv)
            if flag==1:
                if add_cv>=cv_max:
                    cv_max=add_cv
                    isi_for_max_cv=interval
                    cv_max_indx=o
    if flag==0:
        return mn,cv
    else:
        return mn,cv,cv_max,isi_for_max_cv,cv_max_indx
    
    