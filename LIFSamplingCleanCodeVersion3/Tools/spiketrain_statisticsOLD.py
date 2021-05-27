#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 11:56:04 2020

@author: achattoraj
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 14:21:37 2018

@author: achattoraj
"""
import numpy as np
def compute_cv_ff_corr(s,params, bnd):
    
    time_steps = np.shape(s)[2]#int(params.duration/params.sampling_bin_ms)
    num_reps = np.shape(s)[0]
    
    kk = 0
    isi_all = np.zeros((num_reps,params.N,time_steps))
    cv = np.array([])
    for i in range(params.N):
        isi = np.array([])
        for r in range(num_reps):
            if np.sum(np.squeeze(s[r,i,:]))>bnd :
                st = np.argwhere(np.squeeze(s[r,i,:])==1)[0,0]
                temp = st
                if st<time_steps:
                    for j in range(st+1,time_steps):
                        if s[r,i,j]==1:
                            isi = np.append(isi,float(j*params.sampling_bin_ms-temp*params.sampling_bin_ms))
                            isi_all[r,i,j] = float(j*params.sampling_bin_ms-temp*params.sampling_bin_ms)
                            temp = j
                        else:
                            isi_all[r,i,j] = -1
        if np.mean(np.sum(np.squeeze(s[:,i,:]),1))>bnd:
            add_cv = float(np.sqrt(np.var(isi)))/np.mean(isi)
            cv = np.append(cv,add_cv)
        else:
            cv = np.append(cv,-1)
            
#######################################################################################################################
#    cv = -1*np.ones((num_reps,params.N))
#    for i in range(params.N):
#        isi = np.array([])
#        for r in range(num_reps):
#            isi = []
#            if np.sum(np.squeeze(s[r,i,:]))>bnd :
#                st = np.where(np.squeeze(s[r,i,:])==1)[0][0]
#                temp = st
#                if st<time_steps:
#                    for j in range(st+1,time_steps):
#                        if s[r,i,j]==1:
#                            isi = np.append(isi,float(j*params.sampling_bin_ms-temp*params.sampling_bin_ms))
#                            isi_all[r,i,j] = float(j*params.sampling_bin_ms-temp*params.sampling_bin_ms)
#                            temp = j
#                        else:
#                            isi_all[r,i,j] = -1
#                    if np.mean(np.sum(np.squeeze(s[:,i,:]),1))>bnd:
#                        add_cv = float(np.sqrt(np.var(isi)))/np.mean(isi)
#                        cv[r,i] = add_cv
#                    else:
#                        cv[r,i] = -1
    
    print("Coefficient of variation done!")
    
   
    c_r = -100*np.ones((params.N,params.N))
    mspk = np.zeros((params.N,num_reps))
    
    for j in range(params.N):
        for k in range(num_reps):
            mspk[j,k] = np.sum(s[k,j,:])    
    for i in range(params.N):
        if np.mean(mspk[i,:])>bnd:
            for j in range(params.N):
                if j<i and np.mean(mspk[j,:])>bnd:
                    d = np.corrcoef(np.double(np.squeeze(mspk[i,:])),np.double(np.squeeze(mspk[j,:])))[0,1]
                    c_r[i,j] = d
    
    print("NoiseCorr done!")
                        
    
    ff = np.array([])
    for j in range(params.N):
        for k in range(num_reps):
            mspk[j,k] = np.sum(s[k,j,:])    
    for i in range(params.N):
        if np.mean(mspk[i,:])>bnd:
            ff = np.append(ff,float(np.var(mspk[i,:]))/np.mean(mspk[i,:]))
        else:
            ff = np.append(ff,-1)
            
    print("FanoFactor done!")
            
    
    return ff,cv,c_r,isi_all