import numpy as np
from spikes_from_BRIAN import *
def return_FF(N,spike_array_binned,rng,flag=0,cv_max_indx=0):
    # spike_array_binned = Extract_Spikes(N,duration_ms,sampling_bin_ms,t,spikes) 
    # mn = np.array([])
    # vr = np.array([])
    # window = int(duration_ms/rng)
    # ind = np.array(range(window))*rng
    # for i in range(N):
    # 	    m = np.array([])
    # 	    for k in range(ind.size-1):
    # 	        m = np.append(m,sum(spike_array_binned[i,ind[k]:ind[k+1]]))
    # 	    mn = np.append(mn,np.mean(m))
    # 	    vr = np.append(vr,np.var(m))
    # ff = np.array([])
    # for i in range(mn.size):
    #     if mn[i]>0:
    #         ff = np.append(ff,vr[i]/mn[i])
    # ff = ff[np.logical_not(np.isnan(ff))]
    # return ff   


    mn = np.array([])
    vr = np.array([])
    limit = spike_array_binned.shape[1]
    step = rng#int(rng/sampling_bin_ms)
    ff = np.array([])
    for i in range(N):
            m = np.array([])
            k=0
            while k<(limit-step):
                m = np.append(m,sum(spike_array_binned[i,k:k+step]))
                k = k+step
            mn = np.append(mn,np.mean(m))
            vr = np.append(vr,np.var(m))
            if flag==1 and i==cv_max_indx:
                ff_for_max_cv=vr[i]/mn[i]
            spk_counts_for_max_cv = m
            if flag==0 and i==cv_max_indx:
                ff_temp = vr[i]/mn[i]
            if mn[i]>0:
                ff = np.append(ff,vr[i]/mn[i])
                ff = ff[np.logical_not(np.isnan(ff))]
    
    if flag==0:    
        return ff,ff_temp
    else:
        return ff,ff_for_max_cv
        
        
        
