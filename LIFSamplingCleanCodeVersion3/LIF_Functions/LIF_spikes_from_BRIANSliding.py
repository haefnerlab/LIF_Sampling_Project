import numpy as np
import array
import scipy as sp
from brian2 import *
def Extract_Spikes_Sliding(N,duration,sampling_bin_ms,spikes):
    
    times = np.array(spikes.t/ms)
    indices = np.array(spikes.i)
    
    spike_array_binned = np.zeros((N,int(duration-sampling_bin_ms+1)))
    
    for i in range(len(times)):
        t = times[i]
        left_ind = range(max(0,int(t-sampling_bin_ms+1)),int(min(t+1,duration-sampling_bin_ms+1)))
        for l in left_ind:
            spike_array_binned[indices[i]][l] = 1
            
	
    return spike_array_binned



    
    
    
#    for ii in range(int(duration-sampling_bin_ms)):
#    for t in range(len(times)):
#        if times[t]>=(ii) and times[t]<(ii+sampling_bin_ms):
#            neuro_indx = indices[t]
#            spike_array_binned[neuro_indx,ii] = 1
                
    
    
    
    
#    
#    for s in start:
#        
#        spike_array_binned = np.zeros((N,int((duration-s)/(sampling_bin_ms))))
#        for i,t in enumerate(times):
#            neuro_indx = indices[i]
#            bin_indx = int(t/sampling_bin_ms)
#            spike_array_binned[neuro_indx,bin_indx] = 1