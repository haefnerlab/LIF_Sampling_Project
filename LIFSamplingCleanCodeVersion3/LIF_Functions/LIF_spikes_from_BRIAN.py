import numpy as np
import array
import scipy as sp
from brian2 import *
#Give number of neurons, duration of sampling, sampling bin size and spikes from BRIAN,
#returns array of spikes ,i.e, 0s and 1s of size neuron x bins
def Extract_Spikes(N,duration,sampling_bin_ms,spikes):
    
    start = [0]
    times = np.array(spikes.t/ms)
    indices = np.array(spikes.i)
    for s in start:
        
        spike_array_binned = np.zeros((N,int((duration-s)/(sampling_bin_ms))))
        for i,t in enumerate(times):
            neuro_indx = indices[i]
            bin_indx = int(t/sampling_bin_ms)
            spike_array_binned[neuro_indx,bin_indx] = 1
	
	return spike_array_binned



