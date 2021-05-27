import itertools
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
def bool2int(x): 
    y = 0 
    for i,j in enumerate(x): 
        y += j<<i 
    return y 

def sampling_prob_joint(samples):
    N = samples.shape[0]
    lst = list(itertools.product([0, 1], repeat= N))
    lst_arr = np.array(lst)
    time_steps = samples.shape[1]
    prob_sample = np.zeros(lst_arr.shape[0])
    for i in range(time_steps):
        flag = bool2int(samples[:,i].astype(np.int64))
        prob_sample[flag]=prob_sample[flag]+1
    return prob_sample/float(time_steps)