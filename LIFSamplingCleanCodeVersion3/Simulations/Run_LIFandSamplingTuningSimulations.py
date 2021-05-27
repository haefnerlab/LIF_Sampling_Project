import sys
sys.path.append('../')
from brian2 import *
import numpy as np
import array
import scipy as sp
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import scipy.io
import seaborn as sns
from scipy.stats import norm
from matplotlib.colors import *

from Tools.load_params import *
from Tools.TuningComputingFunction import *

#Runs the initial script to initialize values
path = '../Results/'
model_set = ['NN','IN','ISN']
neurons_set = [128, 64]
pix = 64

sf_cases = np.load('../Data/Grating_sf.npy')
contrasts = np.load('../Data/Contrasts.npy')
num_repeats = 20
condition_duration = 25
condition_gray = 100
dimension = 8
    
#%%
for gp in range(len(sf_cases)):
    print("For sf case " + str(gp+1) + "/" + str(len(sf_cases)))
    print("=========================================================================================================")
    for sm in range(2):
        if sm==0:
            txt = "Sampling"
        else:
            txt = "LIF"
        for nn in range(len(neurons_set)):
            for m in range(len(model_set)):
                for k in range(len(contrasts)):
                    compute_tuning((neurons_set[nn],model_set[m],num_repeats,contrasts[k],gp,txt,condition_duration,condition_gray))
            
            
            
            
            
            