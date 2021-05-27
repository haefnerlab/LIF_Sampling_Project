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
import itertools
import multiprocessing

from Tools.load_params import *
from Tools.TuningComputingFunction import *

#Runs the initial script to initialize values
path = '../Results/'
model_set = ['NN']
neurons_set = [128, 64]
pix = 64
simul_case = ["Sampling","LIF"]

sf_cases = np.load('../Data/Grating_sf.npy')
sf_ind = range(len(sf_cases))
#sf_ind = np.array([0])
contrasts = np.load('../Data/Contrasts.npy')
num_repeats = [20]
condition_duration = [25]
condition_gray = [100]
dimension = [8]

paramlist = list(itertools.product(neurons_set,model_set,num_repeats,contrasts,sf_ind,simul_case,condition_duration,condition_gray))

#print paramlist[0]
pool = multiprocessing.Pool()
res = pool.map(compute_tuning, paramlist)
