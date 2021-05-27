# -*- coding: utf-8
#
# LinCA run for the MCA/BSC Journal Paper
#
# import numpy as np
from __future__ import division

# Training-data
data_factor = 1.
datafile = "../data/patches-20-dog-norm.h5"
N = 100000

# to initialize W
# mat = np.load('weights64_0.00.npy')
# mat = np.squeeze(mat[-1,:,:,:])
# shp = np.array(mat.shape)
# G = np.transpose(np.reshape(mat,(shp[0],shp[1]*shp[2])))

# Model to use
from pulp.em.camodels.bsc_et import BSC_ET
model_class=BSC_ET

# Number of hidden causes
H = 400


# Which parameters should be learned:
to_learn = ['W', 'pi', 'sigma']

# ET approximation parameters
Hprime = 8 #8
gamma = 8

#In this case, we need an estimation for the parameters
W_init = 'estimate'
pi_init = 'estimate'
sigma_init = 'estimate'


# np.random.normal(scale=W_noise_intensity, size=(H, D)) is added after each run
W_noise     = 0.0
pi_noise    = 0.0
sigma_noise = 0.0


# Annealing:
temp_start = 1.0
temp_end = 1.0

anneal_steps = 500
anneal_start = 20
anneal_end   = 80

cut_start = 1./3
cut_end = 2./3

noise_decrease = 80
noise_end = 90

anneal_prior = False

iter_noise = 0.0


# Images used:
channel_splitted = False #This means single channel (this convention'll be reversed)

# Post-Processing:
s_p = 3.
s_m = 1.


processing = 'deconvolve'
fit = False #True

