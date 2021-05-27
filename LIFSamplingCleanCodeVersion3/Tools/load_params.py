## This file loads saved parameters PFs, sigma and pi.
## It assigns default values to parameters that will be used in simulations.
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
from os import path

from params_generate import *

def load_simulation_parameters(neurons,model, prior_prob=0.025, sigma_I = 0.25, duration = 1000,
                 sampling_bin_ms = 5.0, membrane_noise = 0.0, photo_noise = 0.0, photo_noise_refresh_ms = 5.0, rec_noise = 0.0):
    path = '../Data/'
    
    if model=='NN':
        pf_fl = 'weights_' + str(neurons) + '_0.00.npy'
        pi_fl = 'pi_' + str(neurons) + '_0.00.npy'
        sigma_fl = 'sigma_' + str(neurons) + '_0.00.npy'
        
        prior_prob = float(np.load(path + pi_fl))## initial firing probability of each neuron
        sigma_I = float(np.load(path + sigma_fl))######pixel_wise noise
        photo_noise = np.sqrt(0.00)
        PFs = np.load(path + pf_fl)
        shp = np.array(PFs.shape)   
        rec_noise = 0.0
    
    elif  model=='IN':
        pf_fl = 'weights_' + str(neurons) + '_0.00.npy' #we assume same weights are learned with changes only to pi and sigma, so not using :'weights_' + str(neurons) + '_0.05.npy'
        pi_fl = 'pi_' + str(neurons) + '_0.05.npy'
        sigma_fl = 'sigma_' + str(neurons) + '_0.05.npy'
        
        prior_prob = float(np.load(path + pi_fl))## initial firing probability of each neuron
        sigma_I = float(np.load(path + sigma_fl))######pixel_wise noise
        photo_noise = np.sqrt(0.05)
        PFs = np.load(path + pf_fl)
        shp = np.array(PFs.shape)   
        rec_noise = 0.0
   
    elif model=='ISN' :
        pf_fl = 'weights_' + str(neurons) + '_0.00.npy' #we assume same weights are learned with changes only to pi and sigma, so not using :'weights_' + str(neurons) + '_0.05.npy'
        pi_fl = 'pi_' + str(neurons) + '_0.05.npy'
        sigma_fl = 'sigma_' + str(neurons) + '_0.05.npy'
        
        prior_prob = float(np.load(path + pi_fl))## initial firing probability of each neuron
        sigma_I = float(np.load(path + sigma_fl))######pixel_wise noise
        photo_noise = np.sqrt(0.05)
        PFs = np.load(path + pf_fl)
        shp = np.array(PFs.shape) 
        rec_noise = 0.9
#        if neurons==64:
#            rec_noise = 0.9
#        if neurons==128:
#            rec_noise = 0.9
    
    G = np.transpose(np.reshape(PFs,(shp[0],shp[1]*shp[2])))
    N = neurons
        
    # Initialize other variables of interest 
    sampling_bin_ms = 5.0 # sampling bin size in ms
    duration = 1000 # duration in ms for which the model simulation runs

    params = params_generate(N, G, prior_prob, sigma_I, duration, sampling_bin_ms,  0.0, photo_noise, sampling_bin_ms, rec_noise)

    return params
