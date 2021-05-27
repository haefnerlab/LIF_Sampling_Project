#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 17:55:22 2020

@author: achattoraj
"""

import sys
sys.path.append('../')
from brian2 import *
import numpy as np
prefs.codegen.target='numpy'  #important!! throws errors otherwise
import array
import scipy as sp
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import scipy.io
import seaborn as sns
from scipy.stats import norm
from matplotlib.colors import *

from Tools.params_generate import *
from Tools.load_params import *
#from Tools.load_gratings import *
from Tools.load_image_patches import *
from Tools.spiketrain_statistics import *

from Sampling_Functions.Marg_samples import *
from Sampling_Functions.Inference_Gibbs import * 

from LIF_Functions.LIF_spikes_from_BRIAN import *
from LIF_Functions.LIF_with_BRIAN import LIFSamplingModel


colors = np.array(['blue','lightcoral','red'])
## Initialize variables    
path = '../Results/'
model_set = ['ISN']
neurons_set = [128]#, 64]
nat_im_show = np.array([1017, 1732])
pix = 64 #image pixels
dimension = 8 #sqrt(image pixels)
rec_noise_cases = np.array([[0.0, 0.5, 0.9],[0.0, 0.5, 0.9]])
m = 0
#%%
for nn in range(len(neurons_set)):
    plt.figure()
    ##NaturalImage_data = load_natural_image_patches(8) # load natural image reprocessed patches
    NaturalImage_data = np.load('../Data/PreprocessedImagePatches-8-zca-norm.npy') # load natural image reprocessed patches
    chosen_natural_im = nat_im_show[nn]
    print('Chosen Natural Image number:' + str(chosen_natural_im) + ' for ' + str(neurons_set[nn]) + ' neurons')
    NaturalImage_chosen = np.squeeze(NaturalImage_data[chosen_natural_im,:,:])
    np.save(path + 'NaturalImage' + '_' + str(neurons_set[nn])  + '.npy',NaturalImage_chosen)
    natural_image_duration = 500 # duration of spike train simulation
    print('Computing for natural image...')
    print('----------------------------------------------------------------------------------------------------------------------------------------------')
    for rc in range(np.shape(rec_noise_cases)[1]):
        params = load_simulation_parameters(neurons_set[nn],model_set[m])
        params.rec_noise = rec_noise_cases[nn,rc] 
        print params.rec_noise
        n_samples = int(natural_image_duration/params.sampling_bin_ms)
 
        # Determines if initial samples in Sampling and membrane potentials in LIF simulations are assigned randomly
        random_init = 1 
        # not useful if ranom_init = 1 else, sets initial samples and membrane potential to values as in init_sample and init_membrane_potential
        init_sample = np.zeros(params.N)
        init_membrane_potential = -70 * np.ones(params.N)
        # Simulating Gibbs sampling and returning samples, feedforward input, recurrent input and Gibbs probability
        samplesNat, ff, rec, prob = Inference_Gibbs(params, NaturalImage_chosen, n_samples, random_init, init_sample)
        # Simulating LIF firing in BRIAN and returning spiketimes with corresponding neuron indices. 
        # Also monitoring voltage, probability, input current, post synaptic potential and firing rate
        M = LIFSamplingModel(params.N, params.G, init_membrane_potential, membrane_noise=0.0, prior=params.prior, pixel_noise=params.sigma_I , psp_length_ms=params.sampling_bin_ms, verbose=False, trial_duration_ms=natural_image_duration, photo_noise=params.photo_noise , photo_noise_refresh_ms=params.photo_noise_refresh_ms, rec_noise=params.rec_noise,random_init_process=random_init)
        M.condition(NaturalImage_chosen)
        spikesNat = M.simulate(monitor=["v", "P", "I", "psp", "FR", "is_active"])
#                print ('LIF done for natural image')
        times = np.array(spikesNat.t/ms)
        indices = np.array(spikesNat.i)
        # Obtaining binned spike train from BRIAN simulations
        spike_array_binnedNat = Extract_Spikes(params.N, natural_image_duration, params.sampling_bin_ms, spikesNat) 
#                print ('Spikes done for natural image')
        print('Sampling and LIF done for chosen natural image for model ' + str(model_set[m]) + ' with ' + str(neurons_set[nn]) + ' neurons.')
        print('==============================================================================================================================================')
        
        np.save(path + 'NaturalImageSamples' + '_' + str(params.rec_noise) + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',samplesNat)
        np.save(path + 'NaturalImageLIF_SpikeIndices' + '_' + str(params.rec_noise) + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',indices)
        np.save(path + 'NaturalImageLIF_SpikeTimes' + '_' + str(params.rec_noise) + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',times)
        
        
        
        plt.subplot(np.shape(rec_noise_cases)[1],1,rc+1)
        txt_load = 'Natural'
        if rc==0:
            indices = np.load(path + txt_load + 'ImageLIF_SpikeIndices' + '_' + str(neurons_set[nn]) +  '_' + 'IN' + '.npy')
            times = np.load(path + txt_load + 'ImageLIF_SpikeTimes' + '_' + str(neurons_set[nn]) +  '_' + 'IN' + '.npy') 
        if rc==np.shape(rec_noise_cases)[1]-1:
            indices = np.load(path + txt_load + 'ImageLIF_SpikeIndices' + '_' + str(neurons_set[nn]) +  '_' + 'ISN' + '.npy')
            times = np.load(path + txt_load + 'ImageLIF_SpikeTimes' + '_' + str(neurons_set[nn]) +  '_' + 'ISN' + '.npy') 
            
        gap = 1000
        for tt in range(neurons_set[nn]):
#            plt.plot(times[indices==tt],(tt+1)*np.ones(np.sum(indices==tt)),'|',markersize=3,color=colors[rc])
            plt.plot(np.array(times[indices==tt]),(tt*gap)*np.ones(np.sum(indices==tt)),'|',markeredgewidth=2.5,markersize=6,color=colors[rc])
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['left'].set_linewidth(2)
        plt.gca().spines['bottom'].set_linewidth(2)
    #            plt.xlim([0, int(natural_image_duration)])
    #            plt.ylim([1, neurons_set[nn]])
        plt.xlabel("Simulation time in ms",fontsize=18)
        plt.ylabel("Neurons",fontsize=18)
        plt.yticks(np.array([0*gap, 24*gap, 49*gap, 74*gap, 99*gap, 124*gap]),np.array([1, 25, 50, 75, 100, 125]),fontsize=18,fontweight='bold')
        plt.xticks([0, 250, 500],fontsize=18,fontweight='bold')
        plt.xlim([0, 500])
        plt.ylim([0, 128*gap])
        
#        plt.title("Raster for recurrent noise " + str(rec_noise_cases[nn,rc]))

                
                
                
                