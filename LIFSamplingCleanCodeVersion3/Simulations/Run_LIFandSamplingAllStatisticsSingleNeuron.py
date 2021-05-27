#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:40:01 2020

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
import itertools
from itertools import combinations

from Tools.params_generate import *
from Tools.load_params import *
from Tools.get_multi_info import *
from Tools.load_image_patches import *
from Tools.spiketrain_statistics import *

from Sampling_Functions.Marg_samples import *
from Sampling_Functions.Inference_Gibbs import * 
from Sampling_Functions.Pairwise_joint_samples import *
from Sampling_Functions.Pairwise_joint_difference import *

from LIF_Functions.LIF_spikes_from_BRIANSliding import *
from LIF_Functions.LIF_spikes_from_BRIAN import *
from LIF_Functions.LIF_with_BRIAN import LIFSamplingModel


colors = np.array(['green','blue','red'])
colors1 = np.flip(np.array(['greenyellow','lawngreen','limegreen','forestgreen','green','darkgreen']))
colors2 = np.flip(np.array(['dodgerblue','deepskyblue','cornflowerblue','royalblue','blue','darkblue']))
colors3 = np.flip(np.array(['lightcoral','indianred','firebrick','red','maroon','darkred']))

## Initialize variables    
path = '../Results/'
path1 = '../Results1/'
model_set =  ['NN', 'IN','ISN']
neurons_set = [128]#, 64]
contrasts = np.array([0.3, 0.4, 0.5, 0.75, 1.0])
angles = np.load('../Data/Grating_angles.npy')
pix = 64 #image pixels
dimension = 8 #sqrt(image pixels)
simulation_cases =  ['NaturalImage']



for m in range(len(model_set)):
    for nn in range(len(neurons_set)):
        neurons = neurons_set[nn]
        model = model_set[m]
        params = load_simulation_parameters(neurons,model)
        sample_hertz = 1.0/(params.sampling_bin_s)
       
        
        filename1 = 'SamplesforStatistics_NaturalImagesSingleNeuron' + '_' + str(neurons) + '_' + model + '.npy'
      
        filename2 = 'SpikesforStatistics_NaturalImagesSingleNeuron' + '_' + str(neurons) + '_' + model + '.npy'
        
        samples = np.load(path + filename1)
        spikes = np.load(path + filename2)

        num_im = np.shape(samples)[1]
        num_repeats = np.shape(samples)[0]
        n_samples = np.shape(samples)[3]
        bnd1 = 0
        bnd2 = 5
         
#                sampling_isi_separate = np.zeros((num_repeats,num_im,params.N,n_samples))
#                sampling_cv_separate = np.zeros((num_repeats,num_im,params.N))
        sampling_isi = np.zeros((num_im,num_repeats,params.N,n_samples))
        sampling_cv = np.zeros((num_im,params.N))
        sampling_ff = np.zeros((num_im,params.N))
        sampling_corr = np.zeros((num_im,params.N,params.N))
        sampling_sig_corr = np.zeros((num_im,params.N,params.N))
        
#                LIF_isi_separate = np.zeros((num_repeats,num_im,params.N,n_samples))
#                LIF_cv_separate = np.zeros((num_repeats,num_im,params.N))
        
        LIF_isi = np.zeros((num_im,num_repeats,params.N,n_samples))
        LIF_cv = np.zeros((num_im,params.N))
        LIF_ff = np.zeros((num_im,params.N))
        LIF_corr = np.zeros((num_im,params.N,params.N))
        LIF_sig_corr = np.zeros((num_im,params.N,params.N))
        
        
        for i in range(num_im):
            print("Counting image " + str(i+1) + " for model " + model)
            temp_samples = np.squeeze(samples[:,i,:,:])
            temp_spikes = np.squeeze(spikes[:,i,:,:])
            FF_smp, CV_smp, corr_smp, ISI_smp, sig_corr_smp = compute_cv_ff_corr(temp_samples,params,bnd=bnd1)
            FF_lif, CV_lif, corr_lif, ISI_lif, sig_corr_lif = compute_cv_ff_corr(temp_spikes,params,bnd=bnd2)

            sampling_isi[i,:,:,:] = ISI_smp
            sampling_cv[i,:] = CV_smp                    
            sampling_ff[i,:] = FF_smp
            sampling_corr[i,:,:] = corr_smp
            sampling_sig_corr[i,:,:] = sig_corr_smp
            
            
            LIF_isi[i,:,:,:] = ISI_lif
            LIF_cv[i,:] = CV_lif
            LIF_ff[i,:] = FF_lif
            LIF_corr[i,:,:] = corr_lif
            LIF_sig_corr[i,:,:] = sig_corr_lif
            
        np.save(path1 + 'SamplingStats_ISI_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy',sampling_isi)
        np.save(path1 + 'SamplingStats_CV_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy',sampling_cv)
        np.save(path1 + 'SamplingStats_FF_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy',sampling_ff)
        np.save(path1 + 'SamplingStats_Corr_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy',sampling_corr)
        np.save(path1 + 'SamplingStats_SigCorr_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy',sampling_sig_corr)
        
      
        np.save(path1 + 'LIFStats_ISI_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy',LIF_isi)
        np.save(path1 + 'LIFStats_CV_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy',LIF_cv)
        np.save(path1 + 'LIFStats_FF_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy',LIF_ff)
        np.save(path1 + 'LIFStats_Corr_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy',LIF_corr)
        np.save(path1 + 'LIFStats_SigCorr_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy',LIF_sig_corr)   
    
    