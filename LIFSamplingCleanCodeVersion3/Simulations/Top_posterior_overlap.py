#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:39:34 2020

@author: achattoraj
"""

import sys
sys.path.append('../')
from brian2 import *
import numpy as np
import array
import scipy as sp
import scipy.stats
from scipy import stats
import itertools
from itertools import combinations
from scipy.misc import imshow
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import scipy.io
from Tools.params_generate import *
from Tools.spiketrain_statistics import *
from Sampling_Functions.Inference_Gibbs import *
from Tools.load_params import *
from Tools.load_image_patches import *
from LIF_Functions.LIF_spikes_from_BRIAN import *
from LIF_Functions.LIF_with_BRIAN import LIFSamplingModel
from Sampling_Functions.Inference_Gibbs import *
from Sampling_Functions.Marg_samples import *
from Sampling_Functions.Pairwise_joint_samples import *
from Sampling_Functions.Pairwise_joint_difference import *
#from LIF_Functions.Pairwise_binned_spikes_LIF import *
#from LIF_Functions.Marg_binned_spikes_LIF import *

path = '../Results/'
model_set = ['NN', 'IN','ISN']
neurons_set = [128]
pix = 64
chosen_angle_set = [0, 5, 10, 15, 20]
contrast = 1.0
sf_case = 1
repeat_num = 0
top_n = 5
for nn in range(len(neurons_set)):
    for m in range(len(model_set)):  
        neurons = neurons_set[nn]
        model = model_set[m]
        params = load_simulation_parameters(neurons,model)
        sample_hertz = 1.0/(params.sampling_bin_s)
        
        dimension = 8
        num_im = 25
        temp_data = np.load('../Data/SuitableNaturalImages_'+str(neurons_set[nn])+'.npy')  
#        temp_data = load_natural_image_patches(8)
        chosen = range(num_im)#np.random.randint(0,np.shape(temp_data)[0],num_im)  1,5,8,11
        print('Chosen indices:'+str(chosen))
        data = temp_data[chosen,:,:]
        params.duration = 1000
        
        n_samples = int(params.duration/params.sampling_bin_ms)
        rec_noise_sample = params.rec_noise
        photo_noise_sample = params.photo_noise
        samples_im = np.zeros((num_im,params.N,n_samples))
        comb = combinations(range(params.N), 2)
        comb = np.array(list(comb)) 
        prob_pair_length = (comb.shape[0]*4)
        
        comb1 = combinations(range(top_n), 2)
        comb1 = np.array(list(comb1)) 
        prob_pair_length1 = (comb1.shape[0]*4)

        Sampling_marginal_prob = np.zeros((num_im,params.N))
        Sampling_pairwise_joint_prob = np.zeros((num_im,prob_pair_length))
        Sampling_pairwise_joint_prob00 = np.zeros((num_im,int(prob_pair_length/4)))
        Sampling_pairwise_joint_prob01 = np.zeros((num_im,int(prob_pair_length/4)))
        Sampling_pairwise_joint_prob10 = np.zeros((num_im,int(prob_pair_length/4)))
        Sampling_pairwise_joint_prob11 = np.zeros((num_im,int(prob_pair_length/4)))
        Sampling_pairwise_diff = np.zeros((num_im,prob_pair_length))
        Sampling_pairwise_logdiff = np.zeros((num_im,prob_pair_length))
        SubSampling_pairwise_diff = np.zeros((num_im,prob_pair_length1))
        SubSampling_pairwise_logdiff = np.zeros((num_im,prob_pair_length1))
        for i in range(num_im):
            print i
            random_init = 1
            init_sample = np.zeros(params.N)
            init_membrane_potential = -75 * np.ones(params.N)
            Image = np.reshape(data[i,:,:],(64))# + sigma_I**2 * np.random.normal(0,1,pix)
        
            samples,_,_,_ = Inference_Gibbs(params.G, params.sigma_I, photo_noise_sample, params.prior, params.N, Image, n_samples, rec_noise_sample, random_init, init_sample)
            print ('Sampling Done')
            samples_im[i,:,:] = samples           
#            M = LIFSamplingModel(params.N, params.G, init_membrane_potential, membrane_noise=0.0, prior=params.prior, pixel_noise=params.sigma_I , psp_length_ms=params.sampling_bin_ms, verbose=True, trial_duration_ms=params.duration, photo_noise=params.photo_noise , photo_noise_refresh_ms=params.photo_noise_refresh_ms, rec_noise=params.rec_noise, random_init_process=random_init)
#            M.condition(Image)
#            spikes = M.simulate(monitor=["v", "P", "I", "psp", "FR"])
#            print ('LIF Done')
#            
#            times = np.array(spikes.t/ms)
#            indices = np.array(spikes.i)
#            spike_array_binned = Extract_Spikes(params.N, params.duration, params.sampling_bin_ms, spikes)  
#            print ('Spikes Done')
                        
            Sampling_marginal_prob[i,:] = marg_prob_compute(samples,params.N) 
#            LIF_marginal_prob[i,:] = marg_prob_compute(spike_array_binned,params.N) 
            
            Sampling_pairwise_joint_prob[i,:],Sampling_pairwise_joint_prob00[i,:],Sampling_pairwise_joint_prob01[i,:],Sampling_pairwise_joint_prob10[i,:],Sampling_pairwise_joint_prob11[i,:] = pairwise_prob_compute(samples,params.N,1) 
#            LIF_pairwise_joint_prob[i,:],LIF_pairwise_joint_prob00[i,:],LIF_pairwise_joint_prob01[i,:],LIF_pairwise_joint_prob10[i,:],LIF_pairwise_joint_prob11[i,:] = pairwise_prob_compute(spike_array_binned,params.N,1) 
            
            Sampling_pairwise_diff[i,:],Sampling_pairwise_logdiff[i,:] = pairwise_prob_difference_compute(samples,params.N)
#            LIF_pairwise_diff[i,:],LIF_pairwise_logdiff[i,:] = pairwise_prob_difference_compute(spike_array_binned,params.N)
            
            
            highest_fr_indx_samples = np.flip(np.argsort(sum(samples,1)))
            tmp_indx = highest_fr_indx_samples[:top_n]
            tmp_indx = tmp_indx.astype(int)
            samples_sub = samples[tmp_indx,:]
            
            SubSampling_pairwise_diff[i,:],SubSampling_pairwise_logdiff[i,:] = pairwise_prob_difference_compute(samples_sub,top_n)            
            
            
            
        np.save(path + 'AllSamples25Im' + '_' + str(neurons) + '_' + model + '.npy',samples_im)
        np.save(path + 'AllSamples25Im_pairwise' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_joint_prob)
        np.save(path + 'AllSamples25Im_pairwise_diff' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_diff)        
        np.save(path + 'AllSamples25Im_pairwise_logdiff' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_logdiff)        
        np.save(path + 'AllSamples25Im_pairwise_joint_prob00' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_joint_prob00)
        np.save(path + 'AllSamples25Im_pairwise_joint_prob01' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_joint_prob01)
        np.save(path + 'AllSamples25Im_pairwise_joint_prob10' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_joint_prob10)
        np.save(path + 'AllSamples25Im_pairwise_joint_prob11' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_joint_prob11)
      
        np.save(path + 'AllSamples25Im_top_pairwise_diff' + '_' + str(neurons) + '_' + model + '.npy',SubSampling_pairwise_diff)        
        np.save(path + 'AllSamples25Im_top_pairwise_logdiff' + '_' + str(neurons) + '_' + model + '.npy',SubSampling_pairwise_logdiff)        
            
            
            
#%%
color_set = ['green','blue','red']            
for nn in range(len(neurons_set)):
    for m in range(len(model_set)):
        plt.figure()
        neurons = neurons_set[nn]            
        model = model_set[m]
        SubSampling_pairwise_diff = np.load(path + 'AllSamples25Im_top_pairwise_diff' + '_' + str(neurons) + '_' + model + '.npy')                
        for i in range(num_im):
            plt.subplot(5,5,i+1)
            plt.hist(SubSampling_pairwise_diff[i,:],color = color_set[m])
            plt.yticks([])
        
            
#%%            
color_set = ['green','blue','red']            
for nn in range(len(neurons_set)):
    for m in range(len(model_set)):
        plt.figure()
        neurons = neurons_set[nn]            
        model = model_set[m]
        Sampling_pairwise_diff = np.load(path + 'AllSamples25Im_pairwise_diff' + '_' + str(neurons) + '_' + model + '.npy')                
        for i in range(num_im):
            plt.subplot(5,5,i+1)
            plt.hist(Sampling_pairwise_diff[i,:],color = color_set[m])
        
            
            
            
##2,7,8,9,12,20          
            
            
            
            
            