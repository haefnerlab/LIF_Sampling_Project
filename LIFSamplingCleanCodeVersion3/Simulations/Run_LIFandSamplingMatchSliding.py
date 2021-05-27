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
model_set = ['NN', 'IN','ISN']
neurons_set = [128]#, 64]
contrasts = np.array([0.25, 0.5, 0.75, 1.0])
angles = np.load('../Data/Grating_angles.npy')
pix = 64 #image pixels
dimension = 8 #sqrt(image pixels)
simulation_cases =  ['NaturalImage']
top_n = 5

for sm in range(len(simulation_cases)):   
    for nn in range(len(neurons_set)):
        data = np.load('../Data/SuitableNaturalImages_'+str(neurons_set[nn])+'.npy') 
        for m in range(len(model_set)): 
            if simulation_cases[sm]=='NaturalImage':
                neurons = neurons_set[nn]
                model = model_set[m]
                params = load_simulation_parameters(neurons,model)
                sample_hertz = 1.0/(params.sampling_bin_s) 
                num_im = np.shape(data)[0]
                      
                comb = combinations(range(params.N), 2)
                comb = np.array(list(comb)) 
                prob_pair_length = (comb.shape[0]*4)
        
                Sampling_marginal_prob = np.zeros((num_im,params.N))
                Sampling_pairwise_joint_prob = np.zeros((num_im,prob_pair_length))
                Sampling_pairwise_diff = np.zeros((num_im,prob_pair_length))
                Sampling_pairwise_diffbounded = -100*np.ones((num_im,prob_pair_length))
                Sampling_pairwise_logdiff = np.zeros((num_im,prob_pair_length))
                Sampling_pairwise_logdiffbounded = np.zeros((num_im,prob_pair_length))        
                Sampling_pairwise_joint_prob00 = np.zeros((num_im,int(prob_pair_length/4)))
                Sampling_pairwise_joint_prob01 = np.zeros((num_im,int(prob_pair_length/4)))
                Sampling_pairwise_joint_prob10 = np.zeros((num_im,int(prob_pair_length/4)))
                Sampling_pairwise_joint_prob11 = np.zeros((num_im,int(prob_pair_length/4)))
                 
                LIF_marginal_prob = np.zeros((num_im,params.N))
                LIF_pairwise_joint_prob = np.zeros((num_im,prob_pair_length))
                LIF_pairwise_diff = np.zeros((num_im,prob_pair_length))
                LIF_pairwise_diffbounded = -100*np.ones((num_im,prob_pair_length))
                LIF_pairwise_logdiff = np.zeros((num_im,prob_pair_length))
                LIF_pairwise_logdiffbounded = np.zeros((num_im,prob_pair_length))
                LIF_pairwise_joint_prob00 = np.zeros((num_im,int(prob_pair_length/4)))
                LIF_pairwise_joint_prob01 = np.zeros((num_im,int(prob_pair_length/4)))
                LIF_pairwise_joint_prob10 = np.zeros((num_im,int(prob_pair_length/4)))
                LIF_pairwise_joint_prob11 = np.zeros((num_im,int(prob_pair_length/4)))
                
                Multi_info = np.zeros(num_im)
                spikes_im = np.zeros((num_im,params.N,100))
                samples_im = np.zeros((num_im,params.N,100))
                for i in range(num_im):
                    print ("Natural Image number: " + str(i+1))
                    NaturalImage_chosen = np.squeeze(data[i,:,:])
                    natural_image_duration = 500
                    n_samples = int(natural_image_duration/params.sampling_bin_ms)  
                    random_init = 1 
                    # not useful if ranom_init = 1 else, sets initial samples and membrane potential to values as in init_sample and init_membrane_potential
                    init_sample = np.zeros(params.N)
                    init_membrane_potential = -70 * np.ones(params.N)
                    # Simulating Gibbs sampling and returning samples, feedforward input, recurrent input and Gibbs probability
                    samplesNat, ff, rec, prob = Inference_Gibbs(params, NaturalImage_chosen, n_samples, random_init, init_sample)
                    samples_im[i,:,:] = samplesNat
                    # Simulating LIF firing in BRIAN and returning spiketimes with corresponding neuron indices. 
                    # Also monitoring voltage, probability, input current, post synaptic potential and firing rate
                    M = LIFSamplingModel(params.N, params.G, init_membrane_potential, membrane_noise=0.0, prior=params.prior, pixel_noise=params.sigma_I , psp_length_ms=params.sampling_bin_ms, verbose=False, trial_duration_ms=natural_image_duration, photo_noise=params.photo_noise , photo_noise_refresh_ms=params.photo_noise_refresh_ms, rec_noise=params.rec_noise,random_init_process=random_init)
                    M.condition(NaturalImage_chosen)
                    spikesNat = M.simulate(monitor=["v", "P", "I", "psp", "FR", "is_active"])
                    # Obtaining binned spike train from BRIAN simulations
                    spike_array_binnedNat = Extract_Spikes_Sliding(params.N, natural_image_duration, params.sampling_bin_ms, spikesNat) 
                    spikes_im[i,:,:] = Extract_Spikes(params.N, natural_image_duration, params.sampling_bin_ms, spikesNat) 
                                
                    Sampling_marginal_prob[i,:] = marg_prob_compute(samplesNat,params.N)
                    LIF_marginal_prob[i,:] = marg_prob_compute(spike_array_binnedNat,params.N) 
                    
                    H_indx = np.flip(np.argsort(np.squeeze(Sampling_marginal_prob[i,:])))
                    sampling_p_marg = np.squeeze(Sampling_marginal_prob[i,H_indx[:top_n]])
                    
                    Sampling_pairwise_joint_prob[i,:],Sampling_pairwise_joint_prob00[i,:],Sampling_pairwise_joint_prob01[i,:],Sampling_pairwise_joint_prob10[i,:],Sampling_pairwise_joint_prob11[i,:] = pairwise_prob_compute(samplesNat,params.N,1) 
                    LIF_pairwise_joint_prob[i,:],LIF_pairwise_joint_prob00[i,:],LIF_pairwise_joint_prob01[i,:],LIF_pairwise_joint_prob10[i,:],LIF_pairwise_joint_prob11[i,:] = pairwise_prob_compute(spike_array_binnedNat,params.N,1) 
                    
                    jt,tmp1,tmp2,tmp3,tmp4 = pairwise_prob_compute(np.squeeze(samplesNat[H_indx[:top_n],:]),top_n,1) 
                    Multi_info[i] = compute_multi_info(sampling_p_marg,jt)
                    
                    Sampling_pairwise_diff[i,:],Sampling_pairwise_diffbounded[i,:],Sampling_pairwise_logdiff[i,:],Sampling_pairwise_logdiffbounded[i,:] = pairwise_prob_difference_compute(samplesNat,params.N)
                    LIF_pairwise_diff[i,:],LIF_pairwise_diffbounded[i,:],LIF_pairwise_logdiff[i,:],LIF_pairwise_logdiffbounded[i,:] = pairwise_prob_difference_compute(spike_array_binnedNat,params.N)
                    
                    
                np.save(path + 'SlidingSampling_marginal_probNaturalImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_marginal_prob)
                np.save(path + 'SlidingSampling_pairwise_joint_probNaturalImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_joint_prob)
                np.save(path + 'SlidingSampling_pairwise_joint_diffprobNaturalImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_diff)        
                np.save(path + 'SlidingSampling_pairwise_joint_diffprobboundedNaturalImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_diffbounded)                
                np.save(path + 'SlidingSampling_pairwise_joint_logdiffprobNaturalImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_logdiff)        
                np.save(path + 'SlidingSampling_pairwise_joint_logdiffprobboundedNaturalImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_logdiffbounded)                
                np.save(path + 'SlidingSampling_pairwise_joint_prob00NaturalImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_joint_prob00)
                np.save(path + 'SlidingSampling_pairwise_joint_prob01NaturalImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_joint_prob01)
                np.save(path + 'SlidingSampling_pairwise_joint_prob10NaturalImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_joint_prob10)
                np.save(path + 'SlidingSampling_pairwise_joint_prob11NaturalImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_joint_prob11)
            
                np.save(path + 'SlidingLIF_marginal_probNaturalImages' + '_'  + str(neurons) + '_' + model + '.npy',LIF_marginal_prob)
                np.save(path + 'SlidingLIF_pairwise_joint_probNaturalImages' '_'  + str(neurons) + '_' + model + '.npy',LIF_pairwise_joint_prob)            
                np.save(path + 'SlidingLIF_pairwise_joint_diffprobNaturalImages' '_'  + str(neurons) + '_' + model + '.npy',LIF_pairwise_diff)                    
                np.save(path + 'SlidingLIF_pairwise_joint_diffprobboundedNaturalImages' + '_' + str(neurons) + '_' + model + '.npy',LIF_pairwise_diffbounded)                        
                np.save(path + 'SlidingLIF_pairwise_joint_logdiffprobNaturalImages' '_'  + str(neurons) + '_' + model + '.npy',LIF_pairwise_logdiff)                    
                np.save(path + 'SlidingLIF_pairwise_joint_logdiffprobboundedNaturalImages' '_'  + str(neurons) + '_' + model + '.npy',LIF_pairwise_logdiffbounded)                    
                np.save(path + 'SlidingLIF_pairwise_joint_prob00NaturalImages' '_'  + str(neurons) + '_' + model + '.npy',LIF_pairwise_joint_prob00)            
                np.save(path + 'SlidingLIF_pairwise_joint_prob01NaturalImages' '_'  + str(neurons) + '_' + model + '.npy',LIF_pairwise_joint_prob01)            
                np.save(path + 'SlidingLIF_pairwise_joint_prob10NaturalImages' '_'  + str(neurons) + '_' + model + '.npy',LIF_pairwise_joint_prob10)            
                np.save(path + 'SlidingLIF_pairwise_joint_prob11NaturalImages' '_'  + str(neurons) + '_' + model + '.npy',LIF_pairwise_joint_prob11)
                
                np.save(path + 'SlidingMulti_info_NaturalImages' + '_' + str(neurons) + '_' + model + '.npy',Multi_info)
                np.save(path + 'Spikes_NaturalImages' + '_' + str(neurons) + '_' + model + '.npy',spikes_im)
                np.save(path + 'Samples_NaturalImages' + '_' + str(neurons) + '_' + model + '.npy',samples_im)
    
            elif simulation_cases[sm]=='Grating':
                GratingImage_data = np.load('../Data/PreprocessedGratings-8-zca-norm.npy') # load grating image preprocessed 
                angles = np.load('../Data/Grating_angles.npy')
                sf_chosen_indx = 0
                contrasts = np.load('../Data/Contrasts.npy')
                Grating_Image_all = np.squeeze(GratingImage_data[sf_chosen_indx,:,:,:])
                num_ph = np.shape(Grating_Image_all)[0]
                grating_image_duration = 25
                
                num_im = len(contrasts) * len(angles)
                neurons = neurons_set[nn]
                model = model_set[m]
                params = load_simulation_parameters(neurons,model)
                sample_hertz = 1.0/(params.sampling_bin_s)
                
                comb = combinations(range(params.N), 2)
                comb = np.array(list(comb)) 
                prob_pair_length = (comb.shape[0]*4)
        
                Sampling_marginal_prob = np.zeros((num_im,params.N))
                Sampling_pairwise_joint_prob = np.zeros((num_im,prob_pair_length))
                Sampling_pairwise_diff = np.zeros((num_im,prob_pair_length))
                Sampling_pairwise_diffbounded = -100*np.ones((num_im,prob_pair_length))
                Sampling_pairwise_logdiff = np.zeros((num_im,prob_pair_length))
                Sampling_pairwise_logdiffbounded = np.zeros((num_im,prob_pair_length))        
                Sampling_pairwise_joint_prob00 = np.zeros((num_im,int(prob_pair_length/4)))
                Sampling_pairwise_joint_prob01 = np.zeros((num_im,int(prob_pair_length/4)))
                Sampling_pairwise_joint_prob10 = np.zeros((num_im,int(prob_pair_length/4)))
                Sampling_pairwise_joint_prob11 = np.zeros((num_im,int(prob_pair_length/4)))
                 
                LIF_marginal_prob = np.zeros((num_im,params.N))
                LIF_pairwise_joint_prob = np.zeros((num_im,prob_pair_length))
                LIF_pairwise_diff = np.zeros((num_im,prob_pair_length))
                LIF_pairwise_diffbounded = -100*np.ones((num_im,prob_pair_length))
                LIF_pairwise_logdiff = np.zeros((num_im,prob_pair_length))
                LIF_pairwise_logdiffbounded = np.zeros((num_im,prob_pair_length))
                LIF_pairwise_joint_prob00 = np.zeros((num_im,int(prob_pair_length/4)))
                LIF_pairwise_joint_prob01 = np.zeros((num_im,int(prob_pair_length/4)))
                LIF_pairwise_joint_prob10 = np.zeros((num_im,int(prob_pair_length/4)))
                LIF_pairwise_joint_prob11 = np.zeros((num_im,int(prob_pair_length/4)))
                
                Multi_info = np.zeros(num_im)
                
                i = 0
                n_slides = 21
                for ccon in range(len(contrasts)):
                    for ang in range(len(angles)):
                        print ("Grating Image number: " +str(i+1))
                        random_init = 1
                        init_sample = np.zeros(params.N)
                        init_membrane_potential = -75 * np.ones(params.N)
                        n_samples = int(grating_image_duration/params.sampling_bin_ms)
                        samples_all = np.zeros((params.N, num_ph*n_samples))
                        spike_array_binned = np.zeros((params.N, num_ph*n_slides))
                        for j in range(num_ph):
                            Grating_Image = contrasts[ccon] * np.squeeze(Grating_Image_all[ang,j,:])
                            samples, ff, rec, prob = Inference_Gibbs(params, Grating_Image, n_samples, random_init, init_sample)
                            M = LIFSamplingModel(params.N, params.G, init_membrane_potential, membrane_noise=0.0, prior=params.prior, pixel_noise=params.sigma_I , psp_length_ms=params.sampling_bin_ms, verbose=False, trial_duration_ms=grating_image_duration, photo_noise=params.photo_noise , photo_noise_refresh_ms=params.photo_noise_refresh_ms, rec_noise=params.rec_noise,random_init_process=random_init)
                            M.condition(Grating_Image)
                            spikes = M.simulate(monitor=["v", "P", "I", "psp", "FR"])
                            spike_array_binned[:,j*n_slides:((j+1)*n_slides)] =  Extract_Spikes_Sliding(params.N, grating_image_duration, params.sampling_bin_ms, spikes) 
                            samples_all[:,j*n_samples:((j+1)*n_samples)] = samples
                            random_init = 0
                            init_sample = samples[:,-1]
                            init_membrane_potential = M.monitor.v[:,-1]
                
                        
                        Sampling_marginal_prob[i,:] = marg_prob_compute(samples_all,params.N)
                        LIF_marginal_prob[i,:] = marg_prob_compute(spike_array_binned,params.N) 
                        
                        H_indx = np.flip(np.argsort(np.squeeze(Sampling_marginal_prob[i,:])))
                        sampling_p_marg = np.squeeze(Sampling_marginal_prob[i,H_indx[:top_n]])
                        
                        Sampling_pairwise_joint_prob[i,:],Sampling_pairwise_joint_prob00[i,:],Sampling_pairwise_joint_prob01[i,:],Sampling_pairwise_joint_prob10[i,:],Sampling_pairwise_joint_prob11[i,:] = pairwise_prob_compute(samples_all,params.N,1) 
                        LIF_pairwise_joint_prob[i,:],LIF_pairwise_joint_prob00[i,:],LIF_pairwise_joint_prob01[i,:],LIF_pairwise_joint_prob10[i,:],LIF_pairwise_joint_prob11[i,:] = pairwise_prob_compute(spike_array_binned,params.N,1) 
                        
                        jt,tmp1,tmp2,tmp3,tmp4 = pairwise_prob_compute(np.squeeze(samplesNat[H_indx[:top_n],:]),top_n,1) 
                        Multi_info[i] = compute_multi_info(sampling_p_marg,jt)
                        
                        Sampling_pairwise_diff[i,:],Sampling_pairwise_diffbounded[i,:],Sampling_pairwise_logdiff[i,:],Sampling_pairwise_logdiffbounded[i,:] = pairwise_prob_difference_compute(samples_all,params.N)
                        LIF_pairwise_diff[i,:],LIF_pairwise_diffbounded[i,:],LIF_pairwise_logdiff[i,:],LIF_pairwise_logdiffbounded[i,:] = pairwise_prob_difference_compute(spike_array_binned,params.N)
                    
                        i = i + 1
           
                np.save(path + 'SlidingSampling_marginal_probGratingImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_marginal_prob)
                np.save(path + 'SlidingSampling_pairwise_joint_probGratingImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_joint_prob)
                np.save(path + 'SlidingSampling_pairwise_joint_diffprobGratingImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_diff)        
                np.save(path + 'SlidingSampling_pairwise_joint_diffprobboundedGratingImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_diffbounded)                
                np.save(path + 'SlidingSampling_pairwise_joint_logdiffprobGratingImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_logdiff)        
                np.save(path + 'SlidingSampling_pairwise_joint_logdiffprobboundedGratingImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_logdiffbounded)                
                np.save(path + 'SlidingSampling_pairwise_joint_prob00GratingImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_joint_prob00)
                np.save(path + 'SlidingSampling_pairwise_joint_prob01GratingImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_joint_prob01)
                np.save(path + 'SlidingSampling_pairwise_joint_prob10GratingImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_joint_prob10)
                np.save(path + 'SlidingSampling_pairwise_joint_prob11GratingImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_joint_prob11)
            
                np.save(path + 'SlidingLIF_marginal_probGratingImages' + '_'  + str(neurons) + '_' + model + '.npy',LIF_marginal_prob)
                np.save(path + 'SlidingLIF_pairwise_joint_probGratingImages' '_'  + str(neurons) + '_' + model + '.npy',LIF_pairwise_joint_prob)            
                np.save(path + 'SlidingLIF_pairwise_joint_diffprobGratingImages' '_'  + str(neurons) + '_' + model + '.npy',LIF_pairwise_diff)                    
                np.save(path + 'SlidingLIF_pairwise_joint_diffprobboundedGratingImages' + '_' + str(neurons) + '_' + model + '.npy',LIF_pairwise_diffbounded)                        
                np.save(path + 'SlidingLIF_pairwise_joint_logdiffprobGratingImages' '_'  + str(neurons) + '_' + model + '.npy',LIF_pairwise_logdiff)                    
                np.save(path + 'SlidingLIF_pairwise_joint_logdiffprobboundedGratingImages' '_'  + str(neurons) + '_' + model + '.npy',LIF_pairwise_logdiffbounded)                    
                np.save(path + 'SlidingLIF_pairwise_joint_prob00GratingImages' '_'  + str(neurons) + '_' + model + '.npy',LIF_pairwise_joint_prob00)            
                np.save(path + 'SlidingLIF_pairwise_joint_prob01GratingImages' '_'  + str(neurons) + '_' + model + '.npy',LIF_pairwise_joint_prob01)            
                np.save(path + 'SlidingLIF_pairwise_joint_prob10GratingImages' '_'  + str(neurons) + '_' + model + '.npy',LIF_pairwise_joint_prob10)            
                np.save(path + 'SlidingLIF_pairwise_joint_prob11GratingImages' '_'  + str(neurons) + '_' + model + '.npy',LIF_pairwise_joint_prob11) 
                
                np.save(path + 'SlidingMulti_info_GratingImages' + '_' + str(neurons) + '_' + model + '.npy',Multi_info)