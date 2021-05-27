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

for nn in range(len(neurons_set)):
    for m in range(len(model_set)):  
        neurons = neurons_set[nn]
        model = model_set[m]
        params = load_simulation_parameters(neurons,model)
        sample_hertz = 1.0/(params.sampling_bin_s)
        
        dimension = 8
        
        temp_data = np.load('../Data/SuitableNaturalImages_'+str(neurons_set[nn])+'.npy')  
#        temp_data = load_natural_image_patches(8)
        chosen = np.array([2])#,7,8,9,12,20])#np.array([1,5,8,11,14])#np.random.randint(0,np.shape(temp_data)[0],num_im)  1,5,8,11
        print('Chosen indices:'+str(chosen))
        num_im = len(chosen)
        data = temp_data[chosen,:,:]
        params.duration = 1000
        
        n_samples = int(params.duration/params.sampling_bin_ms)
        rec_noise_sample = params.rec_noise
        photo_noise_sample = params.photo_noise
        
        comb = combinations(range(params.N), 2)
        comb = np.array(list(comb)) 
        prob_pair_length = (comb.shape[0]*4)
        
        print('Photo noise:'+str(photo_noise_sample))
        print('SigmaI:'+str(params.sigma_I))

        Sampling_marginal_prob = np.zeros((num_im,params.N))
        Sampling_pairwise_joint_prob = np.zeros((num_im,prob_pair_length))
        Sampling_pairwise_diff = np.zeros((num_im,prob_pair_length))
        Sampling_pairwise_diffbounded = -100*np.ones((num_im,prob_pair_length))
        Sampling_pairwise_logdiff = np.zeros((num_im,prob_pair_length))
        Sampling_pairwise_joint_prob00 = np.zeros((num_im,int(prob_pair_length/4)))
        Sampling_pairwise_joint_prob01 = np.zeros((num_im,int(prob_pair_length/4)))
        Sampling_pairwise_joint_prob10 = np.zeros((num_im,int(prob_pair_length/4)))
        Sampling_pairwise_joint_prob11 = np.zeros((num_im,int(prob_pair_length/4)))
        
#        Sampling_diff_prob00 = np.zeros((num_im,int(prob_pair_length/4)))
#        Sampling_diff_prob01 = np.zeros((num_im,int(prob_pair_length/4)))
#        Sampling_diff_prob10 = np.zeros((num_im,int(prob_pair_length/4)))
#        Sampling_diff_prob11 = np.zeros((num_im,int(prob_pair_length/4)))
#        
#        Sampling_logdiff_prob00 = np.zeros((num_im,int(prob_pair_length/4)))
#        Sampling_logdiff_prob01 = np.zeros((num_im,int(prob_pair_length/4)))
#        Sampling_logdiff_prob10 = np.zeros((num_im,int(prob_pair_length/4)))
#        Sampling_logdiff_prob11 = np.zeros((num_im,int(prob_pair_length/4)))
        
        
        LIF_marginal_prob = np.zeros((num_im,params.N))
        LIF_pairwise_joint_prob = np.zeros((num_im,prob_pair_length))
        LIF_pairwise_diff = np.zeros((num_im,prob_pair_length))
        LIF_pairwise_diffbounded = -100*np.ones((num_im,prob_pair_length))
        LIF_pairwise_logdiff = np.zeros((num_im,prob_pair_length))
        LIF_pairwise_joint_prob00 = np.zeros((num_im,int(prob_pair_length/4)))
        LIF_pairwise_joint_prob01 = np.zeros((num_im,int(prob_pair_length/4)))
        LIF_pairwise_joint_prob10 = np.zeros((num_im,int(prob_pair_length/4)))
        LIF_pairwise_joint_prob11 = np.zeros((num_im,int(prob_pair_length/4)))
        
        for i in range(num_im):
            print i
            random_init = 1
            init_sample = np.zeros(params.N)
            init_membrane_potential = -75 * np.ones(params.N)
            Image = np.reshape(data[i,:,:],(64))# + sigma_I**2 * np.random.normal(0,1,pix)
        
            samples,_,_,_ = Inference_Gibbs(params.G, params.sigma_I, photo_noise_sample, params.prior, params.N, Image, n_samples, rec_noise_sample, random_init, init_sample)
            print ('Sampling Done')
                        
            M = LIFSamplingModel(params.N, params.G, init_membrane_potential, membrane_noise=0.0, prior=params.prior, pixel_noise=params.sigma_I , psp_length_ms=params.sampling_bin_ms, verbose=True, trial_duration_ms=params.duration, photo_noise=params.photo_noise , photo_noise_refresh_ms=params.photo_noise_refresh_ms, rec_noise=params.rec_noise, random_init_process=random_init)
            M.condition(Image)
            spikes = M.simulate(monitor=["v", "P", "I", "psp", "FR"])
            print ('LIF Done')
            
            times = np.array(spikes.t/ms)
            indices = np.array(spikes.i)
            spike_array_binned = Extract_Spikes(params.N, params.duration, params.sampling_bin_ms, spikes)  
            print ('Spikes Done')
                        
            Sampling_marginal_prob[i,:] = marg_prob_compute(samples,params.N) 
            print Sampling_marginal_prob[i,:]
            LIF_marginal_prob[i,:] = marg_prob_compute(spike_array_binned,params.N) 
            print LIF_marginal_prob[i,:]
            
            Sampling_pairwise_joint_prob[i,:],Sampling_pairwise_joint_prob00[i,:],Sampling_pairwise_joint_prob01[i,:],Sampling_pairwise_joint_prob10[i,:],Sampling_pairwise_joint_prob11[i,:] = pairwise_prob_compute(samples,params.N,1) 
            LIF_pairwise_joint_prob[i,:],LIF_pairwise_joint_prob00[i,:],LIF_pairwise_joint_prob01[i,:],LIF_pairwise_joint_prob10[i,:],LIF_pairwise_joint_prob11[i,:] = pairwise_prob_compute(spike_array_binned,params.N,1) 
            
            Sampling_pairwise_diff[i,:],Sampling_pairwise_diffbounded[i,:],Sampling_pairwise_logdiff[i,:] = pairwise_prob_difference_compute(samples,params.N)
            LIF_pairwise_diff[i,:],LIF_pairwise_diffbounded[i,:],LIF_pairwise_logdiff[i,:] = pairwise_prob_difference_compute(spike_array_binned,params.N)
            
            
        np.save(path + 'Sampling_marginal_probNaturalImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_marginal_prob)
        np.save(path + 'Sampling_pairwise_joint_probNaturalImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_joint_prob)
        np.save(path + 'Sampling_pairwise_joint_diffprobNaturalImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_diff)        
        np.save(path + 'Sampling_pairwise_joint_diffprobboundedNaturalImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_diffbounded)                
        np.save(path + 'Sampling_pairwise_joint_logdiffprobNaturalImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_logdiff)        
        np.save(path + 'Sampling_pairwise_joint_prob00NaturalImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_joint_prob00)
        np.save(path + 'Sampling_pairwise_joint_prob01NaturalImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_joint_prob01)
        np.save(path + 'Sampling_pairwise_joint_prob10NaturalImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_joint_prob10)
        np.save(path + 'Sampling_pairwise_joint_prob11NaturalImages' + '_' + str(neurons) + '_' + model + '.npy',Sampling_pairwise_joint_prob11)
    
        np.save(path + 'LIF_marginal_probNaturalImages' + '_'  + str(neurons) + '_' + model + '.npy',LIF_marginal_prob)
        np.save(path + 'LIF_pairwise_joint_probNaturalImages' '_'  + str(neurons) + '_' + model + '.npy',LIF_pairwise_joint_prob)            
        np.save(path + 'LIF_pairwise_joint_diffprobNaturalImages' '_'  + str(neurons) + '_' + model + '.npy',LIF_pairwise_diff)                    
        np.save(path + 'LIF_pairwise_joint_diffprobboundedNaturalImages' + '_' + str(neurons) + '_' + model + '.npy',LIF_pairwise_diffbounded)                        
        np.save(path + 'LIF_pairwise_joint_logdiffprobNaturalImages' '_'  + str(neurons) + '_' + model + '.npy',LIF_pairwise_logdiff)                    
        np.save(path + 'LIF_pairwise_joint_prob00NaturalImages' '_'  + str(neurons) + '_' + model + '.npy',LIF_pairwise_joint_prob00)            
        np.save(path + 'LIF_pairwise_joint_prob01NaturalImages' '_'  + str(neurons) + '_' + model + '.npy',LIF_pairwise_joint_prob01)            
        np.save(path + 'LIF_pairwise_joint_prob10NaturalImages' '_'  + str(neurons) + '_' + model + '.npy',LIF_pairwise_joint_prob10)            
        np.save(path + 'LIF_pairwise_joint_prob11NaturalImages' '_'  + str(neurons) + '_' + model + '.npy',LIF_pairwise_joint_prob11)            

        
        comb = combinations(range(params.N), 2)
        comb = np.array(list(comb)) 
        prob_pair_length = (comb.shape[0]*4)

        Sampling_marginal_prob = np.zeros((num_im,params.N))
        Sampling_pairwise_joint_prob = np.zeros((num_im,prob_pair_length))
        Sampling_pairwise_joint_prob00 = np.zeros((num_im,int(prob_pair_length/4)))
        Sampling_pairwise_joint_prob01 = np.zeros((num_im,int(prob_pair_length/4)))
        Sampling_pairwise_joint_prob10 = np.zeros((num_im,int(prob_pair_length/4)))
        Sampling_pairwise_joint_prob11 = np.zeros((num_im,int(prob_pair_length/4)))
        LIF_marginal_prob = np.zeros((num_im,params.N))
        LIF_pairwise_joint_prob = np.zeros((num_im,prob_pair_length))
        LIF_pairwise_joint_prob00 = np.zeros((num_im,int(prob_pair_length/4)))
        LIF_pairwise_joint_prob01 = np.zeros((num_im,int(prob_pair_length/4)))
        LIF_pairwise_joint_prob10 = np.zeros((num_im,int(prob_pair_length/4)))
        LIF_pairwise_joint_prob11 = np.zeros((num_im,int(prob_pair_length/4)))
        
#        for i in range(num_im):
#            chosen_angle = chosen_angle_set[i]
#            file_samples = 'AllSamples' + str(sf_case) + '_' + str(neurons_set[nn]) + '_' + str(contrast) + '_' + model_set[m] + '.npy'
#            file_LIF = 'AllSpikes' + str(sf_case) + '_' + str(neurons_set[nn]) + '_' + str(contrast) + '_' + model_set[m] + '.npy'
#            
#            samples = np.load(path + file_samples)
#            spike_array_binned = np.load(path + file_LIF)
#            
#            num_phases = np.shape(samples)[2]
#            time_steps = np.shape(samples)[4]
#            samples_all_phases = np.zeros((neurons_set[nn],num_phases*time_steps))
#            for ii in range(neurons_set[nn]):
#                samples_all_phases[ii,:] = np.reshape(np.squeeze(samples[repeat_num, chosen_angle,:,ii,:]),(num_phases*time_steps))
#            
#            num_phases = np.shape(spike_array_binned)[2]
#            time_steps = np.shape(spike_array_binned)[4]
#            spikes_all_phases = np.zeros((neurons_set[nn],num_phases*time_steps))
#            for ii in range(neurons_set[nn]):
#                spikes_all_phases[ii,:] = np.reshape(np.squeeze(spike_array_binned[repeat_num, chosen_angle,:,ii,:]),(num_phases*time_steps))
#        
#            Sampling_marginal_prob[i,:] = marg_prob_compute(samples_all_phases,params.N) 
#            LIF_marginal_prob[i,:] = marg_prob_compute(spikes_all_phases,params.N) 
#            
#            Sampling_pairwise_joint_prob[i,:],Sampling_pairwise_joint_prob00[i,:],Sampling_pairwise_joint_prob01[i,:],Sampling_pairwise_joint_prob10[i,:],Sampling_pairwise_joint_prob11[i,:] = pairwise_prob_compute(samples_all_phases,params.N,1) 
#            LIF_pairwise_joint_prob[i,:],LIF_pairwise_joint_prob00[i,:],LIF_pairwise_joint_prob01[i,:],LIF_pairwise_joint_prob10[i,:],LIF_pairwise_joint_prob11[i,:] = pairwise_prob_compute(spikes_all_phases,params.N,1) 
#            
#        np.save(path + 'Sampling_marginal_probGratingImages' + str(sf_case) + '_' + str(contrast) + str(neurons) + '_' + model + '.npy',Sampling_marginal_prob)
#        np.save(path + 'Sampling_pairwise_joint_probGratingImages' + str(sf_case) + '_' + str(contrast) + str(neurons) + '_' + model + '.npy',Sampling_pairwise_joint_prob)
#        np.save(path + 'Sampling_pairwise_joint_prob00GratingImages' + str(sf_case) + '_' + str(contrast) + str(neurons) + '_' + model + '.npy',Sampling_pairwise_joint_prob00)
#        np.save(path + 'Sampling_pairwise_joint_prob01GratingImages' + str(sf_case) + '_' + str(contrast) + str(neurons) + '_' + model + '.npy',Sampling_pairwise_joint_prob01)
#        np.save(path + 'Sampling_pairwise_joint_prob10GratingImages' + str(sf_case) + '_' + str(contrast) + str(neurons) + '_' + model + '.npy',Sampling_pairwise_joint_prob10)
#        np.save(path + 'Sampling_pairwise_joint_prob11GratingImages' + str(sf_case) + '_' + str(contrast) + str(neurons) + '_' + model + '.npy',Sampling_pairwise_joint_prob11)

#    
#        np.save(path + 'LIF_marginal_probGratingImages' + str(sf_case) + '_' + str(contrast) + str(neurons) + '_' + model + '.npy',LIF_marginal_prob)
#        np.save(path + 'LIF_pairwise_joint_probGratingImages' + str(sf_case) + '_' + str(contrast) + str(neurons) + '_' + model + '.npy',LIF_pairwise_joint_prob)            
#        np.save(path + 'LIF_pairwise_joint_prob00GratingImages' + str(sf_case) + '_' + str(contrast) + str(neurons) + '_' + model + '.npy',LIF_pairwise_joint_prob00)            
#        np.save(path + 'LIF_pairwise_joint_prob01GratingImages' + str(sf_case) + '_' + str(contrast) + str(neurons) + '_' + model + '.npy',LIF_pairwise_joint_prob01)            
#        np.save(path + 'LIF_pairwise_joint_prob10GratingImages' + str(sf_case) + '_' + str(contrast) + str(neurons) + '_' + model + '.npy',LIF_pairwise_joint_prob10)            
#        np.save(path + 'LIF_pairwise_joint_prob11GratingImages' + str(sf_case) + '_' + str(contrast) + str(neurons) + '_' + model + '.npy',LIF_pairwise_joint_prob11)            
        
