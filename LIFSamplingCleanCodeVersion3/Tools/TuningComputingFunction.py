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

from Tools.load_params import *

from Sampling_Functions.Marg_samples import *
from Sampling_Functions.Inference_Gibbs import * 

from LIF_Functions.LIF_spikes_from_BRIAN import *
from LIF_Functions.LIF_with_BRIAN import LIFSamplingModel

def compute_tuning(inputs):
    print(inputs)
    neurons, model, num_repeats, contrast, sf_case, sim_type, condition_duration, condition_gray = inputs
    params = load_simulation_parameters(neurons,model)
    angles = np.load('../Data/Grating_angles.npy')
    phases = np.load('../Data/Grating_phases.npy')
    PreprocessedGratings_all = np.load('../Data/PreprocessedGratings-8-zca-norm.npy') 
    Grating_Image_set = np.squeeze(PreprocessedGratings_all[sf_case,:,:,:])
#====================================================================================================================================    
#    ##If needed to test for small values
#    test_indx = np.array([0,5,10,15,20])
#    angles = angles[test_indx]
#    phases = phases[test_indx]
#    temp = np.zeros((5,5,64))
#    for i in range(5):
#        for j in range(5):
#            temp[i,j,:] = PreprocessedGratings_all[sf_case,test_indx[i],test_indx[j],:]
#    Grating_Image_set = temp
#====================================================================================================================================   
    path = '../Results/'
    pix = 64
    #Grating_Image_set = [angles x phases x pixels]
    num_angles = len(angles)#np.shape(Grating_Image_set)[0]
    num_ph = len(phases)#np.shape(Grating_Image_set)[1]
        
    if (sim_type=="LIF"):
        LIF = 1
    else:
        LIF = 0    
## In this simulation each phase of an orientated grating is shown for 25ms, making it ~500 msec per orientation followed by gray image for 100 msecs 
## and repeated for ~20 orientations   
    n_samples = int(condition_duration/params.sampling_bin_ms)
    n_samples_gray = int(condition_gray/params.sampling_bin_ms)
    sample_hertz = 1.0/(params.sampling_bin_s)
    
    if (LIF==0):
        fr_sampling = np.zeros((num_repeats, num_angles, num_ph, params.N))
        samples_all = np.zeros((num_repeats, num_angles, num_ph, params.N, n_samples))
        Gray_Image = 0.1*np.ones(pix)
    
        c = 1
        for rep in range(num_repeats):
            random_init = 1
            init_sample = np.zeros(params.N)
            
            for i in range(num_angles):
                gray_samples,_,_,_ = Inference_Gibbs(params, Gray_Image, n_samples_gray, random_init, init_sample)
#                print ('Gray Sampling Done')
                
                random_init = 0
                init_sample = gray_samples[:,-1]
                  
                for j in range(num_ph):
                    if np.mod(c,100)==1:
                        print(str(c) + "/" + str((num_repeats*num_angles*num_ph)) + " for contrast " + str(contrast) + " for Sampling of " + str(params.N) + " neurons and model " + str(model)) 
                    Grating_Image = contrast * Grating_Image_set[i,j,:]
                    samples, ff, rec, prob = Inference_Gibbs(params, Grating_Image, n_samples, random_init, init_sample)
#                    print ('Sampling Done')
                    
                    fr_sampling[rep,i,j,:] = marg_prob_compute(samples,params.N) * sample_hertz
                    samples_all[rep,i,j,:,:] = samples
                    c = c + 1
                    
                    random_init = 0
                    init_sample = samples[:,-1]
        np.save(path + sim_type + "Tuning_" + "Con=" + str(contrast) + '_' + str(neurons) +  '_' + model + '_sf' + str(sf_case+1) + '.npy',fr_sampling)
        np.save(path + sim_type + "TuningFR_" + "Con=" + str(contrast) + '_' + str(neurons) +  '_' + model + '_sf' + str(sf_case+1) + '.npy',samples_all)
           
    else:
        
        fr_LIF = np.zeros((num_repeats, num_angles, num_ph, params.N))
        spikes_all = np.zeros((num_repeats, num_angles, num_ph, params.N, n_samples))   
        Gray_Image = 0.1*np.ones(pix)
    
        c = 1
        for rep in range(num_repeats):
            random_init = 1
            init_membrane_potential = -75 * np.ones(params.N)
            for i in range(num_angles):
                gray_M = LIFSamplingModel(params.N, params.G, init_membrane_potential, membrane_noise=0.0, prior=params.prior, pixel_noise=params.sigma_I , psp_length_ms=params.sampling_bin_ms, verbose=True, trial_duration_ms=condition_duration, photo_noise=params.photo_noise , photo_noise_refresh_ms=params.photo_noise_refresh_ms, rec_noise=params.rec_noise,random_init_process=random_init)
                gray_M.condition(Gray_Image)
                gray_spikes = gray_M.simulate(monitor=["v", "P", "I", "psp", "FR"])
#                print ('Gray LIF Done')
#                gray_spike_array_binned = Extract_Spikes(params.N, condition_gray, params.sampling_bin_ms, gray_spikes) 
#                print ('Gray Spikes Done')
            
                random_init = 0
                init_membrane_potential = gray_M.monitor.v[:,-1]
                    
                for j in range(num_ph):
                    if np.mod(c,100)==1:
                        print(str(c) + "/" + str((num_repeats*num_angles*num_ph)) + " for contrast " + str(contrast) + " for Sampling of " + str(params.N) + " neurons and model " + str(model)) 
                    Grating_Image = contrast * Grating_Image_set[i,j,:]
                    M = LIFSamplingModel(params.N, params.G, init_membrane_potential, membrane_noise=0.0, prior=params.prior, pixel_noise=params.sigma_I , psp_length_ms=params.sampling_bin_ms, verbose=True, trial_duration_ms=condition_duration, photo_noise=params.photo_noise , photo_noise_refresh_ms=params.photo_noise_refresh_ms, rec_noise=params.rec_noise,random_init_process=random_init)
                    M.condition(Grating_Image)
                    spikes = M.simulate(monitor=["v", "P", "I", "psp", "FR"])
#                    print ('LIF Done')
                    spike_array_binned = Extract_Spikes(params.N, condition_duration, params.sampling_bin_ms, spikes) 
#                    print ('Spikes Done')
                    
                    fr_LIF[rep,i,j,:] = np.sum(spike_array_binned,1)/spike_array_binned.shape[1] * sample_hertz
                    spikes_all[rep,i,j,:,:] = spike_array_binned
                    c = c + 1
                    
                    random_init = 0
                    init_membrane_potential = M.monitor.v[:,-1]
        np.save(path + sim_type + "Tuning_" + "Con=" + str(contrast) + '_' + str(neurons) +  '_' + model + '_sf' + str(sf_case+1) + '.npy',fr_LIF)
        np.save(path + sim_type + "TuningFR_" + "Con=" + str(contrast) + '_' + str(neurons) +  '_' + model + '_sf' + str(sf_case+1) + '.npy',spikes_all)
            
    
