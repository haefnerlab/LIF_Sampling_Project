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
model_set = ['ISN']#['NN', 'IN',
neurons_set = [128]#, 64]
contrasts = np.array([0.25, 0.5, 0.75, 1.0])
angles = np.load('../Data/Grating_angles.npy')
pix = 64 #image pixels
dimension = 8 #sqrt(image pixels)
simulation_cases =  ['NaturalImage','Grating']
num_repeats = 50

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
        
                spikes_im = np.zeros((num_repeats,num_im,params.N,100))
                samples_im = np.zeros((num_repeats,num_im,params.N,100))
                kk = 0
                for i in range(num_im):
                    for nr in range(num_repeats):
                        print ("Counting " + str(kk+1) + " out of " + str(num_repeats*num_im))
                        NaturalImage_chosen = np.squeeze(data[i,:,:])
                        natural_image_duration = 500
                        n_samples = int(natural_image_duration/params.sampling_bin_ms)  
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
                        # Obtaining binned spike train from BRIAN simulations
                        kk = kk+1
                        samples_im[nr,i,:,:] = samplesNat
                        spikes_im[nr,i,:,:] = Extract_Spikes(params.N, natural_image_duration, params.sampling_bin_ms, spikesNat) 
                    
                np.save(path + 'SpikesforStatistics_NaturalImages' + '_' + str(neurons) + '_' + model + '.npy',spikes_im)
                np.save(path + 'SamplesforStatistics_NaturalImages' + '_' + str(neurons) + '_' + model + '.npy',samples_im)
    
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
                
                spikes_im = np.zeros((num_repeats,len(contrasts),len(angles),params.N,105))
                samples_im = np.zeros((num_repeats,len(contrasts),len(angles),params.N,105))
                i=0
                for nr in range(num_repeats):
                    for ccon in range(len(contrasts)):
                        for ang in range(len(angles)):
                            print ("Counting: " +str(i+1) + " out of " + str(num_repeats*len(contrasts)*len(angles)))
                            random_init = 1
                            init_sample = np.zeros(params.N)
                            init_membrane_potential = -75 * np.ones(params.N)
                            n_samples = int(grating_image_duration/params.sampling_bin_ms)
                            samples_all = np.zeros((params.N, num_ph*n_samples))
                            spike_array_binned = np.zeros((params.N, num_ph*n_samples))
                            for j in range(num_ph):
                                Grating_Image = contrasts[ccon] * np.squeeze(Grating_Image_all[ang,j,:])
                                samples, ff, rec, prob = Inference_Gibbs(params, Grating_Image, n_samples, random_init, init_sample)
                                M = LIFSamplingModel(params.N, params.G, init_membrane_potential, membrane_noise=0.0, prior=params.prior, pixel_noise=params.sigma_I , psp_length_ms=params.sampling_bin_ms, verbose=False, trial_duration_ms=grating_image_duration, photo_noise=params.photo_noise , photo_noise_refresh_ms=params.photo_noise_refresh_ms, rec_noise=params.rec_noise,random_init_process=random_init)
                                M.condition(Grating_Image)
                                spikes = M.simulate(monitor=["v", "P", "I", "psp", "FR"])
                                spike_array_binned[:,j*n_samples:((j+1)*n_samples)] =  Extract_Spikes(params.N, grating_image_duration, params.sampling_bin_ms, spikes) 
                                samples_all[:,j*n_samples:((j+1)*n_samples)] = samples
                                random_init = 0
                                init_sample = samples[:,-1]
                                init_membrane_potential = M.monitor.v[:,-1]
                            i = i+1
                            samples_im[nr,ccon,ang,:,:] = samples_all
                            spikes_im[nr,ccon,ang,:,:] = spike_array_binned
           
                np.save(path + 'SpikesforStatistics_GratingImages' + '_' + str(neurons) + '_' + model + '.npy',spikes_im)
                np.save(path + 'SamplesforStatistics_GratingImages' + '_' + str(neurons) + '_' + model + '.npy',samples_im)