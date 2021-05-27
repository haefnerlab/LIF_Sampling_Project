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


colors = np.array(['green','blue','red'])
colors1 = np.flip(np.array(['greenyellow','lawngreen','limegreen','forestgreen','green','darkgreen']))
colors2 = np.flip(np.array(['dodgerblue','deepskyblue','cornflowerblue','royalblue','blue','darkblue']))
colors3 = np.flip(np.array(['lightcoral','indianred','firebrick','red','maroon','darkred']))
#colors1 = np.array(['darkgreen','greenyellow'])
#colors2 = np.array(['darkblue','dodgerblue'])
#colors3 = np.array(['darkred','lightcoral'])

## Initialize variables    
path = '../Results/'
model_set = ['ISN']
neurons_set = [128]#, 64]
nat_im_show = np.array([1017, 1732])#np.array([5460, 1732])#np.array([1017, 1732])   4187,1253,1343,1000,
pix = 64 #image pixels
dimension = 8 #sqrt(image pixels)
simulation_cases =  ['NaturalImage']#['Grating']#

#%%
for sm in range(len(simulation_cases)):   
    for nn in range(len(neurons_set)):
        if simulation_cases[sm]=='NaturalImage':
            ##NaturalImage_data = load_natural_image_patches(8) # load natural image reprocessed patches
            NaturalImage_data = np.load('../Data/PreprocessedImagePatches-8-zca-norm.npy') # load natural image reprocessed patches
            chosen_natural_im = nat_im_show[nn]
            print('Chosen Natural Image number:' + str(chosen_natural_im) + ' for ' + str(neurons_set[nn]) + ' neurons')
            NaturalImage_chosen = np.squeeze(NaturalImage_data[chosen_natural_im,:,:])
            np.save(path + 'NaturalImage' + '_' + str(neurons_set[nn])  + '.npy',NaturalImage_chosen)
            natural_image_duration = 500 # duration of spike train simulation
            print('Computing for natural image...')
            print('----------------------------------------------------------------------------------------------------------------------------------------------')
                
            ## Choose colors depending on model
            for m in range(len(model_set)):
                params = load_simulation_parameters(neurons_set[nn],model_set[m])
                n_samples = int(natural_image_duration/params.sampling_bin_ms)
         
                # Determines if initial samples in Sampling and membrane potentials in LIF simulations are assigned randomly
                random_init = 1 
                # not useful if ranom_init = 1 else, sets initial samples and membrane potential to values as in init_sample and init_membrane_potential
                init_sample = np.zeros(params.N)
                init_membrane_potential = -70 * np.ones(params.N)
                # Simulating Gibbs sampling and returning samples, feedforward input, recurrent input and Gibbs probability
                samplesNat, ff, rec, prob = Inference_Gibbs(params, NaturalImage_chosen, n_samples, random_init, init_sample)
                np.save(path + 'NaturalImageSamplesFF' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',ff)
                np.save(path + 'NaturalImageSamplesRec' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',rec)
                np.save(path + 'NaturalImageSamplesGibbsProb' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',prob)
                print ('Sampling done for natural image')
                
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
                
                np.save(path + 'NaturalImageSamples' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',samplesNat)
                np.save(path + 'NaturalImageLIF_SpikeIndices' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',indices)
                np.save(path + 'NaturalImageLIF_SpikeTimes' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',times)        
                np.save(path + 'NaturalImageLIF_BinnedSpikes' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',spike_array_binnedNat)
                np.save(path + 'NaturalImageLIF_MembranePotential' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',M.monitor.v)
                np.save(path + 'NaturalImageLIF_Probability' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',M.monitor.P)
                np.save(path + 'NaturalImageLIF_InputCurrent' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',M.monitor.I*1000)
                np.save(path + 'NaturalImageLIF_PSP' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',M.monitor.psp)
                np.save(path + 'NaturalImageLIF_FR' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',M.monitor.FR)
                
                # Storing indices of highest firing neurons in both Gibbs Sampling and LIF Simulations
                highest_fr_indx_samplesNat = np.flip(np.argsort(sum(samplesNat,1)))
                highest_fr_indx_LIFNat = np.flip(np.argsort(sum(spike_array_binnedNat,1)))
                h_smp = highest_fr_indx_samplesNat
                h_lif = highest_fr_indx_LIFNat
                h_smp = h_smp.astype(int)
                h_lif = h_lif.astype(int)
#                np.save(path + 'NaturalImageHighestFR_PFindex_samples' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',highest_fr_indx_samplesNat)
#                np.save(path + 'NaturalImageHighestFR_PFindex_LIF' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',highest_fr_indx_LIFNat)
        
        elif simulation_cases[sm]=='Grating':
            GratingImage_data = np.load('../Data/PreprocessedGratings-8-zca-norm.npy') # load grating image preprocessed 
            angles = np.load('../Data/Grating_angles.npy')
            sf_cases = np.load('../Data/Grating_sf.npy')
            sf_chosen_indx = 0
            chosen_angle_indx = 6  
            contrast = 0.25
            Grating_Image_all = np.squeeze(GratingImage_data[sf_chosen_indx,chosen_angle_indx,:,:])
            np.save(path + 'GratingImageAllPhases' + '_' + str(neurons_set[nn])  + '.npy',Grating_Image_all)
            num_ph = np.shape(Grating_Image_all)[0]
            grating_image_duration = 25
            print('Chosen Grating Image of angle ' + str(angles[chosen_angle_indx]) + ' of sf ' + str(sf_cases[sf_chosen_indx]))
            print('Computing for grating image with changing phases...')
            print('----------------------------------------------------------------------------------------------------------------------------------------------')
            for m in range(len(model_set)):
                params = load_simulation_parameters(neurons_set[nn],model_set[m])
                random_init = 1
                init_sample = np.zeros(params.N)
                init_membrane_potential = -75 * np.ones(params.N)
                n_samples = int(grating_image_duration/params.sampling_bin_ms)
                samples_all = np.zeros((params.N, num_ph*n_samples))
                spike_array_binned = np.zeros((params.N, num_ph*n_samples))
                ff_gr = np.zeros((params.N,n_samples*num_ph))
                rec_gr = np.zeros((params.N,n_samples*num_ph))
                gibbs_prob_gr = np.zeros((params.N,n_samples*num_ph))
                membrane_potential_gr = np.zeros((params.N,10*grating_image_duration*num_ph))
                lif_prob_gr = np.zeros((params.N,10*grating_image_duration*num_ph))
                input_current_gr = np.zeros((params.N,10*grating_image_duration*num_ph))
                psp_gr = np.zeros((params.N,10*grating_image_duration*num_ph))
                fr_gr = np.zeros((params.N,10*grating_image_duration*num_ph))
                times_ph = np.array([])
                indices_ph = np.array([])
                
                for j in range(num_ph):
                    Grating_Image = contrast * np.squeeze(Grating_Image_all[j,:])
                    samples, ff, rec, prob = Inference_Gibbs(params, Grating_Image, n_samples, random_init, init_sample)
                    M = LIFSamplingModel(params.N, params.G, init_membrane_potential, membrane_noise=0.0, prior=params.prior, pixel_noise=params.sigma_I , psp_length_ms=params.sampling_bin_ms, verbose=False, trial_duration_ms=grating_image_duration, photo_noise=params.photo_noise , photo_noise_refresh_ms=params.photo_noise_refresh_ms, rec_noise=params.rec_noise,random_init_process=random_init)
                    M.condition(Grating_Image)
                    spikes = M.simulate(monitor=["v", "P", "I", "psp", "FR"])
                    times_ph = np.append(times_ph,np.array(spikes.t/ms) + (j*grating_image_duration))
                    indices_ph = np.append(indices_ph,np.array(spikes.i))
                    indices_ph = indices_ph.astype(int)
                    
                    samples_all[:,j*n_samples:((j+1)*n_samples)] = samples
                    ff_gr[:,j*n_samples:((j+1)*n_samples)] = ff
                    rec_gr[:,j*n_samples:((j+1)*n_samples)] = rec
                    gibbs_prob_gr[:,j*n_samples:((j+1)*n_samples)] = prob
                    spike_array_binned[:,j*n_samples:((j+1)*n_samples)] = Extract_Spikes(params.N, grating_image_duration, params.sampling_bin_ms, spikes) 
                    membrane_potential_gr[:,j*10*grating_image_duration:((j+1)*10*grating_image_duration)] =  M.monitor.v
                    lif_prob_gr[:,j*10*grating_image_duration:((j+1)*10*grating_image_duration)] =  M.monitor.P
                    input_current_gr[:,j*10*grating_image_duration:((j+1)*10*grating_image_duration)] =  M.monitor.I
                    psp_gr[:,j*10*grating_image_duration:((j+1)*10*grating_image_duration)] =  M.monitor.psp
                    fr_gr[:,j*10*grating_image_duration:((j+1)*10*grating_image_duration)] =  M.monitor.FR

                    random_init = 0
                    init_sample = samples[:,-1]
                    init_membrane_potential = M.monitor.v[:,-1]
                print('Sampling and LIF done for grating image with changing phases for model ' + str(model_set[m]) + ' with ' + str(neurons_set[nn]) + ' neurons.')
                print('==============================================================================================================================================')
                np.save(path + 'GratingImageSamplesFF' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',ff_gr)
                np.save(path + 'GratingImageSamplesRec' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',rec_gr)
                np.save(path + 'GratingImageSamplesGibbsProb' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',gibbs_prob_gr)
                np.save(path + 'GratingImageSamples' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',samples_all)
                np.save(path + 'GratingImageLIF_SpikeIndices' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',indices_ph)
                np.save(path + 'GratingImageLIF_SpikeTimes' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',times_ph)        
                np.save(path + 'GratingImageLIF_BinnedSpikes' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',spike_array_binned)
                np.save(path + 'GratingImageLIF_MembranePotential' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',membrane_potential_gr)
                np.save(path + 'GratingImageLIF_Probability' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',lif_prob_gr)
                np.save(path + 'GratingImageLIF_InputCurrent' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',input_current_gr)
                np.save(path + 'GratingImageLIF_PSP' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',psp_gr)
                np.save(path + 'GratingImageLIF_FR' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',fr_gr)
                
                # Storing indices of highest firing neurons in both Gibbs Sampling and LIF Simulations
                highest_fr_indx_samples = np.flip(np.argsort(sum(samples_all,1)))
                temp_lif_indx = np.zeros(neurons_set[nn])
                for tt in range(neurons_set[nn]):
                    temp_lif_indx[tt] = (np.sum(indices_ph==tt))
                highest_fr_indx_LIF = np.flip(np.argsort(temp_lif_indx))#np.flip(np.argsort(sum(spike_array_binned,1)))
                h_smp = highest_fr_indx_samples
                h_lif = highest_fr_indx_LIF
                h_smp = h_smp.astype(int)
                h_lif = h_lif.astype(int)
                np.save(path + 'GratingImageHighestFR_PFindex_samples' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',highest_fr_indx_samples)
                np.save(path + 'GratingImageHighestFR_PFindex_LIF' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy',highest_fr_indx_LIF)
        
        else:
            print("No such image type!!")