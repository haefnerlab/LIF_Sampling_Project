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
model = 'IN'
neurons_set = [128]#, 64]
contrasts = np.array([0.25, 0.5, 0.75, 1.0])
angles = np.load('../Data/Grating_angles.npy')
pix = 64 #image pixels
dimension = 8 #sqrt(image pixels)
simulation_cases =  ['NaturalImage']#,'Grating']
num_repeats = 30
durations = np.array([500, 5000, 10000])
ff_noise = np.array([np.sqrt(0.1), np.sqrt(0.5), np.sqrt(1.0), np.sqrt(2.0), np.sqrt(3.0)])
bound = 5

#%%

for sm in range(len(simulation_cases)):   
    for nn in range(len(neurons_set)):
        data = np.load('../Data/SuitableNaturalImages_'+str(neurons_set[nn])+'.npy') 
        data = data[range(5),:,:]
        num_im = np.shape(data)[0]
        print("Number of images is " + str(num_im))
        neurons = neurons_set[nn]
        params = load_simulation_parameters(neurons,model)
        sample_hertz = 1.0/(params.sampling_bin_s) 
                    
        
        

#            LIF_isi = np.zeros((num_im,num_repeats,params.N,n_samples))
#            LIF_cv = np.zeros((num_im,params.N))
#            LIF_ff = np.zeros((num_im,params.N))
#            LIF_corr = np.zeros((num_im,params.N,params.N))
#            LIF_sig_corr = np.zeros((num_im,params.N,params.N))
        kk = 0
        for d in range(len(durations)):
            natural_image_duration = durations[d]
            n_samples = int(natural_image_duration/params.sampling_bin_ms)
            sampling_isi = np.zeros((len(ff_noise),num_im,num_repeats,params.N,n_samples))
            sampling_cv = np.zeros((len(ff_noise),num_im,params.N))
            sampling_ff = np.zeros((len(ff_noise),num_im,params.N))
            sampling_corr = np.zeros((len(ff_noise),num_im,params.N,params.N))
            sampling_sig_corr = np.zeros((len(ff_noise),num_im,params.N,params.N))
            samples_im = np.zeros((len(ff_noise),num_repeats,num_im,params.N,n_samples))
            for noi in range(len(ff_noise)):
                params.photo_noise = ff_noise[noi]
                if simulation_cases[sm]=='NaturalImage':
#                        spikes_im = np.zeros((len(durations),len(ff_noise),num_repeats,num_im,params.N,100))
                    
                    for i in range(num_im):
                        for nr in range(num_repeats):
                            print ("Counting " + str(kk+1) + " out of " + str(num_repeats*num_im*len(durations)*len(ff_noise)))
                            NaturalImage_chosen = np.squeeze(data[i,:,:])
                            
                            random_init = 1 
                            # not useful if ranom_init = 1 else, sets initial samples and membrane potential to values as in init_sample and init_membrane_potential
                            init_sample = np.zeros(params.N)
#                                init_membrane_potential = -70 * np.ones(params.N)
                            # Simulating Gibbs sampling and returning samples, feedforward input, recurrent input and Gibbs probability
                            samplesNat, ff, rec, prob = Inference_Gibbs(params, NaturalImage_chosen, n_samples, random_init, init_sample)
                            # Simulating LIF firing in BRIAN and returning spiketimes with corresponding neuron indices. 
                            # Also monitoring voltage, probability, input current, post synaptic potential and firing rate
#                                M = LIFSamplingModel(params.N, params.G, init_membrane_potential, membrane_noise=0.0, prior=params.prior, pixel_noise=params.sigma_I , psp_length_ms=params.sampling_bin_ms, verbose=False, trial_duration_ms=natural_image_duration, photo_noise=params.photo_noise , photo_noise_refresh_ms=params.photo_noise_refresh_ms, rec_noise=params.rec_noise,random_init_process=random_init)
#                                M.condition(NaturalImage_chosen)
#                                spikesNat = M.simulate(monitor=["v", "P", "I", "psp", "FR", "is_active"])
                            # Obtaining binned spike train from BRIAN simulations
                            kk = kk+1
                            samples_im[noi,nr,i,:,:] = samplesNat
#                                spikes_im[noi,nr,i,:,:] = Extract_Spikes(params.N, natural_image_duration, params.sampling_bin_ms, spikesNat) 
                        
#                        
                    temp_samples = np.squeeze(samples_im[noi,:,i,:,:])
                    FF_smp, CV_smp, corr_smp, ISI_smp, sig_corr_smp = compute_cv_ff_corr(temp_samples,params,bnd=bound)
                    sampling_isi[noi,i,:,:,:] = ISI_smp
                    sampling_cv[noi,i,:] = CV_smp                    
                    sampling_ff[noi,i,:] = FF_smp
                    sampling_corr[noi,i,:,:] = corr_smp
                    sampling_sig_corr[noi,i,:,:] = sig_corr_smp
#           np.save(path + 'SpikesforStatistics_NaturalImagesChangingFF_And_Duration' + '_' + str(neurons) + '_' + model + '.npy',spikes_im)
            np.save(path + 'Samples_NaturalImagesChangingFFNoise_And_Duration' + str(durations[d])+ '_' + str(neurons) + '.npy',samples_im)
            np.save(path + 'SamplingStats_ISI_NaturalImagesChangingFFNoise_And_Duration' + str(durations[d])+ '_'  + str(neurons)  + '.npy',sampling_isi)
            np.save(path + 'SamplingStats_CV_NaturalImagesChangingFFNoise_And_Duration' + str(durations[d])+ '_'  + str(neurons) + '.npy',sampling_cv)
            np.save(path + 'SamplingStats_FF_NaturalImagesChangingFFNoise_And_Duration' + str(durations[d])+ '_'  + str(neurons) + '.npy',sampling_ff)
            np.save(path + 'SamplingStats_Corr_NaturalImagesChangingFFNoise_And_Duration' + str(durations[d])+ '_'  + str(neurons) + '.npy',sampling_corr)
            np.save(path + 'SamplingStats_SigCorr_NaturalImagesChangingFFNoise_And_Duration' + str(durations[d])+ '_'  + str(neurons) + '.npy',sampling_sig_corr)        
#%%
durations = np.array([500, 5000, 10000])
ff_noise = np.array([np.sqrt(0.1), np.sqrt(0.5), np.sqrt(1.0), np.sqrt(2.0), np.sqrt(3.0)])
neurons = 128                  
for sm in range(len(simulation_cases)):   
    for nn in range(len(neurons_set)):                    
        plt.figure() 
        cnt = 0
        for d in range(len(durations)):
            Sampling_Corr = np.load(path + 'SamplingStats_Corr_NaturalImagesChangingFFNoise_And_Duration' + str(durations[d])+ '_'  + str(neurons) + '.npy')
            Sampling_SigCorr = np.load(path + 'SamplingStats_SigCorr_NaturalImagesChangingFFNoise_And_Duration' + str(durations[d])+ '_'  + str(neurons) + '.npy')        
            
            for noi in range(len(ff_noise)):                   
                Sampling_Corr1 =  np.squeeze(Sampling_Corr[noi,:,:,:])
                Sampling_SigCorr1 = np.squeeze(Sampling_SigCorr[noi,:,:,:])                   
                
                sampled_Corr_all = Sampling_Corr1.flatten()
                sampled_SigCorr_all = Sampling_SigCorr1.flatten() 
                sampled_SigCorr_all = sampled_SigCorr_all[np.logical_not(np.isnan(sampled_Corr_all))]
                sampled_Corr_all = sampled_Corr_all[np.logical_not(np.isnan(sampled_Corr_all))]
                sampled_SigCorr_all = sampled_SigCorr_all[np.logical_not((np.abs(sampled_Corr_all)==1.0))]
                sampled_Corr_all = sampled_Corr_all[np.logical_not((np.abs(sampled_Corr_all)==1.0))]
                sampled_SigCorr_all = sampled_SigCorr_all[np.logical_not((np.abs(sampled_Corr_all)==100))]
                sampled_Corr_all = sampled_Corr_all[np.logical_not((np.abs(sampled_Corr_all)==100))]
                #sampled_Corr_all = sampled_Corr_all[np.logical_not((np.abs(sampled_Corr_all)>0.9))]                    
    

                plt.subplot(len(durations),len(ff_noise),cnt+1)
                sns.set_style("whitegrid", {'axes.grid' : False})
                plt.scatter(sampled_SigCorr_all,sampled_Corr_all,color='green',edgecolors= "black")
                plt.xlabel('Signal correlation',fontsize=15)
                plt.ylabel('Sampling based noise correlation',fontsize=15)
                plt.title("duration = " + str(durations[d]) + " and ff noise = " + str(ff_noise[noi]))
                corr = np.corrcoef(sampled_SigCorr_all,sampled_Corr_all)[0][1]
                slp, b = np.polyfit(sampled_SigCorr_all,sampled_Corr_all, 1)
                plt.vlines(0.0,-1.5,1.5,'k',linewidth=0.5,linestyle='--')
                plt.hlines(0.0,-1.5,1.5,'k',linewidth=1,linestyle='--')
                #            plt.xlim(0.0,2)
                #            plt.xticks([0.0,1.0,2.0]) 
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                #            plt.ylim(-0.03,5.5)
                plt.gca().spines['right'].set_visible(False)
                plt.gca().spines['top'].set_visible(False)
                plt.text(1.0, 1.0, 'Slope = '+str(np.round(slp,2)), fontsize=10)
                plt.text(1.0, 0.75, 'Corr = '+str(np.round(corr,2)), fontsize=10)

                cnt = cnt + 1
    
        
        
                    
                    
                    
                    
                    
                    