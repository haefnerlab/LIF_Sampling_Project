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
model_set =  ['NN', 'IN','ISN']
neurons_set = [128]#, 64]
contrasts = np.array([0.3, 0.4, 0.5, 0.75, 1.0])
angles = np.load('../Data/Grating_angles.npy')
pix = 64 #image pixels
dimension = 8 #sqrt(image pixels)
simulation_cases =  ['GratingImage','NaturalImage']

#%%
sf_case = 1
for m in range(len(model_set)):
    for nn in range(len(neurons_set)):
        neurons = neurons_set[nn]
        model = model_set[m]
        params = load_simulation_parameters(neurons,model)
        sample_hertz = 1.0/(params.sampling_bin_s)
        grating_samples_all = np.zeros((100,len(contrasts),21,128,105))
        grating_spikes_all = np.zeros((100,len(contrasts),21,128,105))
        for ccn in range(len(contrasts)):
            print('Contrast ' +str(contrasts[ccn]) + ' for model ' + model_set[m])
            filename_smp = "SamplingTuningFR_" + "Con=" + str(contrasts[ccn]) + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '_sf' + str(sf_case) + '.npy'
            fr_sampling = np.load(path + filename_smp)
            
            filename_lif = "LIFTuningFR_" + "Con=" + str(contrasts[ccn]) + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '_sf' + str(sf_case) + '.npy'
            fr_lif = np.load(path + filename_lif)
            
            for neu in range(params.N):
                for im in range(21):
                    for rep in range(100):
                        t1 = 0
                        t2 = 5
                        for ph in range(21):
                            grating_samples_all[rep,ccn,im,neu,t1:t2] = fr_sampling[rep,im,ph,neu,:]
                            grating_spikes_all[rep,ccn,im,neu,t1:t2] = fr_lif[rep,im,ph,neu,:]
                            t1 = t2
                            t2 = t2 + 5
            
        filename1 = 'SamplesforStatistics_GratingImages' + '_' + str(neurons) + '_' + model + '.npy'
        filename2 = 'SpikesforStatistics_GratingImages' + '_' + str(neurons) + '_' + model + '.npy'
                
        np.save(path + filename1,grating_samples_all )
        np.save(path + filename2, grating_spikes_all)

#%%


for sm in range(len(simulation_cases)):
    for m in range(len(model_set)):
        for nn in range(len(neurons_set)):
            neurons = neurons_set[nn]
            model = model_set[m]
            params = load_simulation_parameters(neurons,model)
            sample_hertz = 1.0/(params.sampling_bin_s)
           
            if simulation_cases[sm]=='NaturalImage':
                filename1 = 'SamplesforStatistics_NaturalImages' + '_' + str(neurons) + '_' + model + '.npy'
                filename2 = 'SpikesforStatistics_NaturalImages' + '_' + str(neurons) + '_' + model + '.npy'
                
                samples = np.load(path + filename1)
                spikes = np.load(path + filename2)
        
                num_im = np.shape(samples)[1]
                num_repeats = np.shape(samples)[0]
                n_samples = np.shape(samples)[3]
                bnd1 = 0
                bnd2 = 0
                 
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
                    
                np.save(path + 'SamplingStats_ISI_NaturalImages_' + str(neurons) + '_' + model + '.npy',sampling_isi)
                np.save(path + 'SamplingStats_CV_NaturalImages_' + str(neurons) + '_' + model + '.npy',sampling_cv)
                np.save(path + 'SamplingStats_FF_NaturalImages_' + str(neurons) + '_' + model + '.npy',sampling_ff)
                np.save(path + 'SamplingStats_Corr_NaturalImages_' + str(neurons) + '_' + model + '.npy',sampling_corr)
                np.save(path + 'SamplingStats_SigCorr_NaturalImages_' + str(neurons) + '_' + model + '.npy',sampling_sig_corr)
                
                np.save(path + 'LIFStats_ISI_NaturalImages_' + str(neurons) + '_' + model + '.npy',LIF_isi)
                np.save(path + 'LIFStats_CV_NaturalImages_' + str(neurons) + '_' + model + '.npy',LIF_cv)
                np.save(path + 'LIFStats_FF_NaturalImages_' + str(neurons) + '_' + model + '.npy',LIF_ff)
                np.save(path + 'LIFStats_Corr_NaturalImages_' + str(neurons) + '_' + model + '.npy',LIF_corr)
                np.save(path + 'LIFStats_SigCorr_NaturalImages_' + str(neurons) + '_' + model + '.npy',LIF_sig_corr)   
            
            else:
                filename1 = 'SamplesforStatistics_GratingImages' + '_' + str(neurons) + '_' + model + '.npy'
                filename2 = 'SpikesforStatistics_GratingImages' + '_' + str(neurons) + '_' + model + '.npy'
                
                samples = np.load(path + filename1)
                spikes = np.load(path + filename2)
                num_con = np.shape(samples)[1]
                num_im = np.shape(samples)[2]
                num_repeats = np.shape(samples)[0]
                n_samples = np.shape(samples)[4]
                bnd1 = 0
                bnd2 = 1
                 
#                sampling_isi1 = np.zeros((num_con,num_im,params.N,n_samples))
#                sampling_cv1 = np.zeros((num_con,num_im,params.N))
                sampling_isi = np.zeros((num_con,num_im,num_repeats,params.N,n_samples))
                sampling_cv = np.zeros((num_con,num_im,params.N))
                sampling_ff = np.zeros((num_con,num_im,params.N))
                sampling_corr = np.zeros((num_con,num_im,params.N,params.N))
                sampling_sig_corr = np.zeros((num_con,num_im,params.N,params.N))
                
#                LIF_isi1 = np.zeros((num_con,num_im,params.N,n_samples))
#                LIF_cv1 = np.zeros((num_con,num_im,params.N))
                LIF_isi = np.zeros((num_con,num_im,num_repeats,params.N,n_samples))
                LIF_cv = np.zeros((num_con,num_im,params.N))
                LIF_ff = np.zeros((num_con,num_im,params.N))
                LIF_corr = np.zeros((num_con,num_im,params.N,params.N))
                LIF_sig_corr = np.zeros((num_con,num_im,params.N,params.N))
                
                for ccn in range(num_con):
                    for i in range(num_im):
                        print("Counting image " + str((i+1)+(ccn*num_im)) + " for model " + model)
                        temp_samples = np.squeeze(samples[:,ccn,i,:,:])
                        temp_spikes = np.squeeze(spikes[:,ccn,i,:,:])
                        FF_smp, CV_smp, corr_smp, ISI_smp, sig_corr_smp = compute_cv_ff_corr(temp_samples,params,bnd=bnd1)
                        FF_lif, CV_lif, corr_lif, ISI_lif, sig_corr_lif = compute_cv_ff_corr(temp_spikes,params,bnd=bnd2)
    
                        sampling_isi[ccn,i,:,:,:] = ISI_smp
                        sampling_cv[ccn,i,:] = CV_smp                    
                        sampling_ff[ccn,i,:] = FF_smp
                        sampling_corr[ccn,i,:,:] = corr_smp
                        sampling_sig_corr[ccn,i,:,:] = sig_corr_smp
                        
                        LIF_isi[ccn,i,:,:,:] = ISI_lif
                        LIF_cv[ccn,i,:] = CV_lif
                        LIF_ff[ccn,i,:] = FF_lif
                        LIF_corr[ccn,i,:,:] = corr_lif
                        LIF_sig_corr[ccn,i,:,:] = sig_corr_lif
                    
                np.save(path + 'SamplingStats_ISI_GratingImages_' + str(neurons) + '_' + model + '.npy',sampling_isi)
                np.save(path + 'SamplingStats_CV_GratingImages_' + str(neurons) + '_' + model + '.npy',sampling_cv)
                np.save(path + 'SamplingStats_FF_GratingImages_' + str(neurons) + '_' + model + '.npy',sampling_ff)
                np.save(path + 'SamplingStats_Corr_GratingImages_' + str(neurons) + '_' + model + '.npy',sampling_corr)
                np.save(path + 'SamplingStats_SigCorr_GratingImages_' + str(neurons) + '_' + model + '.npy',sampling_sig_corr)
                
                np.save(path + 'LIFStats_ISI_GratingImages_' + str(neurons) + '_' + model + '.npy',LIF_isi)
                np.save(path + 'LIFStats_CV_GratingImages_' + str(neurons) + '_' + model + '.npy',LIF_cv)
                np.save(path + 'LIFStats_FF_GratingImages_' + str(neurons) + '_' + model + '.npy',LIF_ff)
                np.save(path + 'LIFStats_Corr_GratingImages_' + str(neurons) + '_' + model + '.npy',LIF_corr)
                np.save(path + 'LIFStats_SigCorr_GratingImages_' + str(neurons) + '_' + model + '.npy',LIF_sig_corr)
            

