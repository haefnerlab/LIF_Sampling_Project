
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

## Initialize variables    
model_set = ['NN', 'IN','ISN']
model = 'ISN'
neurons_set = [128]#, 64]
pix = 64 #image pixels
dimension = 8 #sqrt(image pixels)
path = '../Data/'
NaturalImage_data = np.load('../Data/PreprocessedImagePatches-8-zca-norm.npy') # load natural image reprocessed patches 
max_elms = 200
chosen_neuron = 15
num_rep = 50
#%%
for nn in range(len(neurons_set)):
    selected_natural_images = np.zeros((max_elms,8,8))
    selected_natural_image_indices = np.zeros(max_elms)
    k = 0
    flag = 1
    t = 0
    LIF_isi = np.zeros((max_elms,num_rep,128,100))
    LIF_cv = np.zeros((max_elms,128))
    LIF_ff = np.zeros((max_elms,128))
    LIF_corr = np.zeros((max_elms,128,128))
    LIF_sig_corr = np.zeros((max_elms,128,128))
    while flag==1:
        chosen_natural_im = t
        print('Chosen Natural Image number:' + str(chosen_natural_im))
        NaturalImage_chosen = np.squeeze(NaturalImage_data[chosen_natural_im,:,:])
        p_mx = np.zeros(len(model_set))
#        p_mn = np.zeros(len(model_set))
        p = np.zeros((len(model_set),neurons_set[nn]))
        FF_smp = np.zeros((len(model_set),neurons_set[nn]))
        CV_smp = np.zeros((len(model_set),neurons_set[nn]))
        Corr_smp = np.zeros((len(model_set),neurons_set[nn],neurons_set[nn]))
        for m in range(len(model_set)):
            params = load_simulation_parameters(neurons_set[nn],model_set[m])
            sample_hertz = 1.0/(params.sampling_bin_s)
            natural_image_duration = 500
            n_samples = int(natural_image_duration/params.sampling_bin_ms)
            
            random_init = 1
            init_sample = np.zeros(params.N)
            Image = NaturalImage_data[chosen_natural_im,:,:]
            samplesNat,_,_,_ = Inference_Gibbs(params, NaturalImage_chosen, n_samples, random_init, init_sample)
            print ('Sampling Done for model ' + str(model_set[m]))
            
            p[m,:] = np.sum(samplesNat,1)/n_samples#marg_prob_compute(samplesNat,params.N)
#            p_mn[m] = np.sum(p[m,:]>0.1)
            p_mx[m] = p[m,chosen_neuron]
        print('Found images = ' + str(k))
        t = t + 1
#        if(np.min(p_mn)>=5 and np.max(p_mx)<=0.4):
        print('Probs for 3 models: ' + str(p_mx))
        if(np.min(p_mx)>=0.1 and np.max(p_mx)<=0.4):
            spikes_im = np.zeros((num_rep,params.N,100))
            print("Computing LIF stats...")
            
            for rep in range(50):
               if np.mod(rep,25)==0 and rep>0:
                   print("Halfway there...")
               init_membrane_potential = -70 * np.ones(params.N)
               M = LIFSamplingModel(params.N, params.G, init_membrane_potential, membrane_noise=0.0, prior=params.prior, pixel_noise=params.sigma_I , psp_length_ms=params.sampling_bin_ms, verbose=False, trial_duration_ms=natural_image_duration, photo_noise=params.photo_noise , photo_noise_refresh_ms=params.photo_noise_refresh_ms, rec_noise=params.rec_noise,random_init_process=random_init)
               M.condition(NaturalImage_chosen)
               spikesNat = M.simulate(monitor=["v", "P", "I", "psp", "FR", "is_active"])
               # Obtaining binned spike train from BRIAN simulations
               spikes_im[rep,:,:] = Extract_Spikes(params.N, natural_image_duration, params.sampling_bin_ms, spikesNat) 
            FF_lif, CV_lif, corr_lif, ISI_lif, sig_corr_lif = compute_cv_ff_corr(spikes_im,params,bnd=0)
            if FF_lif[chosen_neuron]>=0.5:
               selected_natural_images[k,:,:] = NaturalImage_chosen
               selected_natural_image_indices[k] = t - 1
               LIF_isi[k,:,:,:] = ISI_lif
               LIF_cv[k,:] = CV_lif
               LIF_ff[k,:] = FF_lif
               LIF_corr[k,:,:] = corr_lif
               LIF_sig_corr[k,:,:] = sig_corr_lif
               k = k + 1
    #           print('Probs for 3 models: ' + str(p_mx))
               print('Found Natural Image number '+str(k))
               if k==max_elms:
                   flag=0
        print("==================================================================================")
    filename = 'SuitableNaturalImagesSingleNeuron' + '_' + str(neurons_set[nn]) + '.npy'
    np.save(path + filename,selected_natural_images)
    filename1 = 'SuitableNaturalImageIndicesSingleNeuron' + '_' + str(neurons_set[nn]) + '.npy'
    np.save(path + filename1,selected_natural_image_indices)
    
    np.save(path + 'LIFStats_ISI_NaturalImagesSingleNeuron_' + str(neurons_set[nn]) + '_' + model + '.npy',LIF_isi)
    np.save(path + 'LIFStats_CV_NaturalImagesSingleNeuron_' + str(neurons_set[nn]) + '_' + model + '.npy',LIF_cv)
    np.save(path + 'LIFStats_FF_NaturalImagesSingleNeuron_' + str(neurons_set[nn]) + '_' + model + '.npy',LIF_ff)
    np.save(path + 'LIFStats_Corr_NaturalImagesSingleNeuron_' + str(neurons_set[nn]) + '_' + model + '.npy',LIF_corr)
    np.save(path + 'LIFStats_SigCorr_NaturalImagesSingleNeuron_' + str(neurons_set[nn]) + '_' + model + '.npy',LIF_sig_corr)
        
               
               
           
           
           