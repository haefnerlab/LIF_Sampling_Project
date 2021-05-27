
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
from Tools.compute_delta_activity_for_stimulation import *
from Tools.load_image_patches import *
from Tools.spiketrain_statistics import *

from Sampling_Functions.Marg_samples import *
from Sampling_Functions.Inference_Gibbs import *
from Sampling_Functions.Inference_Gibbs_rate_photostimulate import *  

from LIF_Functions.LIF_spikes_from_BRIAN import *
from LIF_Functions.LIF_with_BRIAN import LIFSamplingModel
from LIF_Functions.LIF_with_BRIAN_rate_photostimulate import LIFSamplingModel_stimulate

path = '../Results/'
model_set = ['NN', 'IN','ISN']
neurons_set = [128]
pix = 64


for m in range(len(model_set)):
    for nn in range(len(neurons_set)):
        
        neurons = neurons_set[nn]
        model = model_set[m]
        params = load_simulation_parameters(neurons,model)
        sample_hertz = 1.0/(params.sampling_bin_s)
        
        selected_neuron = 1
        dimension = 8
        num_im = 1
        chosen = 8
        temp_data = np.load('../Data/PreprocessedGratings-8-zca-norm.npy')#load_natural_image_patches(dimension)
        temp_data = 0.1*np.reshape(temp_data[0,:,:,:],(21,21,8,8))
        data = temp_data[chosen,:,:,:]
        bin_rng = 10
        extra_rate = 6.38 * double(params.duration)/250.0
        extra_prob = double(extra_rate)/sample_hertz
        stimulation_noise_sig =  0
        
        n_samples = int(params.duration/params.sampling_bin_ms)
        t = np.ones(n_samples)
        t[int(n_samples/2):n_samples] = t[int(n_samples/2):n_samples]*0
        rec_noise_sample = params.rec_noise
        photo_noise_sample = params.photo_noise

        Sampling_fr_no_stimulation = np.zeros((num_im,params.N))
        Sampling_fr_stimulation = np.zeros((num_im,params.N))
        LIF_fr_no_stimulation = np.zeros((num_im,params.N))
        LIF_fr_stimulation = np.zeros((num_im,params.N))
        
        for i in range(num_im):
            print i
            Image = np.reshape(np.squeeze(data[i,:,:]),(64))# + sigma_I**2 * np.random.normal(0,1,pix)
            random_init = 1
            init_sample = np.zeros(params.N)
            samples1,_,_,_ = Inference_Gibbs(params, Image, n_samples, random_init, init_sample)
            print ('No Stimulation Sampling Done')
            
            samples2 = Inference_Gibbs_stimulate(params.G, params.sigma_I, photo_noise_sample, params.prior, params.N, Image, n_samples, rec_noise_sample, selected_neuron, t, extra_prob)
            print ('Stimulation Sampling Done')
            
            M1 = LIFSamplingModel(params.N, params.G,-75 * np.ones(params.N), membrane_noise=0.0, prior=params.prior, pixel_noise=params.sigma_I , psp_length_ms=params.sampling_bin_ms, verbose=True, trial_duration_ms=params.duration, photo_noise=params.photo_noise , photo_noise_refresh_ms=params.photo_noise_refresh_ms, rec_noise=params.rec_noise,random_init_process=random_init)
            M1.condition(Image)
            spikes1 = M1.simulate(monitor=["v", "P", "I", "psp", "FR"])
            print ('No Stimulation LIF Done')
            
            M2 = LIFSamplingModel_stimulate(params.N, params.G, membrane_noise=0.0, prior=params.prior, pixel_noise=params.sigma_I , psp_length_ms=params.sampling_bin_ms, verbose=True, trial_duration_ms=params.duration, photo_noise=params.photo_noise , photo_noise_refresh_ms=params.photo_noise_refresh_ms, rec_noise=params.rec_noise, photostim_rate_increase=extra_rate, stimulation_noise_sig=stimulation_noise_sig, photostim_cell=selected_neuron, stimulation_duration_ms=params.duration/2)
            M2.condition(Image)
            spikes2 = M2.simulate(monitor=["v", "P", "I", "psp", "FR"])
            print ('Stimulation LIF Done')
            
            times1 = np.array(spikes1.t/ms)
            indices1 = np.array(spikes1.i)
            spike_array_binned1 = Extract_Spikes(params.N, params.duration, params.sampling_bin_ms, spikes1) 
            
            times2 = np.array(spikes2.t/ms)
            indices2 = np.array(spikes2.i)
            spike_array_binned2 = Extract_Spikes(params.N, params.duration, params.sampling_bin_ms, spikes2) 
            
            print ('Spikes Done')
            
            LIF_fr_no_stimulation[i,:] = np.sum(spike_array_binned1,1)/spike_array_binned1.shape[1] * sample_hertz
            LIF_fr_stimulation[i,:] = np.sum(spike_array_binned2,1)/spike_array_binned2.shape[1] * sample_hertz
                     
            Sampling_fr_no_stimulation[i,:] = marg_prob_compute(samples1,params.N) * sample_hertz
            Sampling_fr_stimulation[i,:] = marg_prob_compute(samples2,params.N) * sample_hertz
            

        np.save(path + 'Sampling_fr_no_stimulation' + str(neurons) + '_' + model + '.npy',Sampling_fr_no_stimulation)
        np.save(path + 'Sampling_fr_stimulation' + str(neurons) + '_' + model + '.npy',Sampling_fr_stimulation)
        
#        np.save(path + 'LIF_fr_no_stimulation' + str(neurons) + '_' + model + '.npy',LIF_fr_no_stimulation)
#        np.save(path + 'LIF_fr_stimulation' + str(neurons) + '_' + model + '.npy',LIF_fr_stimulation)            
        
        intervals = 4
        
        sig_corr_LIF,mean_influence_LIF,binned_mid_sig_corr_LIF,binned_mean_influence_LIF,err_LIF = compute_delta_activity(params.G, selected_neuron, params.N, num_im, LIF_fr_no_stimulation, LIF_fr_stimulation, intervals)
        slope_LIF, intercept_LIF, r_value_LIF, p_value_LIF, std_err_LIF = stats.linregress(sig_corr_LIF, mean_influence_LIF)
        print(slope_LIF)
        
        sig_corr_Gibbs,mean_influence_Gibbs,binned_mid_sig_corr_Gibbs,binned_mean_influence_Gibbs,err_Gibbs = compute_delta_activity(params.G, selected_neuron, params.N, num_im, Sampling_fr_no_stimulation, Sampling_fr_stimulation, intervals)
        slope_Gibbs, intercept_Gibbs, r_value_Gibbs, p_value_Gibbs, std_err_Gibbs = stats.linregress(sig_corr_Gibbs, mean_influence_Gibbs)
        print(slope_Gibbs)
        
        np.save(path + 'Sampling_binned_sig_corr' + str(neurons) + '_' + model + '.npy',binned_mid_sig_corr_Gibbs)
        np.save(path + 'Sampling_binned_mean_influence' + str(neurons) + '_' + model + '.npy',binned_mean_influence_Gibbs)            
        np.save(path + 'Sampling_err' + str(neurons) + '_' + model + '.npy',err_Gibbs)            
        np.save(path + 'Sampling_sig_corr' + str(neurons) + '_' + model + '.npy',sig_corr_Gibbs)
        np.save(path + 'Sampling_mean_influence' + str(neurons) + '_' + model + '.npy',mean_influence_Gibbs)
        np.save(path + 'Sampling_slope_of_fit' + str(neurons) + '_' + model + '.npy',slope_Gibbs)
        
#        np.save(path + 'LIF_binned_sig_corr' + str(neurons) + '_' + model + '.npy',binned_mid_sig_corr_LIF)
#        np.save(path + 'LIF_binned_mean_influence' + str(neurons) + '_' + model + '.npy',binned_mean_influence_LIF)            
#        np.save(path + 'LIF_err' + str(neurons) + '_' + model + '.npy',err_LIF)            
#        np.save(path + 'LIF_sig_corr' + str(neurons) + '_' + model + '.npy',sig_corr_LIF)
#        np.save(path + 'LIF_mean_influence' + str(neurons) + '_' + model + '.npy',mean_influence_LIF)
#        np.save(path + 'LIF_slope_of_fit' + str(neurons) + '_' + model + '.npy',slope_LIF)
        
        
        plt.figure()
        plt.subplot(1,2,1)
        plt.errorbar(binned_mid_sig_corr_LIF, binned_mean_influence_LIF, yerr=err_LIF,c='k',fmt='--o',capsize=2)
        plt.hold('on')
        plt.scatter(sig_corr_LIF,mean_influence_LIF,s=10)
        plt.hold('on')
        plt.plot(np.zeros(100),np.linspace(-1,1,100),'k')
        plt.hold('on')
        plt.plot(np.linspace(-0.9,0.9,100),np.zeros(100),'k')
        plt.hold('on')
        plt.xlabel('Signal Corr')
        plt.ylabel('Change in Firing Rate')
        plt.title('LIF')
        plt.text(0.25, 0.5,("Beta = {0:.2f}".format(round(slope_LIF,2))) , fontsize=12)
        
        plt.subplot(1,2,2)
        plt.errorbar(binned_mid_sig_corr_Gibbs, binned_mean_influence_Gibbs, err_Gibbs,c='k',fmt='--o',capsize=2)
        plt.hold('on')
        plt.scatter(sig_corr_Gibbs,mean_influence_Gibbs,s=10)
        plt.hold('on')
        plt.plot(np.zeros(100),np.linspace(-1,1,100),'k')
        plt.hold('on')
        plt.plot(np.linspace(-0.9,0.9,100),np.zeros(100),'k')
        plt.hold('on')
        plt.xlabel('Sig Corr')
        plt.ylabel('Change in Firing Rate')
        plt.title('Sampling')
        plt.text(0.25, 0.5,("Beta = {0:.2f}".format(round(slope_Gibbs,2))) , fontsize=12)
        
        plt.suptitle([str(neurons) + '_' + model])
        plt.show()