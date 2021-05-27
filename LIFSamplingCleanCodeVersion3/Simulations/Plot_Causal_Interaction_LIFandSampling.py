#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:47:39 2020

@author: achattoraj
"""

import sys
sys.path.append('../')
from brian2 import *
import numpy as np
import array
import scipy as sp
import scipy.stats
import itertools
from itertools import combinations
prefs.codegen.target='numpy' #important!! throws errors otherwise
from scipy.misc import imshow
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import scipy.io
from LIF_Functions.LIF_with_BRIAN import LIFSamplingModel
from Tools.params_generate import *
from LIF_Functions.LIF_spikes_from_BRIAN import *
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy import stats
from numpy.linalg import inv
from scipy.optimize import curve_fit
from scipy import stats
from Tools.compute_delta_activity_for_stimulation import *
from Tools.load_params import *
from Tools.load_gratings import *

path = '../Results/'
model_set = ['NN', 'IN','ISN']
neurons_set = [128]#, 128]
colors1 = np.array(['greenyellow','green','darkgreen'])
colors2 = np.array(['dodgerblue','blue','darkblue'])
colors3 = np.array(['lightcoral','red','darkred'])

for nn in range(len(neurons_set)):
    plt.figure()
    for m in range(len(model_set)):
        
        neurons = neurons_set[nn]
        model = model_set[m]
        params = load_simulation_parameters(neurons,model)

        if m==0:
            colors = colors1[-1]
            colors1 = colors1[0]
        elif m==1:
            colors = colors2[-1]
            colors1 = colors2[0]
        elif m==2:
            colors = colors3[-1]
            colors1 = colors3[0]
        
        
        Sampling_fr_no_stimulation = np.load(path + 'Sampling_fr_no_stimulation' + str(neurons) + '_' + model + '.npy')
        Sampling_fr_stimulation = np.load(path + 'Sampling_fr_stimulation' + str(neurons) + '_' + model + '.npy')
        
        LIF_fr_no_stimulation = np.load(path + 'LIF_fr_no_stimulation' + str(neurons) + '_' + model + '.npy')
        LIF_fr_stimulation = np.load(path + 'LIF_fr_stimulation' + str(neurons) + '_' + model + '.npy')            
        
        intervals = 4
        
        sig_corr_LIF,mean_influence_LIF,binned_mid_sig_corr_LIF,binned_mean_influence_LIF,err_LIF = compute_delta_activity(params.G, selected_neuron, params.N, num_im, LIF_fr_no_stimulation, LIF_fr_stimulation, intervals)
        slope_LIF, intercept_LIF, r_value_LIF, p_value_LIF, std_err_LIF = stats.linregress(sig_corr_LIF, mean_influence_LIF)
        print(slope_LIF)
        
        sig_corr_Gibbs,mean_influence_Gibbs,binned_mid_sig_corr_Gibbs,binned_mean_influence_Gibbs,err_Gibbs = compute_delta_activity(params.G, selected_neuron, params.N, num_im, Sampling_fr_no_stimulation, Sampling_fr_stimulation, intervals)
        slope_Gibbs, intercept_Gibbs, r_value_Gibbs, p_value_Gibbs, std_err_Gibbs = stats.linregress(sig_corr_Gibbs, mean_influence_Gibbs)
        print(slope_Gibbs)
        
#        binned_mid_sig_corr_Gibbs = np.load(path + 'Sampling_binned_sig_corr' + str(neurons) + '_' + model + '.npy')
#        binned_mean_influence_Gibbs = np.load(path + 'Sampling_binned_mean_influence' + str(neurons) + '_' + model + '.npy')            
#        err_Gibbs = np.load(path + 'Sampling_err' + str(neurons) + '_' + model + '.npy')            
#        sig_corr_Gibbs = np.load(path + 'Sampling_sig_corr' + str(neurons) + '_' + model + '.npy')
#        mean_influence_Gibbs = np.load(path + 'Sampling_mean_influence' + str(neurons) + '_' + model + '.npy')
#        slope_Gibbs = np.load(path + 'Sampling_slope_of_fit' + str(neurons) + '_' + model + '.npy')
#        
#        binned_mid_sig_corr_LIF = np.load(path + 'LIF_binned_sig_corr' + str(neurons) + '_' + model + '.npy')
#        binned_mean_influence_LIF = np.load(path + 'LIF_binned_mean_influence' + str(neurons) + '_' + model + '.npy')            
#        err_LIF = np.load(path + 'LIF_err' + str(neurons) + '_' + model + '.npy')            
#        sig_corr_LIF = np.load(path + 'LIF_sig_corr' + str(neurons) + '_' + model + '.npy')
#        mean_influence_LIF = np.load(path + 'LIF_mean_influence' + str(neurons) + '_' + model + '.npy')
#        slope_LIF = np.load(path + 'LIF_slope_of_fit' + str(neurons) + '_' + model + '.npy')
        
        
       
        plt.subplot(2,3,4+m)
        plt.errorbar(binned_mid_sig_corr_LIF, binned_mean_influence_LIF, yerr=err_LIF,c=colors,fmt='--o',capsize=0.25)
        plt.hold('on')
        plt.scatter(sig_corr_LIF,mean_influence_LIF,s=10, color = colors1)
        plt.hold('on')
#        plt.plot(np.zeros(100),np.linspace(min(mean_influence_LIF),max(mean_influence_LIF),100).T,'--k')
        plt.plot(np.zeros(100),np.linspace(-0.2,0.2,100).T,'--k')
        plt.hold('on')
#        plt.plot(np.linspace(min(sig_corr_LIF),max(sig_corr_LIF),100).T,np.zeros(100),'--k')
        plt.plot(np.linspace(-2.5,2.5,100).T,np.zeros(100),'--k')
        plt.hold('on')
        plt.xlabel('Signal Correlation')
        plt.ylabel('Change in Firing Rate')
#        plt.xlim([-0.4,0.4])
        plt.ylim([-4,4])
#        plt.title('LIF')
#        plt.text(0.25, 0.5,("Beta = {0:.2f}".format(round(slope_LIF,2))) , fontsize=12)
        
        plt.subplot(2,3,1+m)
        plt.errorbar(binned_mid_sig_corr_Gibbs, binned_mean_influence_Gibbs, err_Gibbs,c=colors,fmt='--o',capsize=0.25)
        plt.hold('on')
        plt.scatter(sig_corr_Gibbs,mean_influence_Gibbs,s=10, color = colors1)
        plt.hold('on')
#        plt.plot(np.zeros(100),np.linspace(min(mean_influence_Gibbs),max(mean_influence_Gibbs),100).T,'--k')
        plt.plot(np.zeros(100),np.linspace(-0.2,0.2,100).T,'--k')
        plt.hold('on')
#        plt.plot(np.linspace(min(sig_corr_Gibbs),max(sig_corr_Gibbs),100).T,np.zeros(100),'--k')
        plt.plot(np.linspace(-0.4,0.4,100).T,np.zeros(100),'--k')
        plt.hold('on')
        plt.xlabel('Signal Correlation')
        plt.ylabel('Change in Firing Rate')
#        plt.xlim([-0.4,0.4])
#        plt.ylim([-0.2,0.2])
#        plt.title('Sampling')
#        plt.text(0.25, 0.5,("Beta = {0:.2f}".format(round(slope_Gibbs,2))) , fontsize=12)
        
#        plt.suptitle([str(neurons) + '_' + model])
        plt.show()