#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 13:32:07 2020

@author: achattoraj
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 12:05:23 2019

@author: achattoraj
"""
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
from LIF_Functions.Pairwise_binned_spikes_LIF import *
from LIF_Functions.Marg_binned_spikes_LIF import *

path = '../Results/'
model_set = ['NN', 'IN','ISN']
neurons_set = [128]
pix = 64
im_selected = np.array(range(5))
sf_case = 2
contrast = 1.0
colors1 = np.array(['greenyellow','limegreen','darkgreen'])
colors2 = np.array(['dodgerblue','cornflowerblue','darkblue'])
colors3 = np.array(['lightcoral','red','darkred'])


for nn in range(len(neurons_set)):
    plt.figure()
    for m in range(len(model_set)):
        if m==0:
            colors = 'green'
            colors_set = colors1
        elif m==1:
            colors = 'blue'
            colors_set = colors2
        elif m==2:
            colors = 'red' 
            colors_set = colors3
            
        neurons = neurons_set[nn]
        model = model_set[m]
        params = load_simulation_parameters(neurons,model)
        sample_hertz = 1.0/(params.sampling_bin_s)
        plt.subplot(2,3,1+m)
        for im in range(len(im_selected)):
            Sampling_marginal_prob = np.load(path + 'Sampling_marginal_probNaturalImages' + '_' + str(neurons) + '_' + model + '.npy')
            sampling_p_marg = Sampling_marginal_prob[im_selected[im],:]
            
            LIF_marginal_prob = np.load(path + 'LIF_marginal_probNaturalImages' + '_' + str(neurons) + '_' + model + '.npy')
            implied_p_marg = LIF_marginal_prob[im_selected[im],:]
            
            plt.scatter(sampling_p_marg, implied_p_marg, s=20, color=colors)#, edgecolors='k')
            plt.plot(np.linspace(0,1 ,100),np.linspace(0,1 ,100),'k',linewidth=2)
            plt.xlim(-0.1,1.1)
            plt.ylim(-0.1,1.1)
            plt.xlabel("Sample marginal",fontsize=15)
            plt.ylabel("Implied marginal",fontsize=15)
            plt.hold('on')
            plt.xticks([0.0,0.5,1.0],fontsize=15,fontweight='bold') 
            plt.yticks([0.0,0.5,1.0],fontsize=15,fontweight='bold') 
#            plt.axis('tight')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        plt.gca().set_aspect('equal')
#        plt.subplot(3,3,4+m)
#        for im in range(len(im_selected)):
#            Sampling_pairwise_joint_prob = np.load(path + 'Sampling_pairwise_joint_probNaturalImages' + '_' + str(neurons) + '_' + model + '.npy')
#            sampling_p_pairwise = Sampling_pairwise_joint_prob[im_selected[im],:]
#            
#            LIF_pairwise_joint_prob = np.load(path + 'LIF_pairwise_joint_probNaturalImages' + '_' + str(neurons) + '_' + model + '.npy')            
#            implied_p_pairwise = LIF_pairwise_joint_prob[im_selected[im],:]
#            
#            
#            plt.scatter(sampling_p_pairwise, implied_p_pairwise, s=50, color=colors, edgecolors='k')
#            plt.plot(np.linspace(0,1,100),np.linspace(0,1,100),'k')
#            plt.xlim(-0.1,1.1)
#            plt.ylim(-0.1,1.1)
#            plt.xlabel("Sample pairwise joint",fontsize=10)
#            plt.ylabel("Implied pairwise joint",fontsize=10)
#            plt.xticks([0.0,0.5,1.0]) 
#            plt.xticks(fontsize=10)
#            plt.yticks([0.0,0.5,1.0]) 
#            plt.yticks(fontsize=10)
#            plt.hold('on')
##            plt.axis('tight')
#        plt.gca().spines['right'].set_visible(False)
#        plt.gca().spines['top'].set_visible(False)  
        
        plt.subplot(2,3,4+m)
        for im in range(len(im_selected)):   
            Sampling_pairwise_joint_prob00 = np.load(path + 'Sampling_pairwise_joint_prob00NaturalImages' + '_' + str(neurons) + '_' + model + '.npy')
            sampling_p_pairwise00 = Sampling_pairwise_joint_prob00[im_selected[im],:]
            Sampling_pairwise_joint_prob01 = np.load(path + 'Sampling_pairwise_joint_prob01NaturalImages' + '_' + str(neurons) + '_' + model + '.npy')
            sampling_p_pairwise01 = Sampling_pairwise_joint_prob01[im_selected[im],:]
            Sampling_pairwise_joint_prob10 = np.load(path + 'Sampling_pairwise_joint_prob10NaturalImages' + '_' + str(neurons) + '_' + model + '.npy')
            sampling_p_pairwise10 = Sampling_pairwise_joint_prob10[im_selected[im],:]
            Sampling_pairwise_joint_prob11 = np.load(path + 'Sampling_pairwise_joint_prob11NaturalImages' + '_' + str(neurons) + '_' + model + '.npy')
            sampling_p_pairwise11 = Sampling_pairwise_joint_prob11[im_selected[im],:]
            
            
            LIF_pairwise_joint_prob00 = np.load(path + 'LIF_pairwise_joint_prob00NaturalImages' + '_' + str(neurons) + '_' + model + '.npy')            
            implied_p_pairwise00 = LIF_pairwise_joint_prob00[im_selected[im],:]
            LIF_pairwise_joint_prob01 = np.load(path + 'LIF_pairwise_joint_prob01NaturalImages' + '_' + str(neurons) + '_' + model + '.npy')            
            implied_p_pairwise01 = LIF_pairwise_joint_prob01[im_selected[im],:]
            LIF_pairwise_joint_prob10 = np.load(path + 'LIF_pairwise_joint_prob10NaturalImages' + '_' + str(neurons) + '_' + model + '.npy')            
            implied_p_pairwise10 = LIF_pairwise_joint_prob10[im_selected[im],:]
            LIF_pairwise_joint_prob11 = np.load(path + 'LIF_pairwise_joint_prob11NaturalImages' + '_' + str(neurons) + '_' + model + '.npy')            
            implied_p_pairwise11 = LIF_pairwise_joint_prob11[im_selected[im],:]

            if im==len(im_selected)-1:
                plt.scatter(sampling_p_pairwise00, implied_p_pairwise00, s=20, color='w',edgecolors=colors_set[0],label='00')#, edgecolors='k')
                plt.scatter(sampling_p_pairwise01, implied_p_pairwise01, s=20, color=colors_set[1],label='01 or 10')#, edgecolors='k')
                plt.scatter(sampling_p_pairwise10, implied_p_pairwise10, s=20, color=colors_set[1])#, edgecolors='k')
                plt.scatter(sampling_p_pairwise11, implied_p_pairwise11, s=20, color=colors_set[2],label='11')#, edgecolors='k')
            
                plt.plot(np.linspace(0,1 ,100),np.linspace(0,1,100),'k',linewidth=2)
                plt.xlim(-0.1,1.1)
                plt.ylim(-0.1,1.1)
                plt.xlabel("Sample pairwise joint",fontsize=15)
                plt.ylabel("Implied pairwise joint",fontsize=15)
                plt.hold('on')
                plt.xticks([0.0,0.5,1.0],fontsize=15,fontweight='bold') 
                plt.yticks([0.0,0.5,1.0],fontsize=15,fontweight='bold') 
#                plt.axis('tight')
                
                plt.legend(loc="lower right")

            else:
                plt.scatter(sampling_p_pairwise00, implied_p_pairwise00, s=20, color='w',edgecolors=colors_set[0])#, edgecolors='k')
                plt.scatter(sampling_p_pairwise01, implied_p_pairwise01, s=20, color=colors_set[1])#, edgecolors='k')
                plt.scatter(sampling_p_pairwise10, implied_p_pairwise10, s=20, color=colors_set[1])#, edgecolors='k')
                plt.scatter(sampling_p_pairwise11, implied_p_pairwise11, s=20, color=colors_set[2])#, edgecolors='k')
            
                plt.plot(np.linspace(0,1 ,100),np.linspace(0,1 ,100),'k',linewidth=2)
                plt.xlim(-0.1,1.1)
                plt.ylim(-0.1,1.1)
                plt.xlabel("Sample pairwise joint",fontsize=10)
                plt.ylabel("Implied pairwise joint",fontsize=10)
                plt.hold('on')
                plt.xticks([0.0,0.5,1.0],fontsize=15,fontweight='bold') 
                plt.yticks([0.0,0.5,1.0],fontsize=15,fontweight='bold') 
#                plt.axis('tight')
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['bottom'].set_linewidth(2)
            plt.gca().spines['left'].set_linewidth(2)
            plt.gca().set_aspect('equal')
            
#        plt.subplot(4,3,10+m)
#        for im in range(len(im_selected)):
#            Sampling_pairwise_diff = np.load(path + 'Sampling_pairwise_joint_diffprobNaturalImages' + '_' + str(neurons) + '_' + model + '.npy')
#            sampling_p_pairwise_diff = Sampling_pairwise_diff[im_selected[im],:]
#        
#            LIF_pairwise_diff = np.load(path + 'LIF_pairwise_joint_diffprobNaturalImages' + '_' + str(neurons) + '_' + model + '.npy')            
#            implied_p_pairwise_diff = LIF_pairwise_diff[im_selected[im],:]
#        
#        
#            plt.scatter(sampling_p_pairwise_diff, implied_p_pairwise_diff, s=50, color=colors, edgecolors='k')
##            plt.plot(np.linspace(-5,15,100),np.linspace(-5,15,100),'k')
#            plt.plot(np.linspace(-0.2,0.2,100),np.linspace(-0.2,0.2,100),'k')
#
##                plt.xlim(-0.1,1.1)
##                plt.ylim(-0.1,1.1)
#            plt.xlabel("Sample pairwise joint diff",fontsize=10)
#            plt.ylabel("Implied pairwise joint diff",fontsize=10)
##                plt.xticks([0.0,0.5,1.0]) 
#            plt.xticks(fontsize=10)
##                plt.yticks([0.0,0.5,1.0]) 
#            plt.yticks(fontsize=10)
#            plt.hold('on')
##            plt.axis('tight')
#        plt.gca().spines['right'].set_visible(False)
#        plt.gca().spines['top'].set_visible(False)  
#    plt.suptitle("LIF-Sampling Match for Natural Images for " + str(neurons) + " neurons")


#%%
for nn in range(len(neurons_set)):
    plt.figure()
    for m in range(len(model_set)):
        if m==0:
            colors = 'green'
            colors_set = colors1
        elif m==1:
            colors = 'blue'
            colors_set = colors2
        elif m==2:
            colors = 'red' 
            colors_set = colors3
            
        neurons = neurons_set[nn]
        model = model_set[m]
        params = load_simulation_parameters(neurons,model)
        sample_hertz = 1.0/(params.sampling_bin_s)
        
        plt.subplot(2,3,1+m)
        for im in range(len(im_selected)):
            Sampling_pairwise_diff = np.load(path + 'Sampling_pairwise_joint_diffprobNaturalImages' + '_' + str(neurons) + '_' + model + '.npy')
            sampling_p_pairwise_diff = Sampling_pairwise_diff[im_selected[im],:]
        
            LIF_pairwise_diff = np.load(path + 'LIF_pairwise_joint_diffprobNaturalImages' + '_' + str(neurons) + '_' + model + '.npy')            
            implied_p_pairwise_diff = LIF_pairwise_diff[im_selected[im],:]
        
        
            plt.scatter(sampling_p_pairwise_diff, implied_p_pairwise_diff, s=20, color=colors)#, edgecolors='k')
#            plt.plot(np.linspace(-5,15,100),np.linspace(-5,15,100),'k')
            plt.plot(np.linspace(-0.1,0.1,100),np.linspace(-0.1,0.1,100),'k',linewidth=2)

            plt.xlim([-0.1,0.1])
            plt.ylim(-0.1,0.1)
            plt.xlabel("Sample pairwise joint and marg diff",fontsize=15)
            plt.ylabel("Implied pairwise joint and marg diff",fontsize=15)
#                plt.xticks([0.0,0.5,1.0]) 
            plt.xticks(fontsize=15, fontweight='bold')
#                plt.yticks([0.0,0.5,1.0]) 
            plt.yticks(fontsize=15, fontweight='bold')
            plt.hold('on')
#            plt.axis('tight')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        plt.gca().set_aspect('equal')
        
        plt.subplot(2,3,4+m)
        for im in range(len(im_selected)):
            Sampling_pairwise_diff = np.load(path + 'Sampling_pairwise_joint_logdiffprobNaturalImages' + '_' + str(neurons) + '_' + model + '.npy')
            sampling_p_pairwise_diff = Sampling_pairwise_diff[im_selected[im],:]
        
            LIF_pairwise_diff = np.load(path + 'LIF_pairwise_joint_logdiffprobNaturalImages' + '_' + str(neurons) + '_' + model + '.npy')            
            implied_p_pairwise_diff = LIF_pairwise_diff[im_selected[im],:]
        
        
            plt.scatter(sampling_p_pairwise_diff, implied_p_pairwise_diff, s=20, color=colors)#, edgecolors='k')
#            plt.plot(np.linspace(-5,15,100),np.linspace(-5,15,100),'k')
            plt.plot(np.linspace(-3,5,100),np.linspace(-3,5,100),'k',linewidth=2)

            plt.xlim([-3,5])
            plt.ylim([-3,5])
            plt.xlabel("Sample log pairwise joint and log marg diff",fontsize=15)
            plt.ylabel("Implied log pairwise joint and log marg diff",fontsize=15)
#                plt.xticks([0.0,0.5,1.0]) 
            plt.xticks(fontsize=15,fontweight='bold')
#                plt.yticks([0.0,0.5,1.0]) 
            plt.yticks(fontsize=15,fontweight='bold')
            plt.hold('on')
#            plt.axis('tight')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        plt.gca().set_aspect('equal')
#    plt.suptitle("LIF-Sampling Match for Natural Images for " + str(neurons) + " neurons")

#%%
for nn in range(len(neurons_set)):
    plt.figure()
    for m in range(len(model_set)):
        if m==0:
            colors = 'green'
            colors_set = colors1
        elif m==1:
            colors = 'blue'
            colors_set = colors2
        elif m==2:
            colors = 'red' 
            colors_set = colors3
            
        neurons = neurons_set[nn]
        model = model_set[m]
        params = load_simulation_parameters(neurons,model)
        sample_hertz = 1.0/(params.sampling_bin_s)
        
        plt.subplot(2,3,1+m)
        for im in range(len(im_selected)):
            Sampling_pairwise_diff = np.load(path + 'Sampling_pairwise_joint_diffprobNaturalImages' + '_' + str(neurons) + '_' + model + '.npy')
            sampling_p_pairwise_diff = Sampling_pairwise_diff[im_selected[im],:]
        
            LIF_pairwise_diff = np.load(path + 'LIF_pairwise_joint_diffprobNaturalImages' + '_' + str(neurons) + '_' + model + '.npy')            
            implied_p_pairwise_diff = LIF_pairwise_diff[im_selected[im],:]
        
        
            plt.scatter(sampling_p_pairwise_diff, implied_p_pairwise_diff, s=15, color=colors)#, edgecolors='k')
#            plt.plot(np.linspace(-5,15,100),np.linspace(-5,15,100),'k')
            plt.plot(np.linspace(-0.1,0.1,100),np.linspace(-0.1,0.1,100),'k',linewidth=2)
            if m==0:
                plt.xlim([-0.1,0.1])
                plt.ylim(-0.1,0.1)
            else:
                plt.xlim([-0.04,0.04])
                plt.ylim(-0.04,0.04)
            plt.xlabel("Sample pairwise and marg diff",fontsize=15)
            plt.ylabel("Implied pairwise and marg diff",fontsize=15)
#                plt.xticks([0.0,0.5,1.0]) 
            plt.xticks(fontsize=15,fontweight='bold')
#                plt.yticks([0.0,0.5,1.0]) 
            plt.yticks(fontsize=15,fontweight='bold')
            plt.hold('on')
#            plt.axis('tight')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        plt.gca().set_aspect('equal')
        
        plt.subplot(2,3,4+m)
        for im in range(len(im_selected)):
            Sampling_pairwise_diff = np.load(path + 'Sampling_pairwise_joint_diffprobboundedNaturalImages' + '_' + str(neurons) + '_' + model + '.npy')
            sampling_p_pairwise_diff = Sampling_pairwise_diff[im_selected[im],:]
        
            LIF_pairwise_diff = np.load(path + 'LIF_pairwise_joint_diffprobboundedNaturalImages' + '_' + str(neurons) + '_' + model + '.npy')            
            implied_p_pairwise_diff = LIF_pairwise_diff[im_selected[im],:]
        
        
            plt.scatter(sampling_p_pairwise_diff, implied_p_pairwise_diff, s=15, color=colors)#, edgecolors='k')
#            plt.plot(np.linspace(-5,15,100),np.linspace(-5,15,100),'k')
            plt.plot(np.linspace(-3,5,100),np.linspace(-3,5,100),'k',linewidth=2)

            if m==0:
                plt.xlim([-0.1,0.1])
                plt.ylim(-0.1,0.1)
            else:
                plt.xlim([-0.04,0.04])
                plt.ylim(-0.04,0.04)
                
            plt.xlabel("Sample pairwise and marg diff (marg>=0.05)",fontsize=10)
            plt.ylabel("Implied pairwise and marg diff (marg>=0.05)",fontsize=10)
#                plt.xticks([0.0,0.5,1.0]) 
            plt.xticks(fontsize=15,fontweight='bold')
#                plt.yticks([0.0,0.5,1.0]) 
            plt.yticks(fontsize=15,fontweight='bold')
            plt.hold('on')
#            plt.axis('tight')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        plt.gca().set_aspect('equal')
#    plt.suptitle("LIF-Sampling Match for Natural Images for " + str(neurons) + " neurons")
        










#%%
#    plt.figure()
#    for m in range(len(model_set)):
#        
#        if m==0:
#            colors = 'green'
#        elif m==1:
#            colors = 'blue'
#        elif m==2:
#            colors = 'red'  
#            
#        neurons = neurons_set[nn]
#        model = model_set[m]
#        params = load_simulation_parameters(neurons,model)
#        sample_hertz = 1.0/(params.sampling_bin_s)
#        
#        Sampling_marginal_prob = np.load(path + 'Sampling_marginal_probGratingImages' + str(sf_case) + '_' + str(contrast) + str(neurons) + '_' + model + '.npy')
#        sampling_p_marg = Sampling_marginal_prob[im_selected,:]
#        Sampling_pairwise_joint_prob = np.load(path + 'Sampling_pairwise_joint_probGratingImages' + str(sf_case) + '_' + str(contrast) + str(neurons) + '_' + model + '.npy')
#        sampling_p_pairwise = Sampling_pairwise_joint_prob[im_selected,:]
#        
#        LIF_marginal_prob = np.load(path + 'LIF_marginal_probGratingImages' + str(sf_case) + '_' + str(contrast) + str(neurons) + '_' + model + '.npy')
#        implied_p_marg = LIF_marginal_prob[im_selected,:]
#        LIF_pairwise_joint_prob = np.load(path + 'LIF_pairwise_joint_probGratingImages' + str(sf_case) + '_' + str(contrast) + str(neurons) + '_' + model + '.npy')            
#        implied_p_pairwise = LIF_pairwise_joint_prob[im_selected,:]
#
#        plt.subplot(2,3,4+m)
#        plt.scatter(sampling_p_pairwise, implied_p_pairwise, s=50, color=colors, edgecolors='k')
#        plt.plot(np.linspace(0,1 + 0.1,100),np.linspace(0,1 + 0.1,100),'k')
#        plt.xlim(-0.01,1.01)
#        plt.ylim(-0.01,1.01)
#        plt.xlabel("Sample pairwise joint for Grating Images")
#        plt.ylabel("Implied pairwise joint for Grating Images")
#        plt.axis('tight')
#        
#        subplot(2,3,1+m)
#        plt.scatter(sampling_p_marg, implied_p_marg, s=50, color=colors, edgecolors='k')
#        plt.plot(np.linspace(0,1 + 0.1,100),np.linspace(0,1 + 0.1,100),'k')
#        plt.xlim(-0.01,1.01)
#        plt.ylim(-0.01,1.01)
#        plt.xlabel("Sample marginal for Grating Images")
#        plt.ylabel("Implied marginal for Grating Images")
#        plt.axis('tight')
#    plt.suptitle("LIF-Sampling Match for Grating Images for " + str(neurons) + " neurons")
#   
