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
from LIF_Functions.LIF_with_BRIAN import LIFSamplingModel


colors1 = (np.array(['greenyellow','lawngreen','limegreen','forestgreen','green','darkgreen']))
colors2 = (np.array(['dodgerblue','deepskyblue','cornflowerblue','royalblue','blue','darkblue']))
colors3 = (np.array(['lightcoral','indianred','red','firebrick','maroon','darkred']))

#Runs the initial script to initialize values
path = '../Results/'
model_set = ['NN', 'IN','ISN']
neurons_set = [128]#,64]
contrasts = np.array([0.25, 0.5, 0.75, 1.0])
pix = 64
angles = np.load('../Data/Grating_angles.npy')
phases = np.load('../Data/Grating_phases.npy')
#test_indx = [0,5,10,15,20]
#angles = angles[test_indx]
#phases = phases[test_indx]
sf_case = 1
num_angles = len(angles)
num_ph = len(phases)

#%%
#for nn in range(len(neurons_set)):
#    for m in range(len(model_set)):
#        if m==0:
#            colors = colors1
#        elif m==1:
#            colors = colors2
#        elif m==2:
#            colors = colors3
#        if neurons_set[nn]==128:
#            rows = 16
#        elif neurons_set[nn]==16: 
#            rows = 8
#        
#        plt.figure()
#        for n in range(neurons_set[nn]):
#            plt.subplot(rows,8,n+1)    
#            for i in range(len(contrasts)):
#                filename1 = "SamplingTuningFR_" + "Con=" + str(contrasts[i]) + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '_sf' + str(sf_case) + '.npy'
#                fr_sampling = np.load(path + filename1)
#                y_smp = np.mean(np.sum(np.sum(np.squeeze(fr_sampling[:,:,:,n,:]),3),2),0)
#                plt.plot(angles,y_smp,'-o', color = colors[i])
##                plt.ylim([0,150])
#        plt.legend(contrasts)
#        plt.suptitle('Sampling based tuning for ' + model_set[m] + ' for ' + str(neurons_set[nn]) + ' neurons')
#        
#        plt.figure()
#        for n in range(neurons_set[nn]):
#            plt.subplot(rows,8,n+1)    
#            for i in range(len(contrasts)):
#                filename2 = "LIFTuningFR_" + "Con=" + str(contrasts[i]) + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '_sf' + str(sf_case) + '.npy'
#                fr_LIF = np.load(path + filename2)
#                y_lif = np.mean(np.sum(np.sum(np.squeeze(fr_LIF[:,:,:,n,:]),3),2),0)
#                plt.plot(angles,y_lif,'-o', color = colors[i])
##                plt.ylim([0,200])
#        plt.suptitle('LIF based tuning for ' + model_set[m] + ' for ' + str(neurons_set[nn]) + ' neurons')

#%%
selected = np.array([37*np.array([1, 1, 1]), 3*np.array([1, 1, 1])])#37,112,114,108,17,35,76,92,95,104   4,8,11,12,18,26,69      14,15,16,43,60,62,115,125
lm = 100
for nn in range(len(neurons_set)):
    plt.figure()
    chosen_angle_sampling = np.zeros((len(model_set), neurons_set[nn]))
    chosen_angle_lif = np.zeros((len(model_set), neurons_set[nn]))
    for m in range(len(model_set)):
#        if m<2:
#            path = '../Results/'
#        else:
#            path = '../Results_0.9/'
        selected_neuron = selected[nn,m]
        params = load_simulation_parameters(neurons_set[nn],model_set[m])
        if m==0:
            colors = colors1
            lm = 80
        elif m==1:
            colors = colors2
            lm = 80
        elif m==2:
            colors = colors3
            lm = 80
        
        plt.subplot(len(model_set),2,2*m+1)
        for i in range(len(contrasts)):
            filename1 = "SamplingTuningFR_" + "Con=" + str(contrasts[i]) + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '_sf' + str(sf_case) + '.npy'
            fr_sampling = np.load(path + filename1)
            reps = np.shape(fr_sampling)[0]
            y_smp = np.mean(np.sum(np.sum(np.squeeze(fr_sampling[:,:,:,selected_neuron,:]),3),2),0)
#            er_sampling = np.var(np.sum(np.sum(np.squeeze(fr_sampling[:,:,:,selected_neuron,:]),3),2),0)
            er_sampling = np.std(np.sum(np.sum(np.squeeze(fr_sampling[:,:,:,selected_neuron,:]),3),2),0)/np.sqrt(reps)
            er_sampling_low = np.percentile(np.sum(np.sum(np.squeeze(fr_sampling[:,:,:,selected_neuron,:]),3),2),50,0)-np.percentile(np.sum(np.sum(np.squeeze(fr_sampling[:,:,:,selected_neuron,:]),3),2),16,0)
            er_sampling_high = np.percentile(np.sum(np.sum(np.squeeze(fr_sampling[:,:,:,selected_neuron,:]),3),2),84,0)-np.percentile(np.sum(np.sum(np.squeeze(fr_sampling[:,:,:,selected_neuron,:]),3),2),50,0)
            plt.errorbar(angles,y_smp, marker =".",yerr=(er_sampling_low,er_sampling_high),fmt='-o',color=colors[i], capsize = 3, markersize=5,linewidth = 2,label = contrasts[i])
#            plt.errorbar(angles,y_smp, marker =".",yerr=er_sampling,fmt='-o',color=colors[i], capsize = 3, markersize=1,linewidth = 2, label = contrasts[i])
#            plt.fill_between(angles, y_smp-er_sampling_low, y_smp+er_sampling_high, facecolor=colors[i], alpha=0.5, edgecolor='none')
            plt.ylim([0,lm])
            plt.xlim([0,np.pi])
#            plt.legend(contrasts,loc="upper right",fontsize=15)
            for neu in range(params.N):
                chosen_angle_sampling[m,neu] = np.argmax(np.mean(np.sum(np.sum(np.squeeze(fr_sampling[:,:,:,neu,:]),3),2),0))
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['left'].set_linewidth(2)
            plt.gca().spines['bottom'].set_linewidth(2)
            plt.yticks([0,20,40,60, 80],fontsize=15,fontweight='bold')
#            plt.xticks([])
            plt.xticks(fontsize=15,fontweight='bold',)
        plt.vlines(angles[int(chosen_angle_sampling[m,selected_neuron])],0.0,lm,linestyles ="dotted", linewidth = 2, colors ="k")
#        plt.ylabel('Firing Rate',fontsize=15,fontweight='bold')
#        plt.xlabel('Orientation',fontsize=15,fontweight='bold')
#        plt.suptitle('Sampling based tuning for ' + model_set[m] + ' for ' + str(neurons_set[nn]) + ' neurons')
        
        plt.subplot(len(model_set),2,2*m+2)
        for i in range(len(contrasts)):
            filename2 = "LIFTuningFR_" + "Con=" + str(contrasts[i]) + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '_sf' + str(sf_case) + '.npy'
            fr_LIF = np.load(path + filename2)
            reps = np.shape(fr_LIF)[0]
            y_lif = np.mean(np.sum(np.sum(np.squeeze(fr_LIF[:,:,:,selected_neuron,:]),3),2),0)
#            er_lif = np.var(np.sum(np.sum(np.squeeze(fr_LIF[:,:,:,selected_neuron,:]),3),2),0)
            er_lif = np.std(np.sum(np.sum(np.squeeze(fr_LIF[:,:,:,selected_neuron,:]),3),2),0)/np.sqrt(reps)
            er_lif_low = np.percentile(np.sum(np.sum(np.squeeze(fr_LIF[:,:,:,selected_neuron,:]),3),2),50,0)-np.percentile(np.sum(np.sum(np.squeeze(fr_LIF[:,:,:,selected_neuron,:]),3),2),16,0)
            er_lif_high = np.percentile(np.sum(np.sum(np.squeeze(fr_LIF[:,:,:,selected_neuron,:]),3),2),84,0)-np.percentile(np.sum(np.sum(np.squeeze(fr_LIF[:,:,:,selected_neuron,:]),3),2),50,0)
            plt.errorbar(angles,y_lif, marker =".", yerr=(er_lif_low,er_lif_high),fmt='-o',color=colors[i], capsize = 3, markersize=5,linewidth = 2,label = contrasts[i])
#            plt.errorbar(angles,y_lif, marker =".", yerr=er_lif,fmt='-o',color=colors[i], capsize = 3, markersize=1,linewidth = 2, label = contrasts[i])
#            plt.fill_between(angles, y_lif-er_lif, y_lif+er_lif, facecolor=colors[i], alpha=0.5, edgecolor='none')
            plt.ylim([0,lm])
            plt.xlim([0,np.pi])
#            plt.legend(contrasts,loc="upper right",fontsize=15)
            for neu in range(params.N):
                chosen_angle_lif[m,neu] = np.argmax(np.mean(np.sum(np.sum(np.squeeze(fr_LIF[:,:,:,neu,:]),3),2),0))          
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['left'].set_linewidth(2)
            plt.gca().spines['bottom'].set_linewidth(2)
            plt.yticks([0,20,40,60, 80],fontsize=15,fontweight='bold')
#            plt.xticks([])
            plt.xticks(fontsize=15,fontweight='bold')
        plt.vlines(angles[int(chosen_angle_lif[m,selected_neuron])],0.0,lm,linestyles ="dotted", linewidth = 2, colors ="k")
#        plt.ylabel('Firing Rate',fontsize=15,fontweight='bold')
#        plt.xlabel('Orientation',fontsize=15,fontweight='bold')
#        plt.suptitle('Sampling and LIF based tuning for ' + model_set[m] + ' for ' + str(neurons_set[nn]) + ' neurons')

#%%
GratingImage_data = np.load('../Data/Gratings.npy')   
for nn in range(len(neurons_set)):
    plt.figure()
    params = load_simulation_parameters(neurons_set[nn],'NN')
    plt.subplot(1,2,1)
    selected_neuron = selected[nn,0]
    gr_im = GratingImage_data[sf_case-1,int(chosen_angle_sampling[0,selected_neuron]),20,:]
    plt.imshow(np.reshape(params.G[:,selected_neuron],(8,8)), cmap = 'gray')
    plt.axis('off')
    plt.title("Chosen PF")
    plt.subplot(1,2,2)
    selected_neuron = selected[nn,0]
    plt.imshow(np.reshape(gr_im,(8,8)), cmap = 'gray')
    plt.axis('off')
    plt.title("Grating causing max FR")
        


#%%
#sf_case = 1
#ang1 = 15
#ang2 = 5
#for nn in range(len(neurons_set)):
#    plt.figure()
#    for m in range(len(model_set)):
##        if m<2:
##            path = '../Results/'
##        else:
##            path = '../Results_0.9/'
#        params = load_simulation_parameters(neurons_set[nn],model_set[m])
#        weights_smp = np.zeros((len(contrasts),params.N))
#        weights_lif = np.zeros((len(contrasts),params.N))
#        if m==0:
#            colors = colors1
#        elif m==1:
#            colors = colors2
#        elif m==2:
#            colors = colors3
#        
#        plt.subplot(len(model_set),2,2*m+1)
#        for i in range(len(contrasts)):
#            filename1 = "SamplingTuningFR_" + "Con=" + str(contrasts[i]) + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '_sf' + str(sf_case) + '.npy'
#            fr_sampling = np.load(path + filename1)
#            reps = np.shape(fr_sampling)[0]
#            num_angles = np.shape(fr_sampling)[1]
#            y_smp = np.mean(np.sum(np.sum(fr_sampling[:,:,:,:,:],4),2),0)
#            y_sm_diff_coarse = y_smp[ang1,:] - y_smp[ang2,:]
#            temp_all = np.reshape(np.sum(np.sum(fr_sampling[:,:,:,:,:],4),2),(reps*num_angles,neurons_set[nn]))
#            C_sampling = np.cov(np.transpose(temp_all))
#            
#            weights_smp[i,:] = np.matmul(np.linalg.pinv(np.squeeze(C_sampling)),y_sm_diff_coarse)
##            plt.scatter(range(params.N),np.sort(weights_smp[i,:]), color=colors[i],  label = contrasts[i])
#            plt.scatter(np.sort(weights_smp[0,:]),np.sort(weights_smp[i,:]),color=colors[i], s=5, label = contrasts[i])
#            plt.gca().spines['right'].set_visible(False)
#            plt.gca().spines['top'].set_visible(False)
#            plt.yticks([-1,-0.5,0.0,0.5,1],fontsize=12)
#            plt.xticks([-1,-0.5,0.0,0.5,1],fontsize=12)
#            plt.yticks(fontsize=12)
#            plt.xticks(fontsize=12)
#            plt.xlabel('Weights for contrast = 0.25',fontsize=12)
#            plt.ylabel('Weights for \nother contrasts',fontsize=12)
#        plt.legend(contrasts,loc="upper left")
#        plt.plot(np.linspace(-1,1,10),np.linspace(-1,1,10),'k')
#        plt.plot(np.zeros(10),np.linspace(-1,1,10),'k')
#        plt.plot(np.linspace(-1,1,10),np.zeros(10),'k')
#        plt.xlim([-1,1])
#        plt.ylim([-0.5,0.5])
##        plt.suptitle('Sampling based coarse discrimination weights \nfor ' + model_set[m] + ' for ' + str(neurons_set[nn]) + ' neurons')
#        
#        plt.subplot(len(model_set),2,2*m+2)
#        for i in range(len(contrasts)):
#            filename2 = "LIFTuningFR_" + "Con=" + str(contrasts[i]) + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '_sf' + str(sf_case) + '.npy'
#            fr_LIF = np.load(path + filename2)
#            reps = np.shape(fr_LIF)[0]
#            num_angles = np.shape(fr_sampling)[1]
#            y_lif = np.mean(np.sum(np.sum(np.squeeze(fr_LIF[:,:,:,:,:]),4),2),0)
#            y_lif_diff_coarse = y_lif[ang1,:] - y_lif[ang2,:]
#            temp_all = np.reshape(np.sum(np.sum(fr_LIF[:,:,:,:,:],4),2),(reps*num_angles,neurons_set[nn]))
#            C_lif = np.cov(np.transpose(temp_all))
#            
#            weights_lif[i,:] = np.matmul(np.linalg.pinv(np.squeeze(C_lif)),y_lif_diff_coarse)
##            plt.scatter(range(params.N),np.sort(weights_lif[i,:]), color=colors[i],  label = contrasts[i])
#
#            plt.scatter(np.sort(weights_lif[0,:]),np.sort(weights_lif[i,:]),color=colors[i], s=5, label = contrasts[i])
#            plt.gca().spines['right'].set_visible(False)
#            plt.gca().spines['top'].set_visible(False)
#            plt.yticks([-1,-0.5,0.0,0.5,1],fontsize=12)
#            plt.xticks([-1,-0.5,0.0,0.5,1],fontsize=12)
#            plt.yticks(fontsize=12)
#            plt.xticks(fontsize=12)
#            plt.xlabel('Weights for contrast = 0.25',fontsize=12)
#            plt.ylabel('Weights for \nother contrasts',fontsize=12)
#        plt.legend(contrasts,loc="upper left")
#        plt.plot(np.linspace(-1,1,10),np.linspace(-1,1,10),'k')
#        plt.plot(np.zeros(10),np.linspace(-1,1,10),'k')
#        plt.plot(np.linspace(-1,1,10),np.zeros(10),'k')
#        plt.xlim([-1,1])
#        plt.ylim([-0.5,0.5])
#        plt.suptitle('Sampling and LIF based coarse discrimination weights \nfor ' + model_set[m] + ' for ' + str(neurons_set[nn]) + ' neurons')
#
##%%
#sf_case = 1
#ang1 = 15
#ang2 = 18
#for nn in range(len(neurons_set)):
#    plt.figure()
#    for m in range(len(model_set)):
##        if m<2:
##            path = '../Results/'
##        else:
##            path = '../Results_0.9/'
#        params = load_simulation_parameters(neurons_set[nn],model_set[m])
#        weights_smp = np.zeros((len(contrasts),params.N))
#        weights_lif = np.zeros((len(contrasts),params.N))
#        if m==0:
#            colors = colors1
#        elif m==1:
#            colors = colors2
#        elif m==2:
#            colors = colors3
#        
#        plt.subplot(len(model_set),2,2*m+1)
#        for i in range(len(contrasts)):
#            filename1 = "SamplingTuningFR_" + "Con=" + str(contrasts[i]) + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '_sf' + str(sf_case) + '.npy'
#            fr_sampling = np.load(path + filename1)
#            reps = np.shape(fr_sampling)[0]
#            num_angles = np.shape(fr_sampling)[1]
#            y_smp = np.mean(np.sum(np.sum(fr_sampling[:,:,:,:,:],4),2),0)
#            y_sm_diff_coarse = y_smp[ang1,:] - y_smp[ang2,:]
##            y_sm_diff_fine[i,:] = y_smp[0,:] - y_smp[2,:]
#            temp_all = np.reshape(np.sum(np.sum(fr_sampling[:,:,:,:,:],4),2),(reps*num_angles,neurons_set[nn]))
#            C_sampling = np.cov(np.transpose(temp_all))
#            
#            weights_smp[i,:] = np.matmul(np.linalg.pinv(np.squeeze(C_sampling)),y_sm_diff_coarse)
##            plt.scatter(range(params.N),np.sort(weights_smp[i,:]), color=colors[i],  label = contrasts[i])
#            plt.scatter(np.sort(weights_smp[0,:]),np.sort(weights_smp[i,:]),color=colors[i], s=5, label = contrasts[i])
#            plt.gca().spines['right'].set_visible(False)
#            plt.gca().spines['top'].set_visible(False)
#            plt.yticks([-1,-0.5,0.0,0.5,1],fontsize=12)
#            plt.xticks([-1,-0.5,0.0,0.5,1],fontsize=12)
#            plt.xlabel('Weights for contrast = 0.25',fontsize=12)
#            plt.ylabel('Weights for \nother contrasts',fontsize=12)
#        plt.legend(contrasts,loc="upper left")
#        plt.plot(np.linspace(-1,1,10),np.linspace(-1,1,10),'k')
#        plt.plot(np.zeros(10),np.linspace(-1,1,10),'k')
#        plt.plot(np.linspace(-1,1,10),np.zeros(10),'k')
#        plt.xlim([-1,1])
#        plt.ylim([-0.5,0.5])
##        plt.suptitle('Sampling based fine discrimination weights \nfor ' + model_set[m] + ' for ' + str(neurons_set[nn]) + ' neurons')
#        
#        plt.subplot(len(model_set),2,2*m+2)
#        for i in range(len(contrasts)):
#            filename2 = "LIFTuningFR_" + "Con=" + str(contrasts[i]) + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '_sf' + str(sf_case) + '.npy'
#            fr_LIF = np.load(path + filename2)
#            reps = np.shape(fr_LIF)[0]
#            num_angles = np.shape(fr_sampling)[1]
#            y_lif = np.mean(np.sum(np.sum(np.squeeze(fr_LIF[:,:,:,:,:]),4),2),0)
#            y_lif_diff_fine = y_lif[ang1,:] - y_lif[ang2,:]
#            temp_all = np.reshape(np.sum(np.sum(fr_LIF[:,:,:,:,:],4),2),(reps*num_angles,neurons_set[nn]))
#            C_lif = np.cov(np.transpose(temp_all))
#            
#            weights_lif[i,:] = np.matmul(np.linalg.pinv(np.squeeze(C_lif)),y_lif_diff_fine)
##            plt.scatter(range(params.N),np.sort(weights_lif[i,:]), color=colors[i],  label = contrasts[i])
#            
#            plt.scatter(np.sort(weights_lif[0,:]),np.sort(weights_lif[i,:]),color=colors[i], s=5, label = contrasts[i])
#            plt.gca().spines['right'].set_visible(False)
#            plt.gca().spines['top'].set_visible(False)
#            plt.yticks([-1,-0.5,0.0,0.5,1],fontsize=12)
#            plt.xticks([-1,-0.5,0.0,0.5,1],fontsize=12)
#            plt.xlabel('Weights for contrast = 0.25',fontsize=12)
#            plt.ylabel('Weights for \nother contrasts',fontsize=12)
#        plt.legend(contrasts,loc="upper left")
#        plt.plot(np.linspace(-1,1,10),np.linspace(-1,1,10),'k')
#        plt.plot(np.zeros(10),np.linspace(-1,1,10),'k')
#        plt.plot(np.linspace(-1,1,10),np.zeros(10),'k')
#        plt.xlim([-1,1])
#        plt.ylim([-0.5,0.5])
#        plt.suptitle('Sampling and LIF based fine discrimination weights \nfor ' + model_set[m] + ' for ' + str(neurons_set[nn]) + ' neurons')
#
##%%
#GratingImage_data = np.load('../Data/Gratings.npy')  
#plt.figure() 
#for i in range(np.shape(GratingImage_data)[1]):
#    plt.subplot(1,np.shape(GratingImage_data)[1],i+1)
#    gr_im = GratingImage_data[sf_case-1,i,0,:]
#    plt.imshow(np.reshape(gr_im,(8,8)), cmap = 'gray')
#    plt.axis('off')
#
#
