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
from Sampling_Functions.Pairwise_joint_samples import *

from LIF_Functions.LIF_spikes_from_BRIAN import *
from LIF_Functions.LIF_with_BRIAN import LIFSamplingModel

#colors = np.array(['green','blue','red'])
colors1 = (np.array(['greenyellow','lawngreen','limegreen','forestgreen','green','darkgreen']))
colors2 = (np.array(['dodgerblue','deepskyblue','cornflowerblue','royalblue','blue','darkblue']))
colors3 = (np.array(['lightcoral','indianred','red','firebrick','maroon','darkred']))

## Initialize variables    
path =  '../Results/'
path1 =  '../Results1/'
model_set = ['NN','IN','ISN']
neurons_set = [128]
pix = 64 #image pixels
dimension = 8 #sqrt(image pixels)
simulation_cases = ['NaturalImage']#,'GratingImage']#['NaturalImage']#,
#sf_case = 1
#colors1 = np.array(['greenyellow','limegreen','darkgreen'])
#colors2 = np.array(['dodgerblue','cornflowerblue','darkblue'])
#colors3 = np.array(['lightcoral','red','darkred'])

bins = 500
im_chosen = 10
neuron_chosen_set = 15*np.array([[1,1,1],[1,1,1]])
image_chosen_set = 0*np.array([[1,1,1],[1,1,1]])
contrasts = np.array([0.3, 0.4, 0.5, 0.75, 1.0])#np.array([0.25,0.3, 0.35,0.4,0.45, 0.5, 0.75, 1.0])

for sm in range(len(simulation_cases)):
    for nn in range(len(neurons_set)):
        neurons = neurons_set[nn]
        for m in range(len(model_set)):
            model = model_set[m]
            if m==0:
                colors = colors1
            elif m==1:
                colors = colors2
            elif m==2:
                colors = colors3

            if simulation_cases[sm]=='NaturalImage':
                Sampling_CV_SingleNeuron = np.load(path1 + 'SamplingStats_CV_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy')
                Sampling_FF_SingleNeuron = np.load(path1 + 'SamplingStats_FF_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy')            
                Sampling_Corr_SingleNeuron = np.load(path1 + 'SamplingStats_Corr_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy')
                Sampling_SigCorr_SingleNeuron = np.load(path1 + 'SamplingStats_SigCorr_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy')
                Sampling_ISI_SingleNeuron = np.load(path1 + 'SamplingStats_ISI_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy')
                
                Spiking_CV_SingleNeuron = np.load(path1 + 'LIFStats_CV_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy')
                Spiking_FF_SingleNeuron = np.load(path1 + 'LIFStats_FF_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy')            
                Spiking_Corr_SingleNeuron= np.load(path1 + 'LIFStats_Corr_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy')
                Spiking_SigCorr_SingleNeuron = np.load(path1 + 'LIFStats_SigCorr_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy')
                Spiking_ISI_SingleNeuron = np.load(path1 + 'LIFStats_ISI_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy')
            
#                
                Sampling_CV = np.load(path + 'SamplingStats_CV_NaturalImages_' + str(neurons) + '_' + model + '.npy')
                Sampling_FF = np.load(path + 'SamplingStats_FF_NaturalImages_' + str(neurons) + '_' + model + '.npy')            
                Sampling_Corr = np.load(path + 'SamplingStats_Corr_NaturalImages_' + str(neurons) + '_' + model + '.npy')
                Sampling_SigCorr = np.load(path + 'SamplingStats_SigCorr_NaturalImages_' + str(neurons) + '_' + model + '.npy')
                Sampling_ISI = np.load(path + 'SamplingStats_ISI_NaturalImages_' + str(neurons) + '_' + model + '.npy')
                
                Spiking_CV = np.load(path + 'LIFStats_CV_NaturalImages_' + str(neurons) + '_' + model + '.npy')
                Spiking_FF = np.load(path + 'LIFStats_FF_NaturalImages_' + str(neurons) + '_' + model + '.npy')            
                Spiking_Corr = np.load(path + 'LIFStats_Corr_NaturalImages_' + str(neurons) + '_' + model + '.npy')
                Spiking_SigCorr = np.load(path + 'LIFStats_SigCorr_NaturalImages_' + str(neurons) + '_' + model + '.npy')
                Spiking_ISI = np.load(path + 'LIFStats_ISI_NaturalImages_' + str(neurons) + '_' + model + '.npy')
            else:
                chosen_contrast = 1
                Sampling_CV = np.load(path + 'SamplingStats_CV_GratingImages_' + str(neurons) + '_' + model + '.npy')
                Sampling_FF = np.load(path + 'SamplingStats_FF_GratingImages_' + str(neurons) + '_' + model + '.npy')            
                Sampling_Corr = np.load(path + 'SamplingStats_Corr_GratingImages_' + str(neurons) + '_' + model + '.npy')
                Sampling_SigCorr = np.load(path + 'SamplingStats_SigCorr_GratingImages_' + str(neurons) + '_' + model + '.npy')
                Sampling_ISI = np.load(path + 'SamplingStats_ISI_GratingImages_' + str(neurons) + '_' + model + '.npy')
                
                Sampling_CV = np.squeeze(Sampling_CV[chosen_contrast,:,:])
                Sampling_FF = np.squeeze(Sampling_FF[chosen_contrast,:,:])
                Sampling_Corr = np.squeeze(Sampling_Corr[chosen_contrast,:,:,:])
                Sampling_SigCorr = np.squeeze(Sampling_SigCorr[chosen_contrast,:,:,:])
                Sampling_ISI = np.squeeze(Sampling_ISI[chosen_contrast,:,:,:,:])
                
                Spiking_CV = np.load(path + 'LIFStats_CV_GratingImages_' + str(neurons) + '_' + model + '.npy')
                Spiking_FF = np.load(path + 'LIFStats_FF_GratingImages_' + str(neurons) + '_' + model + '.npy')            
                Spiking_Corr = np.load(path + 'LIFStats_Corr_GratingImages_' + str(neurons) + '_' + model + '.npy')  
                Spiking_SigCorr = np.load(path + 'LIFStats_SigCorr_GratingImages_' + str(neurons) + '_' + model + '.npy')
                Spiking_ISI = np.load(path + 'LIFStats_ISI_GratingImages_' + str(neurons) + '_' + model + '.npy')
                
                Spiking_CV = np.squeeze(Spiking_CV[chosen_contrast,:,:])
                Spiking_FF = np.squeeze(Spiking_FF[chosen_contrast,:,:])
                Spiking_Corr = np.squeeze(Spiking_Corr[chosen_contrast,:,:,:])
                Spiking_SigCorr = np.squeeze(Spiking_SigCorr[chosen_contrast,:,:,:])
                Spiking_ISI = np.squeeze(Spiking_ISI[chosen_contrast,:,:,:,:])
    
            neuron_chosen = neuron_chosen_set[nn,m]
            image_chosen = 3#image_chosen_set[nn,m]  3      5 6
            if m<2:
                rep = 4
            else:
                rep = 11   #8
            
            sampled_ISI = np.squeeze(Sampling_ISI_SingleNeuron[image_chosen,rep,neuron_chosen,:])
            sampled_ISI = sampled_ISI[np.logical_not((sampled_ISI<0.0))]
            sampled_CV = np.squeeze(Sampling_CV_SingleNeuron[:,neuron_chosen])
            sampled_CV = sampled_CV[np.logical_not(np.isnan(sampled_CV))]
            sampled_CV = sampled_CV[np.logical_not((sampled_CV<=0.0))]
            sampled_CV_all = Sampling_CV.flatten() #Sampling_CV.flatten()
            sampled_CV_all = sampled_CV_all[np.logical_not(np.isnan(sampled_CV_all))]
            sampled_CV_all = sampled_CV_all[np.logical_not((sampled_CV_all<=0.0))]
            sampled_FF = np.squeeze(Sampling_FF_SingleNeuron[:,neuron_chosen])
            sampled_FF = sampled_FF[np.logical_not(np.isnan(sampled_FF))]
            sampled_FF = sampled_FF[np.logical_not((sampled_FF<=0.0))]
            sampled_FF_all = Sampling_FF.flatten() #Sampling_FF.flatten()
            sampled_FF_all = sampled_FF_all[np.logical_not(np.isnan(sampled_FF_all))]
            sampled_FF_all = sampled_FF_all[np.logical_not((sampled_FF_all<=0.0))]
            sampled_Corr = np.squeeze(Sampling_Corr_SingleNeuron[im_chosen,neuron_chosen,:])
            sampled_Corr = sampled_Corr[np.logical_not(np.isnan(sampled_Corr))]
            sampled_Corr = sampled_Corr[np.logical_not((np.abs(sampled_Corr)==1.0))]
            sampled_Corr = sampled_Corr[np.logical_not((np.abs(sampled_Corr)==100))]
            sampled_Corr = sampled_Corr[np.logical_not((np.abs(sampled_Corr)>0.9))]
            sampled_Corr_all = Sampling_Corr.flatten()
            sampled_SigCorr_all = Sampling_SigCorr.flatten() 
            sampled_SigCorr_all = sampled_SigCorr_all[np.logical_not(np.isnan(sampled_Corr_all))]
            sampled_Corr_all = sampled_Corr_all[np.logical_not(np.isnan(sampled_Corr_all))]
            sampled_SigCorr_all = sampled_SigCorr_all[np.logical_not((np.abs(sampled_Corr_all)==1.0))]
            sampled_Corr_all = sampled_Corr_all[np.logical_not((np.abs(sampled_Corr_all)==1.0))]
            sampled_SigCorr_all = sampled_SigCorr_all[np.logical_not((np.abs(sampled_Corr_all)==100))]
            sampled_Corr_all = sampled_Corr_all[np.logical_not((np.abs(sampled_Corr_all)==100))]
#            sampled_Corr_all = sampled_Corr_all[np.logical_not((np.abs(sampled_Corr_all)>0.9))]
            
            image_chosen1 = 3       #6 (0,10,0)
            if m==0:
                rep1 = 0
            if m==1:
                rep1 = 18
            else:
                rep1 = 1
            LIF_ISI = np.squeeze(Spiking_ISI_SingleNeuron[image_chosen1,rep1,neuron_chosen,:])
            LIF_ISI = LIF_ISI[np.logical_not((LIF_ISI<0.0))]
            LIF_CV = np.squeeze(Spiking_CV_SingleNeuron[image_chosen,:])#neuron_chosen])
            LIF_CV = LIF_CV[np.logical_not(np.isnan(LIF_CV))]
            LIF_CV = LIF_CV[np.logical_not((LIF_CV<=0.0))]
            LIF_CV_all = Spiking_CV.flatten()
            LIF_CV_all = LIF_CV_all[np.logical_not(np.isnan(LIF_CV_all))]
            LIF_CV_all = LIF_CV_all[np.logical_not((LIF_CV_all<=0.0))]
            LIF_FF = np.squeeze(Spiking_FF_SingleNeuron[:,neuron_chosen])
            LIF_FF = LIF_FF[np.logical_not(np.isnan(LIF_FF))]
            LIF_FF = LIF_FF[np.logical_not((LIF_FF<=0.0))]
            LIF_FF_all = Spiking_FF.flatten()
            LIF_FF_all = LIF_FF_all[np.logical_not(np.isnan(LIF_FF_all))]
            LIF_FF_all = LIF_FF_all[np.logical_not((LIF_FF_all<=0.0))]
            LIF_Corr = np.squeeze(Spiking_Corr_SingleNeuron[im_chosen,neuron_chosen,:])
            LIF_Corr = LIF_Corr[np.logical_not(np.isnan(LIF_Corr))]
            LIF_Corr = LIF_Corr[np.logical_not((np.abs(LIF_Corr)==1.0))]
            LIF_Corr = LIF_Corr[np.logical_not((np.abs(LIF_Corr)==100))]
#            LIF_Corr = LIF_Corr[np.logical_not((np.abs(LIF_Corr)>0.9))]
            LIF_Corr_all = Spiking_Corr.flatten()
            LIF_SigCorr_all = Spiking_SigCorr.flatten()
            LIF_SigCorr_all = LIF_SigCorr_all[np.logical_not(np.isnan(LIF_Corr_all))]
            LIF_Corr_all = LIF_Corr_all[np.logical_not(np.isnan(LIF_Corr_all))]
            LIF_SigCorr_all = LIF_SigCorr_all[np.logical_not((np.abs(LIF_Corr_all)==1.0))]
            LIF_Corr_all = LIF_Corr_all[np.logical_not((np.abs(LIF_Corr_all)==1.0))]
            LIF_SigCorr_all = LIF_SigCorr_all[np.logical_not((np.abs(LIF_Corr_all)==100))]
            LIF_Corr_all = LIF_Corr_all[np.logical_not((np.abs(LIF_Corr_all)==100))]
#            LIF_Corr_all = LIF_Corr_all[np.logical_not((np.abs(LIF_Corr_all)>0.9))]
            
#            print('Number of elements for Sampling FF for model ' + model_set[m] +  str(np.shape(sampled_FF_all)))
#            print('Number of elements for Sampling CV for model ' + model_set[m] +  str(np.shape(sampled_CV_all)))
#            print('Number of elements for Spiking FF for model ' + model_set[m] +  str(np.shape(LIF_FF_all)))
#            print('Number of elements for Spiking CV for model ' + model_set[m] +  str(np.shape(LIF_CV_all)))
            
#            kwargs = dict(histtype='stepfilled', alpha=0.3, normed = True,bins=5, ec="k", color=colors[0])
#            kwargs1 = dict(histtype='stepfilled', alpha=0.3, normed = True,bins=25, ec="k", color=colors[0])
            
            plt.figure(nn*7+1)
            plt.subplot(3,4,4*m+1)

            sns.distplot(sampled_ISI[sampled_ISI>0], hist=True, kde=True, norm_hist=True, 
            color = colors[5],bins=15, 
            hist_kws={'color': colors[5],'range': [3, 50],"histtype": "stepfilled", 'density':True,'alpha':0.15},
            kde_kws={'linewidth': 3,'bw':2.,'clip': [4, 50]})
            
            if m==2:
                plt.xlabel('ISI of one neuron for one image', fontsize = 15)
            plt.xticks([0,25,50], fontsize = 18, fontweight='bold')
            plt.xlim(0, 50)
            plt.yticks([])
#            if m<2:
#                plt.ylim([0,0.])
#            else:
#                plt.ylim([0,0.1])
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['bottom'].set_linewidth(2)
            plt.gca().spines['left'].set_linewidth(2)
            plt.gca().spines['bottom'].set_color('black')
            plt.gca().spines['left'].set_color('black')
            
            plt.subplot(3,4,4*m+2)
            sns.set_style("whitegrid", {'axes.grid' : False})
            sns.distplot(sampled_CV, hist=True, kde=True, norm_hist=True,  
            color = colors[5], bins = 14,
            hist_kws={'color': colors[3],'range': [0.05, 3.75],'alpha':0.15},#,'edgecolor':'black'},
            kde_kws={'linewidth': 3,'bw':0.26,'clip': [0.05, 3.75]})
            if m==2:
                plt.xlabel('CV of one neuron across all images', fontsize = 15)
            plt.xticks([0.0, 1.0, 2.0, 3.0, 4.0], fontsize = 18, fontweight='bold')
            plt.xlim(0,4)
            plt.plot(np.mean(sampled_CV),0.0,'o',markersize=12,markeredgecolor='black',color=colors[3],label=str(np.round(np.mean(sampled_CV), 2)))
            plt.yticks([])
            if m==0:
                plt.ylim([0,0.8])
            if m==1:
                plt.ylim([0,1.5])
            if m==2:
                plt.ylim([0,1.5])
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['bottom'].set_linewidth(2)
            plt.gca().spines['left'].set_linewidth(2)
            plt.gca().spines['bottom'].set_color('black')
            plt.gca().spines['left'].set_color('black')
            plt.legend(fontsize=18)
           
            plt.subplot(3,4,4*m+3)
            sns.set_style("whitegrid", {'axes.grid' : False})
            sns.distplot(sampled_FF, hist=True, kde=True, norm_hist=True,  
            color = colors[5], bins=14,
            hist_kws={'color': colors[3],'range': [0.1, 3.75],'alpha':0.15},#,'edgecolor':'black'},
            kde_kws={'linewidth': 3,'bw':0.26,'clip': [0.1, 3.7]})
            if m==2:
                plt.xlabel('FF of one neuron across all images', fontsize = 15)
            plt.xticks([0.0, 1.0, 2.0, 3.0, 4.0], fontsize = 18, fontweight='bold')
            plt.xlim(0,4)
            plt.plot(np.mean(sampled_FF[sampled_FF<=4]),0.0,'o',markersize=12,markeredgecolor='black',color=colors[3],label=str(np.round(np.mean(sampled_FF[sampled_FF<=4]), 2)))
            plt.yticks([])
            if m==0:
                plt.ylim([0,0.8])
            if m==1:
                plt.ylim([0,0.8])
            if m==2:
                plt.ylim([0,1.5])
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['bottom'].set_linewidth(2)
            plt.gca().spines['left'].set_linewidth(2)
            plt.gca().spines['bottom'].set_color('black')
            plt.gca().spines['left'].set_color('black')
            plt.legend(fontsize=18)
            
            plt.subplot(3,4,4*m+4)
            sns.set_style("whitegrid", {'axes.grid' : False})
            sns.distplot(sampled_Corr, hist=True, kde=True, norm_hist=True,  
            color = colors[5], 
            hist_kws={'color': colors[3],'range': [-0.5, 0.5],'alpha':0.15},#,'edgecolor':'black'},
            kde_kws={'linewidth': 3,'clip': [-0.5, 0.5]})
            if m==2:
                plt.xlabel('Corr of one neuron with other neurons for one image', fontsize = 15)
            plt.xticks([-0.5, 0.0, 0.5], fontsize = 18, fontweight='bold')
            plt.xlim(-0.6,0.6)
            plt.plot(np.mean(sampled_Corr),0.0,'o',markersize=12,markeredgecolor='black',color=colors[3],label=str(np.round(np.mean(sampled_Corr), 3)))
            plt.yticks([])
            plt.ylim([0,3])
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['bottom'].set_linewidth(2)
            plt.gca().spines['left'].set_linewidth(2)
            plt.gca().spines['bottom'].set_color('black')
            plt.gca().spines['left'].set_color('black')
            plt.legend(fontsize=18)
            
            plt.suptitle("Sampling based single neuron stats for " + str(neurons) + " neurons")
            
            
            
            plt.figure(nn*7+2)
            plt.subplot(3,4,4*m+1)

            sns.distplot(LIF_ISI[LIF_ISI>0], hist=True, kde=True, norm_hist=True, 
            color = colors[5],bins=20,
            hist_kws={'color': colors[5],'range': [3, 100],"histtype": "stepfilled", 'density':True,'alpha':0.15},
            kde_kws={'linewidth': 3,'bw':5,'clip': [4, 100]})
            
            if m==2:
                plt.xlabel('ISI of one neuron for one image', fontsize = 15)
            plt.xticks([0,50,100], fontsize = 18, fontweight='bold')
            plt.xlim(0, 100)
            plt.yticks([])
#            if m<2:
#                plt.ylim([0,0.])
#            else:
#                plt.ylim([0,0.1])
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['bottom'].set_linewidth(2)
            plt.gca().spines['left'].set_linewidth(2)
            plt.gca().spines['bottom'].set_color('black')
            plt.gca().spines['left'].set_color('black')
            
            plt.subplot(3,4,4*m+2)
            sns.set_style("whitegrid", {'axes.grid' : False})
            sns.distplot(LIF_CV, hist=True, kde=True, norm_hist=True,  
            color = colors[5], bins = 20,
            hist_kws={'color': colors[3],'range': [0.1, 2],'alpha':0.15},#,'edgecolor':'black'},
            kde_kws={'linewidth': 3,'bw':0.09,'clip': [0.1, 2]})
            if m==2:
                plt.xlabel('CV of one neuron across all images', fontsize = 15)
            plt.xticks([0.0, 1.0, 2.0], fontsize = 18, fontweight='bold')
            plt.xlim(0,2)
            plt.plot(np.mean(LIF_CV),0.0,'o',markersize=12,markeredgecolor='black',color=colors[3],label=str(np.round(np.mean(LIF_CV), 1)))
            plt.yticks([])
#            if m==0:
#                plt.ylim([0,0.8])
#            if m==1:
#                plt.ylim([0,1.5])
#            if m==2:
#                plt.ylim([0,1.5])
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['bottom'].set_linewidth(2)
            plt.gca().spines['left'].set_linewidth(2)
            plt.gca().spines['bottom'].set_color('black')
            plt.gca().spines['left'].set_color('black')
            plt.legend(fontsize=18)
           
            plt.subplot(3,4,4*m+3)
            sns.set_style("whitegrid", {'axes.grid' : False})
            sns.distplot(LIF_FF, hist=True, kde=True, norm_hist=True,  
            color = colors[5], bins=20,
            hist_kws={'color': colors[3],'range': [0.1, 2],'alpha':0.15},#,'edgecolor':'black'},
            kde_kws={'linewidth': 3,'bw':0.09,'clip': [0.1, 2]})
            if m==2:
                plt.xlabel('FF of one neuron across all images', fontsize = 15)
            plt.xticks([0.0, 1.0, 2.0], fontsize = 18, fontweight='bold')
            plt.xlim(0,2)
            plt.plot(np.mean(LIF_FF),0.0,'o',markersize=12,markeredgecolor='black',color=colors[3],label=str(np.round(np.mean(LIF_FF), 1)))
            plt.yticks([])
#            if m==0:
#                plt.ylim([0,0.8])
#            if m==1:
#                plt.ylim([0,0.8])
#            if m==2:
#                plt.ylim([0,1.5])
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['bottom'].set_linewidth(2)
            plt.gca().spines['left'].set_linewidth(2)
            plt.gca().spines['bottom'].set_color('black')
            plt.gca().spines['left'].set_color('black')
            plt.legend(fontsize=18)
            
            plt.subplot(3,4,4*m+4)
            sns.set_style("whitegrid", {'axes.grid' : False})
            sns.distplot(LIF_Corr, hist=True, kde=True, norm_hist=True,  
            color = colors[5], 
            hist_kws={'color': colors[3],'range': [-0.5, 0.5],'alpha':0.15},#,'edgecolor':'black'},
            kde_kws={'linewidth': 3,'clip': [-0.5, 0.5]})
            if m==2:
                plt.xlabel('Corr of one neuron with other neurons for one image', fontsize = 15)
            plt.xticks([-0.5, 0.0, 0.5], fontsize = 18, fontweight='bold')
            plt.xlim(-0.6,0.6)
            plt.plot(np.mean(LIF_Corr),0.0,'o',markersize=12,markeredgecolor='black',color=colors[3],label=str(np.round(np.mean(LIF_Corr), 3)))
            plt.yticks([])
            plt.ylim([0,3])
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['bottom'].set_linewidth(2)
            plt.gca().spines['left'].set_linewidth(2)
            plt.gca().spines['bottom'].set_color('black')
            plt.gca().spines['left'].set_color('black')
            plt.legend(fontsize=18)
            
            plt.suptitle("LIF based single neuron stats for " + str(neurons) + " neurons")
#            
            
#            plt.figure(nn*7+3)
#            plt.subplot(3,4,4*m+1)
##            plt.hist(sampled_ISI[sampled_ISI>0],**kwargs1)
##            sns.set_style("whitegrid", {'axes.grid' : False})
##            sns.kdeplot(data=sampled_ISI, shade=True, color=colors[3], kernel='epa',bw=20, cut = 0, clip = [0, 400])
#            sns.set_style("whitegrid", {'axes.grid' : False})
#            sns.distplot(LIF_ISI[LIF_ISI>0], hist=True, kde=True, norm_hist=True,  
#            color = colors[5],bins=7500, 
#            hist_kws={'color': colors[3],'range': [0.05, 400],'alpha':0.1},#,'edgecolor':'black'},
#            kde_kws={'linewidth': 3,'bw':0.053,'clip': [0.15, 400]})
#            plt.xlabel('ISI of one neuron for one image')
#            plt.xticks([0, 100, 200, 300, 400], fontsize = 15, fontweight='bold')
#            plt.xlim(0, 400)
#            plt.yticks([])
#            
#            plt.subplot(3,4,4*m+2)
##            plt.hist(sampled_CV,**kwargs)
##            sns.set_style("whitegrid", {'axes.grid' : False})
##            sns.distplot(sampled_CV, fit=norm, kde=False)#bins=None,kde=True, norm_hist=True, color='gray')
##            sns.kdeplot(sampled_CV, shade=True, color=colors[3], kernel='triw', bw=0.3,clip = [0.2, 3])
#            sns.set_style("whitegrid", {'axes.grid' : False})
#            sns.distplot(LIF_CV, hist=True, kde=True, norm_hist=True,  
#            color = colors[5],bins=75, 
#            hist_kws={'color': colors[3],'range': [0.05, 4],'alpha':0.1},#,'edgecolor':'black'},
#            kde_kws={'linewidth': 3,'bw':0.053,'clip': [0.15, 4]})
#            plt.xlabel('CV of one neuron across all images')
#            plt.xticks([0.0, 1.0, 2.0, 3.0, 4.0], fontsize = 15, fontweight='bold')
#            plt.xlim(0,4)
#            plt.plot(np.mean(LIF_CV),0.0,'o',markersize=12,markeredgecolor='black',color=colors[3])
#            plt.yticks([])
#           
#            plt.subplot(3,4,4*m+3)
##            plt.hist(sampled_FF[sampled_FF<3], **kwargs)
##            sns.set_style("whitegrid", {'axes.grid' : False})
##            sns.distplot(plot_3,  fit=norm, kde=False)#bins=None,kde=True, norm_hist=True, color='gray')
##            sns.kdeplot(sampled_FF[sampled_FF<4], shade=True, color=colors[3], kernel='triw', bw=0.3,clip = [0.2, 3])
#            sns.set_style("whitegrid", {'axes.grid' : False})
#            sns.distplot(LIF_FF[LIF_FF<=4], hist=True, kde=True, norm_hist=True,  
#            color = colors[5],bins=75, 
#            hist_kws={'color': colors[3],'range': [0.05, 4],'alpha':0.1},#,'edgecolor':'black'},
#            kde_kws={'linewidth': 3,'bw':0.053,'clip': [0.15, 4]})
#            plt.xlabel('FF of one neuron across all images')
#            plt.xticks([0.0, 1.0, 2.0, 3.0, 4.0], fontsize = 15, fontweight='bold')
#            plt.xlim(0,4)
#            plt.plot(np.mean(LIF_FF[LIF_FF<=4]),0.0,'o',markersize=12,markeredgecolor='black',color=colors[3])
#            plt.yticks([])
#            
#            plt.subplot(3,4,4*m+4)
##            plt.hist(sampled_Corr, **kwargs)
##            sns.set_style("whitegrid", {'axes.grid' : False})
##            sns.distplot(plot_4,  fit=norm, kde=False)#bins=None,kde=True, norm_hist=True, color='gray')
##            sns.kdeplot(sampled_Corr, sh!ade=True, color=colors[3], kernel='gau', bw = 0.1,clip = [-0.2, 0.2])
#            sns.set_style("whitegrid", {'axes.grid' : False})
#            sns.distplot(LIF_Corr, hist=True, kde=True, norm_hist=True,  
#            color = colors[5],bins=40, 
#            hist_kws={'color': colors[3],'range': [-0.5, 0.5],'alpha':0.1},#,'edgecolor':'black'},
#            kde_kws={'linewidth': 3,'bw':0.05,'clip': [-0.5, 0.5]})
#            plt.xlabel('Corr of one neuron with other neurons for one image')
#            plt.xticks([-0.5, 0.0, 0.5], fontsize = 15, fontweight='bold')
#            plt.xlim(-0.5,0.5)
#            plt.plot(np.mean(LIF_Corr),0.0,'o',markersize=12,markeredgecolor='black',color=colors[3])
#            plt.yticks([])
#            
#            plt.suptitle("LIF based single neuron stats for " + str(neurons) + " neurons")
#            
            
#            plt.figure(nn*7+4)
#            plt.subplot(3,4,4*m+1)
#            plt.hist(LIF_ISI[LIF_ISI>0],**kwargs)
#    #        sns.set_style("whitegrid", {'axes.grid' : False})
##            sns.kdeplot(data=LIF_ISI[LIF_ISI>0], shade=True, color=colors[3],bw=8,clip = [5, 50])
#            plt.xlabel('ISI of one neuron for one image')
#            plt.xlim(0, 100)
#            plt.yticks([])
#            
#            plt.subplot(3,4,4*m+2)
#            plt.hist(LIF_CV,**kwargs)
##            sns.set_style("whitegrid", {'axes.grid' : False})
##            sns.distplot(plot_1, fit=norm, kde=False)#bins=None,kde=True, norm_hist=True, color='gray')
#            sns.kdeplot(LIF_CV, shade=True, color=colors[3], kernel='triw', bw=0.12,clip = [0.1, 4])
#            plt.xlabel('CV of one neuron across all images')
#            plt.xlim(0,1.5)
#    #        plt.plot(sp.stats.mode(plot_1)[0][0],0.0,'o',color=colors[3])
#            plt.plot(np.mean(LIF_CV),0.1,'o',color=colors[3])
#            plt.yticks([])
#           
#            plt.subplot(3,4,4*m+3)
#            plt.hist(LIF_FF, **kwargs1)
##            sns.set_style("whitegrid", {'axes.grid' : False})
##            sns.distplot(LIF_FF,  fit=norm, kde=False)#bins=None,kde=True, norm_hist=True, color='gray')
#            sns.kdeplot(LIF_FF, shade=True, color=colors[3], kernel='triw', bw=0.12,clip = [0.1, 4])
#            plt.xlabel('FF of one neuron across all images')
#            plt.xlim(0,1.5)
#    #        plt.plot(sp.stats.mode(plot_3)[0][0],0.0,'o',color=colors[3])
#            plt.plot(np.mean(LIF_FF),0.1,'o',color=colors[3])
#            plt.yticks([])
#            
#            plt.subplot(3,4,4*m+4)
#            plt.hist(LIF_Corr, **kwargs1)
##            sns.set_style("whitegrid", {'axes.grid' : False})
##            sns.distplot(plot_4,  fit=norm, kde=False)#bins=None,kde=True, norm_hist=True, color='gray')
#            sns.kdeplot(LIF_Corr, shade=True, color=colors[3], kernel='gau', bw = 0.1,clip = [-0.2, 0.2])
#            plt.xlabel('Corr of one neuron with other neurons for one image')
#            plt.xlim(-0.5, 0.5)
#    #        plt.plot(sp.stats.mode(plot_4)[0][0],0.0,'o',color=colors[3])
#            plt.plot(np.mean(LIF_Corr),0.1,'o',color=colors[3])
#            plt.yticks([])
#            
#            plt.suptitle("LIF based single neuron stats for " + str(neurons) + " neurons")
#            
            
            
            plt.figure(nn*7+5)
            plt.subplot(3,2,2*m+1)
            sns.set_style("whitegrid", {'axes.grid' : False})
            plt.scatter(sampled_SigCorr_all,sampled_Corr_all,color=colors[3],alpha=0.1)#,edgecolors= "black")
            plt.xlabel('Signal correlation',fontsize=15)
            plt.ylabel('Sampling based noise correlation',fontsize=15)
            corr = np.corrcoef(sampled_SigCorr_all,sampled_Corr_all)[0][1]
            slp, b = np.polyfit(sampled_SigCorr_all,sampled_Corr_all, 1)
            plt.vlines(0.0,-1.5,1.5,'k',linewidth=2,linestyle='-')
            plt.hlines(0.0,-1.5,1.5,'k',linewidth=2,linestyle='-')
#            plt.xlim(0.0,2)
#            plt.xticks([0.0,1.0,2.0]) 
            plt.xticks(fontsize=15,fontweight='bold')
            plt.yticks(fontsize=15,fontweight='bold')
#            plt.ylim(-0.03,5.5)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.text(1.0, 1.0, 'Slope = '+str(np.round(slp,2)), fontsize=15)
            plt.text(1.0, 0.75, 'Corr = '+str(np.round(corr,2)), fontsize=15)
            
            plt.subplot(3,2,2*m+2)
            sns.set_style("whitegrid", {'axes.grid' : False})
            plt.scatter(LIF_SigCorr_all,LIF_Corr_all,color=colors[3],alpha=0.1)#,edgecolors= "black")
            plt.xlabel('Signal correlation',fontsize=15)
            plt.ylabel('LIF based noise correlation',fontsize=15)
            corr = np.corrcoef(LIF_SigCorr_all,LIF_Corr_all)[0][1]
            slp, b = np.polyfit(LIF_SigCorr_all,LIF_Corr_all, 1)
            plt.vlines(0.0,-1.5,1.5,'k',linewidth=2,linestyle='-')
            plt.hlines(0.0,-1.5,1.5,'k',linewidth=2,linestyle='-')
#            plt.xlim(0.0,2)
#            plt.xticks([0.0,1.0,2.0]) 
            plt.xticks(fontsize=15,fontweight='bold')
            plt.yticks(fontsize=15,fontweight='bold')
#            plt.ylim(-0.03,5.5)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.text(1.0, 1.0, 'Slope = '+str(np.round(slp,2)), fontsize=15)
            plt.text(1.0, 0.75, 'Corr = '+str(np.round(corr,2)), fontsize=15)
#            
            
            
#            plt.figure(nn*7+6)
#            for con in range(1,8):
#                chosen_contrast = con
#                Sampling_FF = np.load(path + 'SamplingStats_FF_GratingImages_' + str(neurons) + '_' + model + '.npy')            
#                Sampling_FF = np.squeeze(Sampling_FF[chosen_contrast,:,:])
#                sampled_FF_all = Sampling_FF.flatten() #Sampling_FF.flatten()
#                sampled_FF_all = sampled_FF_all[np.logical_not(np.isnan(sampled_FF_all))]
#                sampled_FF_all = sampled_FF_all[np.logical_not((sampled_FF_all<=0.0))]
#                
#            
#                plt.subplot(2,3,m+1)
#                sns.set_style("whitegrid", {'axes.grid' : False})
##                    plt.hist(sampled_FF_all, **kwargs)
#                sns.kdeplot(sampled_FF_all,color=colors[con],kernel='gau',shade=False, bw=0.1, clip = [0.1, 3])
#                plt.xlabel('FF of all neurons across all images',fontsize=15)
#                plt.plot(np.mean(sampled_FF_all[sampled_FF_all<4]),0.0,'o',color=colors[con],label=str(np.round(np.mean(sampled_FF_all[sampled_FF_all<4]), 3)))
#                plt.yticks([])
#                plt.vlines(1.0,0,15.5,'k',linewidth=0.5,linestyle='--')
#                plt.hlines(0.0,0,6,'k',linewidth=1,linestyle='--')
#                plt.xlim(0,4)
#                plt.xticks([0.0,1.0,2.0,3.0, 4.0]) 
#                plt.xticks(fontsize=15)
#                plt.ylim(-0.03,5.5)
#                plt.gca().spines['right'].set_visible(False)
#                plt.gca().spines['left'].set_visible(False)
#                plt.gca().spines['top'].set_visible(False)
#                if con==3:
#                    plt.legend()
#            
#            for con in range(1,8):
#                chosen_contrast = con
#                Spiking_FF = np.load(path + 'LIFStats_FF_GratingImages_' + str(neurons) + '_' + model + '.npy')            
#                Spiking_FF = np.squeeze(Spiking_FF[chosen_contrast,:,:])
#                LIF_FF_all = Spiking_FF.flatten()
#                LIF_FF_all = LIF_FF_all[np.logical_not(np.isnan(LIF_FF_all))]
#                LIF_FF_all = LIF_FF_all[np.logical_not((LIF_FF_all<=0.0))]
#                plt.subplot(2,3,3+m+1)
#                sns.set_style("whitegrid", {'axes.grid' : False})
##                    plt.hist(sampled_FF_all, **kwargs)
#                sns.kdeplot(LIF_FF_all,color=colors[con],kernel='triw',shade=False, bw=0.2, clip = [0.175, 1.5])
#                plt.xlabel('FF of all neurons across all images',fontsize=15)
#                plt.plot(np.mean(LIF_FF_all),0.0,'o',color=colors[con],label=str(np.round(np.mean(LIF_FF_all), 3)))
#                plt.yticks([])
#                plt.vlines(1.0,0,5.5,'k',linewidth=0.5,linestyle='--')
#                plt.hlines(0.0,0,3,'k',linewidth=1,linestyle='--')
#                plt.xlim(0,2)
#                plt.xticks([0.0,1.0,2.0]) 
#                plt.xticks(fontsize=15)
#                plt.ylim(-0.03,5.0)
#                plt.gca().spines['right'].set_visible(False)
#                plt.gca().spines['left'].set_visible(False)
#                plt.gca().spines['top'].set_visible(False)
#                if con==3:
#                    plt.legend()
                    
                    
                    
            plt.figure(nn*7+7)
            mn_smp = np.zeros(len(contrasts))
            err_smp = np.zeros(len(contrasts))
            mn_lif = np.zeros(len(contrasts))
            err_lif = np.zeros(len(contrasts))
            kk = 0
            for con in range(len(contrasts)):
                chosen_contrast = con
                Sampling_FF = np.load(path + 'SamplingStats_FF_GratingImages_' + str(neurons) + '_' + model + '.npy')            
                Sampling_FF = np.squeeze(Sampling_FF[chosen_contrast,:,:])
                sampled_FF_all = Sampling_FF.flatten() #Sampling_FF.flatten()
                sampled_FF_all = sampled_FF_all[np.logical_not(np.isnan(sampled_FF_all))]
                sampled_FF_all = sampled_FF_all[np.logical_not((sampled_FF_all<=0.0))]
                mn_smp[kk] = np.mean(sampled_FF_all[sampled_FF_all<4])
                err_smp[kk] = np.std(sampled_FF_all[sampled_FF_all<4])/np.sqrt(len(sampled_FF_all[sampled_FF_all<4]))
                
                Spiking_FF = np.load(path + 'LIFStats_FF_GratingImages_' + str(neurons) + '_' + model + '.npy')            
                Spiking_FF = np.squeeze(Spiking_FF[chosen_contrast,:,:])
                LIF_FF_all = Spiking_FF.flatten()
                LIF_FF_all = LIF_FF_all[np.logical_not(np.isnan(LIF_FF_all))]
                LIF_FF_all = LIF_FF_all[np.logical_not((LIF_FF_all<=0.0))]
#                if m==2:
#                    mn_lif[kk] = np.mean(LIF_FF_all[LIF_FF_all>0.4])
#                    err_lif[kk] = np.std(LIF_FF_all[LIF_FF_all>0.4])/np.sqrt(np.shape(LIF_FF_all[LIF_FF_all>0.4]))
#                else:
                mn_lif[kk] = np.mean(LIF_FF_all)
                err_lif[kk] = np.var(LIF_FF_all)#np.std(LIF_FF_all)/np.sqrt(len(LIF_FF_all))
                kk = kk + 1
            
#            plt.subplot(1,2,1)
            if m==2:
#                plt.errorbar(contrasts,mn_smp,marker =".",yerr=err_smp,fmt='o',markersize=20,color=colors[3],capsize = 3,linewidth=3,label='Sampling')
#                plt.errorbar(contrasts,mn_lif,marker =".",yerr=err_lif,fmt='o',markersize=20,color=colors[3],capsize = 3,linewidth=3,label='LIF')
                plt.plot(contrasts,mn_smp,'o',markersize=10,color=colors[3],linewidth=3)
                plt.plot(contrasts,mn_lif,'o',markersize=10,color=colors[3],linewidth=3)
                plt.plot(contrasts,mn_smp,'-',color=colors[3],linewidth=3,label='Sampling'+'('+model_set[m]+')')
                plt.plot(contrasts,mn_lif,'--',color=colors[3],linewidth=3,label='LIF'+'('+model_set[m]+')')
            else:
#                plt.errorbar(contrasts,mn_smp,marker =".",yerr=err_smp,fmt='o',markersize=20,color=colors[3],capsize = 3,linewidth=3)
#                plt.errorbar(contrasts,mn_lif,marker =".",yerr=err_lif,fmt='o',markersize=20,color=colors[3],capsize = 3,linewidth=3)
                plt.plot(contrasts,mn_smp,'o',markersize=10,color=colors[3],linewidth=3)
                plt.plot(contrasts,mn_lif,'o',markersize=10,color=colors[3],linewidth=3)
                plt.plot(contrasts,mn_smp,'-',color=colors[3],linewidth=3,label='Sampling'+'('+model_set[m]+')')
                plt.plot(contrasts,mn_lif,'--',color=colors[3],linewidth=3,label='LIF'+'('+model_set[m]+')')
            plt.xlabel('Contrasts',fontsize=15)
            plt.ylabel('Mean FF across all gratings across all neurons for sampling',fontsize=15)
#            plt.yticks([])
            plt.hlines(1.0,0,2,'k',linewidth=1,linestyle='--')
            plt.xlim(0.25,1.1)
            plt.yticks([0.0,0.5,1.0,1.5])
#            plt.xticks(contrasts)
            plt.xticks(fontsize=15,fontweight='bold')
            plt.yticks(fontsize=15,fontweight='bold')
            plt.ylim(0.0,1.5)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['left'].set_linewidth(2)
            plt.gca().spines['bottom'].set_linewidth(2)
            plt.gca().spines['bottom'].set_color('black')
            plt.gca().spines['left'].set_color('black')
            plt.legend(fontsize=18,loc='upper right')
            
#            plt.subplot(1,2,2)
#            plt.errorbar(contrasts,mn_lif,marker =".",yerr=err_lif,fmt='-o',color=colors[3],capsize = 3,linewidth=3)
#            plt.xlabel('Contrasts',fontsize=15)
#            plt.ylabel('Mean FF across all gratings across all neurons for LIF ',fontsize=15)
##            plt.yticks([])
#            plt.hlines(1.0,0,2,'k',linewidth=1,linestyle='--')
#            plt.xlim(0.25,1.1)
#            plt.yticks([0.0,0.5,1.0,1.5])
##            plt.xticks(contrasts)
#            plt.xticks(fontsize=15,fontweight='bold')
#            plt.yticks(fontsize=15,fontweight='bold')
##            plt.ylim(0.0,1.5)
#            plt.gca().spines['right'].set_visible(False)
#            plt.gca().spines['top'].set_visible(False)
#            plt.gca().spines['left'].set_linewidth(2)
#            plt.gca().spines['bottom'].set_linewidth(2)
#                
##            
##            
#%%
##colors = np.array(['green','blue','red'])
#colors1 = (np.array(['greenyellow','lawngreen','limegreen','forestgreen','green','darkgreen']))
#colors2 = (np.array(['dodgerblue','deepskyblue','cornflowerblue','royalblue','blue','darkblue']))
#colors3 = (np.array(['lightcoral','indianred','red','firebrick','maroon','darkred']))
#simulation_cases = ['NaturalImage']#,'GratingImage']
#for sm in range(len(simulation_cases)):
#    for nn in range(len(neurons_set)):
#        neurons = neurons_set[nn]
#        for m in range(len(model_set)):
#            model = model_set[m]
#            if m==0:
#                colors = colors1
#            elif m==1:
#                colors = colors2
#            elif m==2:
#                colors = colors3
#            
#            if simulation_cases[sm]=='NaturalImage':
#                Sampling_CV_SingleNeuron = np.load(path1 + 'SamplingStats_CV_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy')
#                Sampling_FF_SingleNeuron = np.load(path1 + 'SamplingStats_FF_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy')            
#                Sampling_Corr_SingleNeuron = np.load(path1 + 'SamplingStats_Corr_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy')
#                Sampling_SigCorr_SingleNeuron = np.load(path1 + 'SamplingStats_SigCorr_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy')
#                Sampling_ISI_SingleNeuron = np.load(path1 + 'SamplingStats_ISI_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy')
#                
#                Spiking_CV_SingleNeuron = np.load(path1 + 'LIFStats_CV_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy')
#                Spiking_FF_SingleNeuron = np.load(path1 + 'LIFStats_FF_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy')            
#                Spiking_Corr_SingleNeuron= np.load(path1 + 'LIFStats_Corr_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy')
#                Spiking_SigCorr_SingleNeuron = np.load(path1 + 'LIFStats_SigCorr_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy')
#                Spiking_ISI_SingleNeuron = np.load(path1 + 'LIFStats_ISI_NaturalImagesSingleNeuron_' + str(neurons) + '_' + model + '.npy')
#            
##                
##                Sampling_CV = np.load(path + 'SamplingStats_CV_NaturalImages_' + str(neurons) + '_' + model + '.npy')
##                Sampling_FF = np.load(path + 'SamplingStats_FF_NaturalImages_' + str(neurons) + '_' + model + '.npy')            
##                Sampling_Corr = np.load(path + 'SamplingStats_Corr_NaturalImages_' + str(neurons) + '_' + model + '.npy')
##                Sampling_SigCorr = np.load(path + 'SamplingStats_SigCorr_NaturalImages_' + str(neurons) + '_' + model + '.npy')
##                Sampling_ISI = np.load(path + 'SamplingStats_ISI_NaturalImages_' + str(neurons) + '_' + model + '.npy')
#                
#                Sampling_CV = Sampling_CV_SingleNeuron#np.load(path + 'SamplingStats_CV_NaturalImages_' + str(neurons) + '_' + model + '.npy')
#                Sampling_FF = Sampling_FF_SingleNeuron#np.load(path + 'SamplingStats_FF_NaturalImages_' + str(neurons) + '_' + model + '.npy')            
#                Sampling_Corr = Sampling_Corr_SingleNeuron#np.load(path + 'SamplingStats_Corr_NaturalImages_' + str(neurons) + '_' + model + '.npy')
#                Sampling_SigCorr = Sampling_SigCorr_SingleNeuron#np.load(path + 'SamplingStats_SigCorr_NaturalImages_' + str(neurons) + '_' + model + '.npy')
#                Sampling_ISI = Sampling_ISI_SingleNeuron#np.load(path + 'SamplingStats_ISI_NaturalImages_' + str(neurons) + '_' + model + '.npy')
#                
#                
#                Spiking_CV = Spiking_CV_SingleNeuron#np.load(path + 'LIFStats_CV_NaturalImages_' + str(neurons) + '_' + model + '.npy')
#                Spiking_FF = Spiking_FF_SingleNeuron#np.load(path + 'LIFStats_FF_NaturalImages_' + str(neurons) + '_' + model + '.npy')            
#                Spiking_Corr = Spiking_Corr_SingleNeuron#np.load(path + 'LIFStats_Corr_NaturalImages_' + str(neurons) + '_' + model + '.npy')
#                Spiking_SigCorr = Spiking_SigCorr_SingleNeuron#np.load(path + 'LIFStats_SigCorr_NaturalImages_' + str(neurons) + '_' + model + '.npy')
#                Spiking_ISI = Spiking_ISI_SingleNeuron#np.load(path + 'LIFStats_ISI_NaturalImages_' + str(neurons) + '_' + model + '.npy')
#                
##                Spiking_CV = np.load(path + 'LIFStats_CV_NaturalImages_' + str(neurons) + '_' + model + '.npy')
##                Spiking_FF = np.load(path + 'LIFStats_FF_NaturalImages_' + str(neurons) + '_' + model + '.npy')            
##                Spiking_Corr = np.load(path + 'LIFStats_Corr_NaturalImages_' + str(neurons) + '_' + model + '.npy')
##                Spiking_SigCorr = np.load(path + 'LIFStats_SigCorr_NaturalImages_' + str(neurons) + '_' + model + '.npy')
##                Spiking_ISI = np.load(path + 'LIFStats_ISI_NaturalImages_' + str(neurons) + '_' + model + '.npy')
#            
#            else:
#                chosen_contrast = 1
#                Sampling_CV = np.load(path + 'SamplingStats_CV_GratingImages_' + str(neurons) + '_' + model + '.npy')
#                Sampling_FF = np.load(path + 'SamplingStats_FF_GratingImages_' + str(neurons) + '_' + model + '.npy')            
#                Sampling_Corr = np.load(path + 'SamplingStats_Corr_GratingImages_' + str(neurons) + '_' + model + '.npy')
#                Sampling_SigCorr = np.load(path + 'SamplingStats_SigCorr_GratingImages_' + str(neurons) + '_' + model + '.npy')
#                Sampling_ISI = np.load(path + 'SamplingStats_ISI_GratingImages_' + str(neurons) + '_' + model + '.npy')
#                
#                Sampling_CV = np.squeeze(Sampling_CV[chosen_contrast,:,:])
#                Sampling_FF = np.squeeze(Sampling_FF[chosen_contrast,:,:])
#                Sampling_Corr = np.squeeze(Sampling_Corr[chosen_contrast,:,:,:])
#                Sampling_SigCorr = np.squeeze(Sampling_SigCorr[chosen_contrast,:,:,:])
#                Sampling_ISI = np.squeeze(Sampling_ISI[chosen_contrast,:,:,:,:])
#                
#                Spiking_CV = np.load(path + 'LIFStats_CV_GratingImages_' + str(neurons) + '_' + model + '.npy')
#                Spiking_FF = np.load(path + 'LIFStats_FF_GratingImages_' + str(neurons) + '_' + model + '.npy')            
#                Spiking_Corr = np.load(path + 'LIFStats_Corr_GratingImages_' + str(neurons) + '_' + model + '.npy')  
#                Spiking_SigCorr = np.load(path + 'LIFStats_SigCorr_GratingImages_' + str(neurons) + '_' + model + '.npy')
#                Spiking_ISI = np.load(path + 'LIFStats_ISI_GratingImages_' + str(neurons) + '_' + model + '.npy')
#                
#                Spiking_CV = np.squeeze(Spiking_CV[chosen_contrast,:,:])
#                Spiking_FF = np.squeeze(Spiking_FF[chosen_contrast,:,:])
#                Spiking_Corr = np.squeeze(Spiking_Corr[chosen_contrast,:,:,:])
#                Spiking_SigCorr = np.squeeze(Spiking_SigCorr[chosen_contrast,:,:,:])
#                Spiking_ISI = np.squeeze(Spiking_ISI[chosen_contrast,:,:,:,:])
#    
#            neuron_chosen = neuron_chosen_set[nn,m]
#            image_chosen = image_chosen_set[nn,m]
#            neuron_chosen1 = 15
#            
#            sampled_ISI = np.squeeze(Sampling_ISI_SingleNeuron[image_chosen,0,neuron_chosen1,:])
#            sampled_ISI = sampled_ISI[np.logical_not((sampled_ISI<0.0))]
#            sampled_CV = np.squeeze(Sampling_CV_SingleNeuron[:,neuron_chosen1])
#            sampled_CV = sampled_CV[np.logical_not(np.isnan(sampled_CV))]
#            sampled_CV = sampled_CV[np.logical_not((sampled_CV<=0.0))]
#            sampled_CV_all = Sampling_CV.flatten() #Sampling_CV.flatten()
#            sampled_CV_all = sampled_CV_all[np.logical_not(np.isnan(sampled_CV_all))]
#            sampled_CV_all = sampled_CV_all[np.logical_not((sampled_CV_all<=0.0))]
#            sampled_FF = np.squeeze(Sampling_FF_SingleNeuron[:,neuron_chosen1])
#            sampled_FF = sampled_FF[np.logical_not(np.isnan(sampled_FF))]
#            sampled_FF = sampled_FF[np.logical_not((sampled_FF<=0.0))]
#            sampled_FF_all = Sampling_FF.flatten() #Sampling_FF.flatten()
#            sampled_FF_all = sampled_FF_all[np.logical_not(np.isnan(sampled_FF_all))]
#            sampled_FF_all = sampled_FF_all[np.logical_not((sampled_FF_all<=0.0))]
#            sampled_Corr = np.squeeze(Sampling_Corr_SingleNeuron[im_chosen,neuron_chosen1,:])
#            sampled_Corr = sampled_Corr[np.logical_not(np.isnan(sampled_Corr))]
#            sampled_Corr = sampled_Corr[np.logical_not((np.abs(sampled_Corr)==1.0))]
#            sampled_Corr = sampled_Corr[np.logical_not((np.abs(sampled_Corr)==100))]
#            sampled_Corr = sampled_Corr[np.logical_not((np.abs(sampled_Corr)>0.9))]
#            sampled_Corr_all = Sampling_Corr.flatten()
#            sampled_SigCorr_all = Sampling_SigCorr.flatten() 
#            sampled_SigCorr_all = sampled_SigCorr_all[np.logical_not(np.isnan(sampled_Corr_all))]
#            sampled_Corr_all = sampled_Corr_all[np.logical_not(np.isnan(sampled_Corr_all))]
#            sampled_SigCorr_all = sampled_SigCorr_all[np.logical_not((np.abs(sampled_Corr_all)==1.0))]
#            sampled_Corr_all = sampled_Corr_all[np.logical_not((np.abs(sampled_Corr_all)==1.0))]
#            sampled_SigCorr_all = sampled_SigCorr_all[np.logical_not((np.abs(sampled_Corr_all)==100))]
#            sampled_Corr_all = sampled_Corr_all[np.logical_not((np.abs(sampled_Corr_all)==100))]
##            sampled_Corr_all = sampled_Corr_all[np.logical_not((np.abs(sampled_Corr_all)>0.9))]
#            
#        
#            LIF_ISI = np.squeeze(Spiking_ISI_SingleNeuron[image_chosen,0,neuron_chosen,:])
#            LIF_ISI = LIF_ISI[np.logical_not((LIF_ISI<0.0))]
#            LIF_CV = np.squeeze(Spiking_CV_SingleNeuron[image_chosen,:])#neuron_chosen])
#            LIF_CV = LIF_CV[np.logical_not(np.isnan(LIF_CV))]
#            LIF_CV = LIF_CV[np.logical_not((LIF_CV<=0.0))]
#            LIF_CV_all = Spiking_CV.flatten()
#            LIF_CV_all = LIF_CV_all[np.logical_not(np.isnan(LIF_CV_all))]
#            LIF_CV_all = LIF_CV_all[np.logical_not((LIF_CV_all<=0.0))]
#            LIF_FF = np.squeeze(Spiking_FF_SingleNeuron[:,neuron_chosen])
#            LIF_FF = LIF_FF[np.logical_not(np.isnan(LIF_FF))]
#            LIF_FF = LIF_FF[np.logical_not((LIF_FF<=0.0))]
#            LIF_FF_all = Spiking_FF.flatten()
#            LIF_FF_all = LIF_FF_all[np.logical_not(np.isnan(LIF_FF_all))]
#            LIF_FF_all = LIF_FF_all[np.logical_not((LIF_FF_all<=0.0))]
#            LIF_Corr = np.squeeze(Spiking_Corr_SingleNeuron[im_chosen,neuron_chosen1,:])
#            LIF_Corr = LIF_Corr[np.logical_not(np.isnan(LIF_Corr))]
#            LIF_Corr = LIF_Corr[np.logical_not((np.abs(LIF_Corr)==1.0))]
#            LIF_Corr = LIF_Corr[np.logical_not((np.abs(LIF_Corr)==100))]
##            LIF_Corr = LIF_Corr[np.logical_not((np.abs(LIF_Corr)>0.9))]
#            LIF_Corr_all = Spiking_Corr.flatten()
#            LIF_SigCorr_all = Spiking_SigCorr.flatten()
#            LIF_SigCorr_all = LIF_SigCorr_all[np.logical_not(np.isnan(LIF_Corr_all))]
#            LIF_Corr_all = LIF_Corr_all[np.logical_not(np.isnan(LIF_Corr_all))]
#            LIF_SigCorr_all = LIF_SigCorr_all[np.logical_not((np.abs(LIF_Corr_all)==1.0))]
#            LIF_Corr_all = LIF_Corr_all[np.logical_not((np.abs(LIF_Corr_all)==1.0))]
#            LIF_SigCorr_all = LIF_SigCorr_all[np.logical_not((np.abs(LIF_Corr_all)==100))]
#            LIF_Corr_all = LIF_Corr_all[np.logical_not((np.abs(LIF_Corr_all)==100))]
##            LIF_Corr_all = LIF_Corr_all[np.logical_not((np.abs(LIF_Corr_all)>0.9))]
#            
#            print('Number of elements for Sampling FF for model ' + model_set[m] +  str(np.shape(sampled_FF_all)))
#            print('Number of elements for Sampling CV for model ' + model_set[m] +  str(np.shape(sampled_CV_all)))
#            print('Number of elements for Spiking FF for model ' + model_set[m] +  str(np.shape(LIF_FF_all)))
#            print('Number of elements for Spiking CV for model ' + model_set[m] +  str(np.shape(LIF_CV_all)))
#            
#            
#            plt.figure(100)            
##            plt.hist(sampled_CV_all, **kwargs)
##            sns.kdeplot(sampled_CV_all,color=colors[3],kernel='gau',shade=True, bw=0.1, clip = [0.15, 4])
#            if simulation_cases[sm]=='NaturalImage':
#                sns.set_style("whitegrid", {'axes.grid' : False})
#                plt.subplot(2,3,1)
#                sns.distplot(sampled_CV_all, hist=True, kde=True, norm_hist=True,  
#                color = colors[5],bins=50, 
#                hist_kws={'color': colors[3],'range': [0.1, 4.0],'alpha':0.1},#,'edgecolor':'black'},
#                kde_kws={'linewidth': 3,'bw':0.1,'clip': [0.1, 4.0]})
#                plt.ylim(0.0,4.0)
#            else:
#                plt.subplot(2,3,4)
#                sns.distplot(sampled_CV_all, hist=True, kde=True, norm_hist=True,  
#                color = colors[5],bins=50, 
#                hist_kws={'color': colors[3],'range': [0.1, 4.0],'alpha':0.1},#,'edgecolor':'black'},
#                kde_kws={'linewidth': 3,'bw':0.14,'clip': [0.25, 4.0]})
#                plt.ylim(0.0,2.5)
#                plt.xlabel('CV of all neurons across all images',fontsize=15)
#            plt.plot(np.mean(sampled_CV_all),0.0,marker='o',markersize=12,markeredgecolor='black',color=colors[3],label=str(np.round(np.mean(sampled_CV_all), 1)))
#            plt.yticks([])
#            plt.vlines(1.0,0,10.5,'k',linewidth=2,linestyle='--')
##            plt.hlines(0.0,-0.1,6,'k',linewidth=1,linestyle='--')
#            plt.xlim(0.0,4)
#            plt.xticks([0.0,1.0,2.0, 3.0, 4.0],fontsize=18,fontweight='bold') 
#            plt.gca().spines['right'].set_visible(False)
#            plt.gca().spines['top'].set_visible(False)
#            plt.gca().spines['bottom'].set_linewidth(2)
#            plt.gca().spines['left'].set_linewidth(2)
#            plt.gca().spines['bottom'].set_color('black')
#            plt.gca().spines['left'].set_color('black') 
#            if m==2:
#                plt.legend(fontsize=18)
##           plt.hist(sampled_FF_all, **kwargs)
##           sns.kdeplot(sampled_FF_all,color=colors[3],kernel='gau',shade=True, bw=0.1, clip = [0.1, 3])
#            if simulation_cases[sm]=='NaturalImage': 
#                plt.subplot(2,3,2)
#                sns.set_style("whitegrid", {'axes.grid' : False})
#                plt.subplot(2,3,2)
#                sns.distplot(sampled_FF_all, hist=True, kde=True, norm_hist=True,  
#                color = colors[5],bins=50, 
#                hist_kws={'color': colors[3],'range': [0.1, 3.5],'alpha':0.1},#,'edgecolor':'black'},
#                kde_kws={'linewidth': 3,'bw':0.18,'clip': [0.1, 3.5]})
#                plt.ylim(0.0,2.0)
#            else:
#                plt.subplot(2,3,5)
#                sns.set_style("whitegrid", {'axes.grid' : False})
#                sns.distplot(sampled_FF_all, hist=True, kde=True, norm_hist=True,  
#                color = colors[5],bins=50, 
#                hist_kws={'color': colors[3],'range': [0.1, 3.5],'alpha':0.1},#,'edgecolor':'black'},
#                kde_kws={'linewidth': 3,'bw':0.1,'clip': [0.25, 3.5]})
#                plt.ylim(0.0,2.5)
#                plt.xlabel('FF of all neurons across all images',fontsize=15)
#            plt.plot(np.mean(sampled_FF_all[sampled_FF_all<4]),0.0,marker='o',markersize=12,markeredgecolor='black',color=colors[3],label=str(np.round(np.mean(sampled_FF_all[sampled_FF_all<4]), 1)))
#            plt.yticks([])
#            plt.vlines(1.0,0,10.5,'k',linewidth=2,linestyle='--')
##            plt.hlines(0.0,0,6,'k',linewidth=1,linestyle='--')
#            plt.xlim(0,4)
#            plt.xticks([0.0,1.0,2.0,3.0, 4.0],fontsize=18,fontweight='bold') 
#            plt.gca().spines['right'].set_visible(False)
##            plt.gca().spines['left'].set_visible(False)
#            plt.gca().spines['top'].set_visible(False)
#            plt.gca().spines['bottom'].set_linewidth(2)
#            plt.gca().spines['left'].set_linewidth(2)
#            plt.gca().spines['bottom'].set_color('black')
#            plt.gca().spines['left'].set_color('black')
#            if m==2:
#                plt.legend(fontsize=18)
##            plt.hist(sampled_Corr_all, **kwargs)
##            sns.kdeplot(sampled_Corr_all,color=colors[3],kernel='gau',shade=True, bw=0.1, clip = [-0.25,0.25])
#            if simulation_cases[sm]=='NaturalImage': 
#                plt.subplot(2,3,3)
#                sns.set_style("whitegrid", {'axes.grid' : False})
#                sns.distplot(sampled_Corr_all, hist=True, kde=True, norm_hist=True,  
#                color = colors[5],bins=40, 
#                hist_kws={'color': colors[3],'range': (-0.5,0.5),'alpha':0.1},#,'edgecolor':'black'},
#                kde_kws={'linewidth': 3,'bw':0.02,'clip': [-0.5,0.5]})
#                plt.ylim(0.0,3.0)
#            else:
#                plt.subplot(2,3,6)
#                sns.set_style("whitegrid", {'axes.grid' : False})
#                sns.distplot(sampled_Corr_all, hist=True, kde=True, norm_hist=True,  
#                color = colors[5],bins=40, 
#                hist_kws={'color': colors[3],'range': (-0.5,0.5),'alpha':0.1},#,'edgecolor':'black'},
#                kde_kws={'linewidth': 3,'bw':0.05,'clip': [-0.5,0.5]})
#                plt.ylim(0.0,4.0)
#                plt.xlabel('Corr of all neuron pairs across all images',fontsize=15)
#            plt.plot(np.mean(sampled_Corr_all),0.0,marker='o',markersize=12,markeredgecolor='black',color=colors[3],label=str(np.round(np.mean(sampled_Corr_all), 2)))
#            plt.yticks([])
#            plt.vlines(0.0,0,6.5,'k',linewidth=2,linestyle='--')
##            plt.hlines(0.0,-1,1,'k',linewidth=1,linestyle='--')
#            plt.xlim(-0.6,0.6)
#            plt.xticks([-0.5,0.0,0.5],fontsize=18,fontweight='bold')             
#            plt.gca().spines['right'].set_visible(False)
##            plt.gca().spines['left'].set_visible(False)
#            plt.gca().spines['top'].set_visible(False)
#            plt.gca().spines['bottom'].set_linewidth(2)
#            plt.gca().spines['left'].set_linewidth(2)
#            plt.gca().spines['bottom'].set_color('black')
#            plt.gca().spines['left'].set_color('black')
#            if m==2:
#                plt.legend(fontsize=18)
#            plt.suptitle("Sampling based summary stats for " + str(neurons) + " neurons")
#            
#            
#            
#            
#            plt.figure(210)
##            plt.hist(LIF_CV_all, **kwargs)
##            sns.kdeplot(LIF_CV_all,color=colors[3],kernel='triw',shade=True, bw=0.2, clip = [0.125, 1.5])#clip = [1, 5],hist=True)#, shade=True, color=colors[3], kernel='gau',bw=0.3, clip = [0, 20])
#            if simulation_cases[sm]=='NaturalImage':
#                plt.subplot(2,3,1)
#                sns.set_style("whitegrid", {'axes.grid' : False})
#                sns.distplot(LIF_CV_all, hist=True, kde=True, norm_hist=True,  
#                color = colors[5],bins=75, 
#                hist_kws={'color': colors[3],'range': [0.05, 4],'alpha':0.1},#,'edgecolor':'black'},
#                kde_kws={'linewidth': 3,'bw':0.0526,'clip': [0.15, 4]})
#                plt.ylim(0.0,6.0)
#            else:
#                plt.subplot(2,3,4)
#                sns.set_style("whitegrid", {'axes.grid' : False})
#                sns.distplot(LIF_CV_all, hist=True, kde=True, norm_hist=True,  
#                color = colors[5],bins=50, 
#                hist_kws={'color': colors[3],'range': [0.0, 4],'alpha':0.1},#,'edgecolor':'black'},
#                kde_kws={'linewidth': 3,'bw':0.1,'clip': [0.15, 4]})
#                plt.ylim(0.0,4.0)
#                plt.xlabel('CV of all neurons across all images',fontsize=15)
#            plt.plot(np.mean(LIF_CV_all),0.0,marker='o',markersize=12,markeredgecolor='black',color=colors[3],label=str(np.round(np.mean(LIF_CV_all), 1)))
#            plt.yticks([])
#            plt.vlines(1.0,0,10.5,'k',linewidth=2,linestyle='--')
##            plt.hlines(0.0,-0.1,6,'k',linewidth=1,linestyle='--')
#            plt.xticks([0.0,1.0,2.0,3.0, 4.0],fontsize=18,fontweight='bold') 
#            plt.xlim(0.0,2)
#            plt.gca().spines['right'].set_visible(False)
#            plt.gca().spines['top'].set_visible(False)
#            plt.gca().spines['bottom'].set_linewidth(2)
#            plt.gca().spines['left'].set_linewidth(2)
#            plt.gca().spines['bottom'].set_color('black')
#            plt.gca().spines['left'].set_color('black')
#            if m==2:
#                plt.legend(fontsize=18)
##            plt.hist(LIF_FF_all, **kwargs)
#            if simulation_cases[sm]=='NaturalImage':
#                plt.subplot(2,3,2)
#                sns.set_style("whitegrid", {'axes.grid' : False})
#                sns.distplot(LIF_FF_all, hist=True, kde=True, norm_hist=True,  
#                color = colors[5],bins=75, 
#                hist_kws={'color': colors[3],'range': [0.05, 5],'alpha':0.1},#,'edgecolor':'black'},
#                kde_kws={'linewidth': 3,'bw':0.0526,'clip': [0.05, 5]})
#                plt.ylim(0.0,6.25)
#            else:
#                plt.subplot(2,3,5)
#                sns.set_style("whitegrid", {'axes.grid' : False})
#                sns.distplot(LIF_FF_all, hist=True, kde=True, norm_hist=True,  
#                color = colors[5],bins=75, 
#                hist_kws={'color': colors[3],'range': [0.05, 5],'alpha':0.1},#,'edgecolor':'black'},
#                kde_kws={'linewidth': 3,'bw':0.03,'clip': [0.075, 5]})
#                plt.ylim(0.0,7)
##           sns.kdeplot(LIF_FF_all,color=colors[3],kernel='triw',shade=True, bw=0.2, clip = [0.0, 1.5])#[0.175, 1.5])
#                plt.xlabel('FF of all neurons across all images',fontsize=15)
#            plt.plot(np.mean(LIF_FF_all[LIF_FF_all<=2.5]),0.0,marker='o',markersize=12,markeredgecolor='black',color=colors[3],label=str(np.round(np.mean(LIF_FF_all), 1)))
#            plt.yticks([])
#            plt.vlines(1.0,0,8,'k',linewidth=2,linestyle='--')
##            plt.hlines(0.0,0,3,'k',linewidth=1,linestyle='--')
#            plt.xticks([0.0,1.0,2.0,3.0, 4.0],fontsize=18,fontweight='bold') 
#            plt.xlim(0,2)
#            plt.gca().spines['right'].set_visible(False)
#            plt.gca().spines['top'].set_visible(False)
#            plt.gca().spines['bottom'].set_linewidth(2)
#            plt.gca().spines['left'].set_linewidth(2)
#            plt.gca().spines['bottom'].set_color('black')
#            plt.gca().spines['left'].set_color('black')
#            if m==2:
#                plt.legend(fontsize=18)
##            plt.hist(LIF_Corr_all, **kwargs)
##            sns.kdeplot(LIF_Corr_all,color=colors[3],kernel='triw',shade=True, bw=0.1, clip = [-0.4, 0.4])
#            if simulation_cases[sm]=='NaturalImage':
#                plt.subplot(2,3,3)
#                sns.set_style("whitegrid", {'axes.grid' : False})
#                sns.distplot(LIF_Corr_all, hist=True, kde=True, norm_hist=True,  
#                color = colors[5],bins=75, 
#                hist_kws={'color': colors[3],'range': [-0.5, 0.5],'alpha':0.1},#,'edgecolor':'black'},
#                kde_kws={'linewidth': 3,'bw':0.018,'clip': [-0.5, 0.5]})
#                plt.ylim(0.0,3)
#            else:
#                plt.subplot(2,3,6)
#                sns.set_style("whitegrid", {'axes.grid' : False})
#                sns.distplot(LIF_Corr_all, hist=True, kde=True, norm_hist=True,  
#                color = colors[5],bins=75, 
#                hist_kws={'color': colors[3],'range': [-0.5, 0.5],'alpha':0.1},#,'edgecolor':'black'},
#                kde_kws={'linewidth': 3,'bw':0.018,'clip': [-0.5, 0.5]})
#                plt.ylim(0.0,4.5)
#                plt.xlabel('Corr of all neuron pairs across all images',fontsize=15)
#            plt.plot(np.mean(LIF_Corr_all),0.0,marker='o',markersize=12,markeredgecolor='black',color=colors[3],label=str(np.round(np.mean(LIF_Corr_all), 3)))
#            plt.yticks([])
#            plt.vlines(0.0,0,6.5,'k',linewidth=2,linestyle='--')
#            plt.xlim(-0.6,0.6)
#            plt.xticks([-0.5,0.0,0.5],fontsize=18,fontweight='bold') 
#            plt.gca().spines['right'].set_visible(False)
#            plt.gca().spines['top'].set_visible(False)
#            plt.gca().spines['bottom'].set_linewidth(2)
#            plt.gca().spines['left'].set_linewidth(2)
#            plt.gca().spines['bottom'].set_color('black')
#            plt.gca().spines['left'].set_color('black')
#
#            if m==2:
#                plt.legend(fontsize=18)
#            plt.suptitle("LIF based summary stats for " + str(neurons) + " neurons")
#            
#                        