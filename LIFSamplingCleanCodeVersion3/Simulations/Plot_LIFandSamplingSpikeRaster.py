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

colors = np.array(['green','blue','red'])
colors1 = np.flip(np.array(['greenyellow','limegreen','darkgreen']))
colors2 = np.flip(np.array(['cornflowerblue','dodgerblue','darkblue']))
colors3 = np.flip(np.array(['lightcoral','indianred','darkred']))
#colors1 = np.array(['darkgreen','greenyellow'])
#colors2 = np.array(['darkblue','dodgerblue'])
#colors3 = np.array(['darkred','lightcoral'])

## Initialize variables    
path = '../Results/'
model_set = ['NN', 'IN','ISN']
neurons_set = [128]
pix = 64 #image pixels
dimension = 8 #sqrt(image pixels)
simulation_cases = ['NaturalImage']#,'Grating']#[]#

#%% Figures being generated for each Image   
im = 11
cnt = -1
time_lm = 20
top_lm = 3 ## number of neurons chosen to display spikes, depending on firing rate, we show top top_lm firing neurons.
for sm in range(len(simulation_cases)):         
    for nn in range(len(neurons_set)):
        cnt = cnt + 1
        for m in range(len(model_set)):
            if m==0:
                colors_set = colors1
            elif m==1:
                colors_set = colors2
            elif m==2:
                colors_set = colors3
            
            if simulation_cases[sm]=='NaturalImage':
                txt_load = 'Natural'
            if simulation_cases[sm]=='Grating':
                txt_load = 'Grating'
            ## Loading saved results
            h_smp = np.load(path + txt_load + 'ImageHighestFR_PFindex_samples' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy')
            h_lif = np.load(path + txt_load + 'ImageHighestFR_PFindex_LIF' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy')
            print(h_lif)
            print(h_smp)
            ff = np.load(path + txt_load + 'ImageSamplesFF' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy')
            rec = np.load(path + txt_load + 'ImageSamplesRec' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy')
            prob = np.load(path + txt_load + 'ImageSamplesGibbsProb' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy')
            samples = np.load(path + txt_load + 'ImageSamples' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy')
            indices = np.load(path + txt_load + 'ImageLIF_SpikeIndices' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy')
            times = np.load(path + txt_load + 'ImageLIF_SpikeTimes' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy')        
            spike_array_binnedNat = np.load(path + txt_load + 'ImageLIF_BinnedSpikes' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy')
            membrane_potential_lif = np.load(path + txt_load + 'ImageLIF_MembranePotential' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy')
            prob_lif = np.load(path + txt_load + 'ImageLIF_Probability' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy')
            input_current_lif = np.load(path + txt_load + 'ImageLIF_InputCurrent' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy')
            psp_lif = np.load(path + txt_load + 'ImageLIF_PSP' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy')
            fr_lif = np.load(path + txt_load + 'ImageLIF_FR' + '_' + str(neurons_set[nn]) +  '_' + model_set[m] + '.npy')
            
            if simulation_cases[sm]=='NaturalImage':
                Image_chosen = np.load(path + 'NaturalImage' + '_' + str(neurons_set[nn])  + '.npy')
            if simulation_cases[sm]=='Grating':
                phase_display = 1
                Image_chosen_temp = np.load(path + 'GratingImageAllPhases' + '_' + str(neurons_set[nn])  + '.npy')
                Image_chosen = np.reshape(np.squeeze(Image_chosen_temp[phase_display,:]),(8,8))
            
            params = load_simulation_parameters(neurons_set[nn],model_set[m])
            pixel_std = params.sigma_I
            G = params.G
            R = -1.0 * np.dot(G.T ,G)
            pix_var = pixel_std**2
            for nneu in range(params.N):
                rec[nneu,:] = rec[nneu,:] - (R[nneu, nneu] / (2.0 * pix_var))
            marg_p_sampling = marg_prob_compute(samples,params.N)
            marg_p_lif = marg_prob_compute(spike_array_binnedNat,params.N)
            ReconstructedImageSampling = np.sum(np.matmul(params.G,np.diag(marg_p_sampling)),axis=1)
            ReconstructedImageLIF = np.sum(np.matmul(params.G,np.diag(marg_p_lif)),axis=1)
            
            natural_image_duration = np.shape(samples)[1] * params.sampling_bin_ms # duration of spike train simulation
            n_samples = int(natural_image_duration/params.sampling_bin_ms)   
            limt = int(100)#int(natural_image_duration)
            
#            ## Figure showing Sampling spikes as binary matrix of 0s and 1s
#            plt.figure(im*cnt+1)
#            plt.subplot(2,3,m+1)
#            tt_smp = samples.astype(int)
#            samples_int = tt_smp[h_smp[:top_lm],:]
#            samples_temp = np.zeros((top_lm,np.shape(samples_int)[1]))
##            samples_int = np.flip(samples_int,0)
#            vls = np.linspace(0.25,1,top_lm)
#            for tst in range(top_lm):
#                samples_temp[tst,:] = samples_int[tst,:] * vls[tst]
#            samples_temp = np.flip(samples_temp,0)
#            str_gen = 'highest FR'
#            row_lb = np.flip(['3rd '+str_gen, '2nd '+str_gen, '1st '+str_gen])#['5th '+str_gen, '4th '+str_gen, '3rd '+str_gen, '2nd '+str_gen, '1st '+str_gen]
#            tb = plt.table(cellText=samples_int[:,:time_lm],loc=(0,0), cellLoc='center', edges='closed')#,colWidths = ones(np.shape(samples_int)[1])/50,rowLabels = row_lb)
#            for rr in range(np.shape(samples_int)[0]):
#                for cc in range(time_lm):#(np.shape(samples_int)[1]):
#                    if samples_int[rr,cc]==1:
#                        tb._cells[(rr, cc)]._text.set_color(colors_set[rr])
#                    else:
#                        tb._cells[(rr, cc)]._text.set_color('k')
#            tb.auto_set_font_size(False)
#            tb.set_fontsize(9)
#            plt.gca().set_aspect(0.35)
#            plt.gca().spines['right'].set_visible(False)
#            plt.gca().spines['top'].set_visible(False)
#            plt.gca().spines['left'].set_visible(False)
#            plt.gca().spines['bottom'].set_visible(False)
#            plt.xlabel("Gibbs Sampling bins (each bin 5 ms)",fontsize=12)
#            plt.xticks([])
#            plt.yticks([])
#            plt.subplot(2,3,3+m+1)
#            plt.pcolor(samples_temp[:,:time_lm],cmap=ListedColormap(np.append('w',colors_set[:top_lm])),edgecolors='k', linewidths=1)
##            plt.xticks([0, int(n_samples/2), int(n_samples)],[0, int(natural_image_duration/2), int(natural_image_duration)],fontsize=12)
#            plt.yticks(np.flip(np.array(range(top_lm))+0.5))#,row_lb)#np.array(range(top_lm))+1,fontsize=12)
#            plt.gca().set_aspect(0.5)
#            plt.xticks([])
#            plt.yticks([])
#            plt.xlabel("Simulation time in ms",fontsize=12)
#            plt.suptitle("Samples for top " + str(top_lm) + " firing \nout of " + str(neurons_set[nn]) + " neurons for " + simulation_cases[sm])
#            
#            
##            
#            ## Figure showing Sampling spikes as table type matrix image
#            plt.figure(im*cnt+2)
#            plt.subplot(3,1,m+1)
#            plt.pcolor(samples[:,:],cmap=ListedColormap(['w',colors[m]]),edgecolors='k', linewidths=0.01)
##            plt.pcolor(samples[h_smp[:top_lm],:],cmap=ListedColormap(['w',colors[m]]),edgecolors='k', linewidths=0.1)
#            plt.xticks([0, int(n_samples/2), int(n_samples)],[0, int(natural_image_duration/2), int(natural_image_duration)],fontsize=15,fontweight='bold')
##            plt.gca().set_aspect(0.15)
##            plt.yticks(np.flip(np.array(range(top_lm))+0.5),row_lb)#np.array(range(top_lm))+1,fontsize=12)
#            plt.xlabel("Simulation time in ms",fontsize=15,fontweight='bold')
##            plt.yticks([0,39,79,99,119],['$\mathcal{x}_1$', '$x_{40}$', '$x_{80}$', '$x_{100}$', '$x_{120}$'],fontsize=15,fontweight='bold')
#            plt.ylabel("Neurons",fontsize=18)
#            plt.gca().spines['right'].set_visible(False)
#            plt.gca().spines['top'].set_visible(False)
#            plt.gca().spines['bottom'].set_linewidth(2)
#            plt.gca().spines['left'].set_linewidth(2)
##            plt.subplots_adjust(wspace=0.2, hspace=0.75)
##            plt.suptitle("Samples shown in binned display for top " + str(top_lm) + " \nfiring out of " + str(neurons_set[nn]) + " neurons for " + simulation_cases[sm])
##            
##
#            ## Figure showing PFs of highest firing nurons based on Gibbs Sampling
#            plt.figure(im*cnt+3)
#            plt.subplot(5,3,m+1)
#            plt.imshow(np.reshape(np.squeeze(params.G[:,h_smp[0]]),(8,8)),cmap='gray',interpolation='none')
#            plt.axis('off')
#            plt.gca().set_aspect('equal')
#            plt.title('PF of Highest FR \nfor ' + model_set[m] + ' model',color=colors[m],fontsize=12)
#            plt.subplot(5,3,m+4)
#            plt.imshow(np.reshape(np.squeeze(params.G[:,h_smp[1]]),(8,8)),cmap='gray',interpolation='none')
#            plt.axis('off')
#            plt.gca().set_aspect('equal')
##            plt.title('PF of 2nd Highest FR \nfor ' + model_set[m] + ' model',color=colors[m],fontsize=12)
#            plt.subplot(5,3,m+7)
#            plt.imshow(np.reshape(np.squeeze(params.G[:,h_smp[2]]),(8,8)),cmap='gray',interpolation='none')
#            plt.axis('off')
#            plt.gca().set_aspect('equal')
##            plt.title('PF of 3rd Highest FR \nfor ' + model_set[m] + ' model',color=colors[m],fontsize=12)
#            plt.subplot(5,3,13)
#            plt.imshow(Image_chosen,cmap='gray',interpolation='none')
#            plt.axis('off')
#            plt.gca().set_aspect('equal')
##            plt.title('Preprocessed Image',fontsize=12)
#            plt.subplot(5,3,m+10)
#            plt.imshow(np.reshape(ReconstructedImageSampling,(8,8)),cmap='gray',interpolation='none')
#            plt.axis('off')
#            plt.gca().set_aspect('equal')
##            plt.title('Reconstructed Image \nfor ' + model_set[m] + ' model',color=colors[m],fontsize=12)
#            plt.subplots_adjust(wspace=0.25, hspace=0.1)
#            plt.suptitle("PFs of highest FR for Sampling of " + str(neurons_set[nn]) + " \nneurons for " + simulation_cases[sm])
#
#            
#            ## Figure showing feedforward and recurrent inputs for Gibbs Sampling
#            plt.figure(im*cnt+4)
#            plt.subplot(3,3,3*m+1)
#            str_gen = 'highest FR'
##            row_lb = np.flip(np.array(['3rd '+str_gen, '2nd '+str_gen, '1st '+str_gen]))#np.flip(np.array(['5th '+str_gen, '4th '+str_gen, '3rd '+str_gen, '2nd '+str_gen, '1st '+str_gen]))
#            row_lb = np.flip(np.array(['1st neuron', '2nd neuron', '3rd neuron ']))#np.flip(np.array(['5th '+str_gen, '4th '+str_gen, '3rd '+str_gen, '2nd '+str_gen, '1st '+str_gen]))
#            for i in range(top_lm):
#                txt = row_lb[i]
##                plt.plot(5*np.array(range(n_samples)),ff[h_smp[i],:],color=colors_set[i],label=txt)
#                plt.plot(np.array(range(time_lm+1)),ff[h_smp[i],:time_lm+1],'-o',markersize=5,color=colors_set[i],label=txt)
#            plt.gca().set_aspect(0.15)
##            plt.plot(5*np.array(range(n_samples)),np.zeros(len(5*np.array(range(n_samples)))),'--k',linewidth=0.5)
#            plt.plot(np.array(range(time_lm+1)),np.zeros(len(np.array(range(time_lm+1)))),'-k',linewidth=2)
#            plt.gca().spines['right'].set_visible(False)
#            plt.gca().spines['top'].set_visible(False)
#            plt.gca().spines['bottom'].set_visible(False)
#            plt.gca().spines['left'].set_linewidth(2)
#            plt.ylim([-33,33])
#            plt.xlim([0,(time_lm)+0.5])
#            plt.xticks([0, int(time_lm/2), int(time_lm)],fontsize=15,fontweight='bold')
#            plt.yticks([-30, 0,30],fontsize=15,fontweight='bold')
##            plt.xlim([0,5*(n_samples-1)])
##            plt.xticks([0, int(natural_image_duration/2), int(natural_image_duration)])
#            plt.legend(fontsize=12,loc='lower left')#, bbox_to_anchor=(0.75, 1.0),ncol=1, fancybox=True)
##            plt.xlabel("Simulation time in ms",fontsize=15)
#            plt.ylabel("Feedforward input",fontsize=15)
#            plt.subplot(3,3,3*m+2)
#            for i in range(top_lm):
##                plt.plot(5*np.array(range(n_samples)),rec[h_smp[i],:],color=colors_set[i])
#                plt.plot(np.array(range(time_lm+1)),rec[h_smp[i],:time_lm+1],'-o',markersize=5,color=colors_set[i])
#            plt.gca().set_aspect(0.15)
##            plt.plot(5*np.array(range(n_samples)),np.zeros(len(5*np.array(range(n_samples)))),'--k',linewidth=0.5)
#            plt.plot(np.array(range(time_lm+1)),np.zeros(len(np.array(range(time_lm+1)))),'-k',linewidth=2)
#            plt.gca().spines['right'].set_visible(False)
#            plt.gca().spines['top'].set_visible(False)
#            plt.gca().spines['bottom'].set_visible(False)
#            plt.gca().spines['left'].set_linewidth(2)
#            plt.ylim([-33,33])
#            plt.xlim([0,(time_lm)+0.5])
#            plt.xticks([0, int(time_lm/2), int(time_lm)],fontsize=15,fontweight='bold')
#            plt.yticks([-30, 0,30],fontsize=15,fontweight='bold')
##            plt.xlim([0,5*(n_samples-1)])
##            plt.xticks([0, int(natural_image_duration/2), int(natural_image_duration)])
##            plt.xlabel("Simulation time in ms",fontsize=15)
#            plt.ylabel("Recurrent Inputs",fontsize=15)
##            plt.subplots_adjust(wspace=0.4, hspace=0.01)
#            plt.subplot(3,3,3*m+3)
#            str_gen = 'highest FR'
#            row_lb = np.flip(np.array(['1st neuron', '2nd neuron', '3rd neuron ']))
##            row_lb = np.flip(np.array(['5th '+str_gen, '4th '+str_gen, '3rd '+str_gen, '2nd '+str_gen, '1st '+str_gen]))
#            for i in range(top_lm):
#                txt = row_lb[i]
##                plt.plot(np.array(range(50*n_samples)),np.repeat(np.squeeze(prob[h_smp[i],:]),50),color=colors_set[i],label=txt)
#                plt.plot(np.array(range(time_lm+1)),prob[h_smp[i],:time_lm+1],'-o',markersize=5,color=colors_set[i])#,label=txt)
#            plt.gca().set_aspect(8)
#            plt.gca().spines['right'].set_visible(False)
#            plt.gca().spines['top'].set_visible(False)
#            plt.gca().spines['left'].set_linewidth(2)
#            plt.gca().spines['bottom'].set_linewidth(2)
#            plt.ylim([0,1.02])
#            plt.xlim([0,(time_lm)+0.2])
#            plt.xticks([0, int(time_lm/2), int(time_lm)],fontsize=15,fontweight='bold')
#            plt.yticks([0,0.5,1.0],fontsize=15,fontweight='bold')
##            plt.xlim([0,50*n_samples])
##            plt.xlim([0,n_samples])
##            plt.xticks([0, int(50*n_samples/2), 50*n_samples],[0, int(natural_image_duration/2), int(natural_image_duration)])
##            plt.legend(fontsize=8,loc='upper center', bbox_to_anchor=(0.75, 0.75),ncol=1, fancybox=True)
##            plt.xlabel("Simulation time in ms",fontsize=15)
#            plt.ylabel("Gibbs probability",fontsize=15)
#            plt.suptitle("FF and Rec to highest FR neurons for Sampling \nof " + str(neurons_set[nn]) + " neurons for " + simulation_cases[sm])
#
##            
            ## Figure showing LIF spikes
            plt.figure(im*cnt+5)
            gap = 1000
            plt.subplot(3,1,m+1)
            for tt in range(neurons_set[nn]):
#                plt.plot(range(500),(tt*gap)*np.ones(500),'--k',linewidth=0.25)
                plt.plot(np.array(times[indices==tt]),(tt*gap)*np.ones(np.sum(indices==tt)),'|',markeredgewidth=2.5,markersize=6,color=colors[m])
#                plt.scatter(np.array(times[indices==tt]),(tt*gap)*np.ones(np.sum(indices==tt)),marker='|',edgewidth=2,linewidths=1000,s=25,color=colors[m])            
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['left'].set_linewidth(2)
            plt.gca().spines['bottom'].set_linewidth(2)
#            plt.xlim([0, int(natural_image_duration)])
#            plt.ylim([1, neurons_set[nn]])
            plt.xlabel("Simulation time in ms",fontsize=18)
            plt.ylabel("Neurons",fontsize=18)
            plt.yticks(np.array([0*gap, 24*gap, 49*gap, 74*gap, 99*gap, 124*gap]),np.array([1, 25, 50, 75, 100, 125]),fontsize=18,fontweight='bold')
            plt.xticks([0, 250, 500],fontsize=18,fontweight='bold')
            plt.xlim([0, 500])
            plt.ylim([0, 128*gap])
#            plt.ylim(0, 128*gap)
#            plt.subplots_adjust(wspace=0.25, hspace=0.5)
#            plt.suptitle("Raster plot of spikes from LIF simulations in BRIAN \nfor " + str(neurons_set[nn]) + " neurons for " + simulation_cases[sm])
##
##            
            ## Figure showing top LIF firing neurons' spikes
            plt.figure(im*cnt+6)
            plt.subplot(3,3,m+1)
            row_lb = (np.array(['1st neuron', '2nd neuron', '3rd neuron ']))
            for ttt in range(top_lm):
                tt = top_lm - (ttt+1)
                plt.plot(range(limt),(ttt+1)*np.ones(limt),'--k',linewidth=1)
                plt.plot(times[indices==h_lif[tt]],(ttt+1)*np.ones(np.sum(indices==h_lif[tt])),'|',markeredgewidth=4,markersize=15,color=colors_set[tt],label=row_lb[tt])
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
#            plt.gca().spines['bottom'].set_linewidth(2)
            plt.gca().spines['left'].set_linewidth(2)
            plt.gca().set_aspect(15)
            plt.yticks(np.flip(np.array(range(top_lm))+1),row_lb)
            plt.xlim(0, limt)
            plt.xlabel("Simulation time in ms",fontsize=18)
            plt.yticks(fontsize=18,fontweight='bold')
            plt.xticks(fontsize=18,fontweight='bold')
#            plt.subplots_adjust(wspace=0.25, hspace=0.5)
            plt.suptitle("Raster plot of  for top " + str(top_lm) + " firing neurons \nfor " + str(neurons_set[nn]) + " neurons for " + simulation_cases[sm])
##
###    
#            ## Figure showing PFs of highest firing nurons based on LIF simulations
#            plt.figure(im*cnt+7)
#            plt.subplot(3,5,5*m+1)
#            plt.imshow(np.reshape(np.squeeze(params.G[:,h_lif[0]]),(8,8)),cmap='gray',interpolation='none')
#            plt.axis('off')
#            plt.gca().set_aspect('equal')
#            plt.title('PF of Highest FR \nfor ' + model_set[m] + ' model',color=colors[m],fontsize=12)
#            plt.subplot(3,5,5*m+2)
#            plt.imshow(np.reshape(np.squeeze(params.G[:,h_lif[1]]),(8,8)),cmap='gray',interpolation='none')
#            plt.axis('off')
#            plt.gca().set_aspect('equal')
#            plt.title('PF of 2nd Highest FR \nfor ' + model_set[m] + ' model',color=colors[m],fontsize=12)
#            plt.subplot(3,5,5*m+3)
#            plt.imshow(np.reshape(np.squeeze(params.G[:,h_lif[2]]),(8,8)),cmap='gray',interpolation='none')
#            plt.axis('off')
#            plt.gca().set_aspect('equal')
#            plt.title('PF of 2nd Highest FR \nfor ' + model_set[m] + ' model',color=colors[m],fontsize=12)
#            plt.subplot(3,5,5)
#            plt.imshow(Image_chosen,cmap='gray',interpolation='none')
#            plt.axis('off')
#            plt.gca().set_aspect('equal')
#            plt.title('Preprocessed Image',fontsize=12)
#            plt.subplot(3,5,5*m+4)
#            plt.imshow(np.reshape(ReconstructedImageLIF,(8,8)),cmap='gray',interpolation='none')
#            plt.axis('off')
#            plt.gca().set_aspect('equal')
#            plt.title('Reconstructed Image \nfor ' + model_set[m] + ' model',color=colors[m],fontsize=12)
#            plt.subplots_adjust(wspace=0.25, hspace=0.1)
#            plt.suptitle("PFs of highest FR for LIF simulation of " + str(neurons_set[nn]) + " \nneurons for " + simulation_cases[sm])
#
#            
##            ## Figure showing probability of firing for Gibbs Sampling and for LIF simulations
#            plt.figure(im*cnt+8)
#            plt.subplot(2,3,m+1)
#            str_gen = 'highest FR'
#            row_lb = np.flip(np.array(['1st neuron', '2nd neuron', '3rd neuron ']))
##            row_lb = np.flip(np.array(['5th '+str_gen, '4th '+str_gen, '3rd '+str_gen, '2nd '+str_gen, '1st '+str_gen]))
#            for i in range(top_lm):
#                txt = row_lb[i]
##                plt.plot(np.array(range(50*n_samples)),np.repeat(np.squeeze(prob[h_smp[i],:]),50),color=colors_set[i],label=txt)
#                plt.plot(np.array(range(time_lm)),prob[h_smp[i],:time_lm],color=colors_set[i],label=txt)
#            plt.gca().set_aspect(20)
#            plt.gca().spines['right'].set_visible(False)
#            plt.gca().spines['top'].set_visible(False)
#            plt.ylim([0,1.0])
#            plt.xlim([0,(time_lm)])
#            plt.xticks([0, int(time_lm/2), int(time_lm)],fontsize=12)
#            plt.yticks(fontsize=12)
##            plt.xlim([0,50*n_samples])
##            plt.xlim([0,n_samples])
##            plt.xticks([0, int(50*n_samples/2), 50*n_samples],[0, int(natural_image_duration/2), int(natural_image_duration)])
##            plt.legend(fontsize=8,loc='upper center', bbox_to_anchor=(0.75, 0.75),ncol=1, fancybox=True)
#            plt.xlabel("Simulation time in ms",fontsize=12)
#            plt.ylabel("Gibbs probability",fontsize=12)
##            plt.subplot(2,3,3+m+1)
##            for i in range(top_lm):
#            plt.plot(np.array(range(50*n_samples)),prob_lif[h_lif[i],:],color=colors_set[i])
#    #        plt.gca().set_aspect(0.7)
#            plt.gca().spines['right'].set_visible(False)
#            plt.gca().spines['top'].set_visible(False)
#            plt.ylim([0,1.0])
#            plt.xlim([0,50*n_samples])
#            plt.xticks([0, int(50*n_samples/2), 50*n_samples],[0, int(natural_image_duration/2), int(natural_image_duration)])
#            plt.xlabel("Simulation time in ms",fontsize=12)
#            plt.ylabel("LIF spiking probability",fontsize=12)
#            plt.subplots_adjust(wspace=0.0, hspace=0.0)
#            plt.suptitle('Gibbs Sampling probability (top) compared to \nLIF simulation probability (bottom) of' + str(neurons_set[nn]) + " \nneurons for " + simulation_cases[sm])
##
##            
#            # Figure showing membrane potential and spiking probabilities for LIF simulations
#            plt.figure(im*cnt+9)
#            plt.subplot(3,3,3*m+3)
#            str_gen = 'highest FR'
#            row_lb = (np.array(['1st neuron', '2nd neuron', '3rd neuron ']))
##            row_lb = np.flip(np.array(['3rd '+str_gen, '2nd '+str_gen, '1st '+str_gen]))#np.flip(np.array(['5th '+str_gen, '4th '+str_gen, '3rd '+str_gen, '2nd '+str_gen, '1st '+str_gen]))
#            for i in range(top_lm):
#                plt.plot(np.array(range(limt*10)),membrane_potential_lif[h_lif[i],:(limt*10)]*1000,color=colors_set[i],linewidth=2)#,label=txt)
#            plt.gca().set_aspect(20)
#            plt.plot(np.array(range(limt*10)),-55*np.ones(limt*10),'k',linewidth=2)
#            plt.gca().spines['right'].set_visible(False)
#            plt.gca().spines['top'].set_visible(False)
#            plt.gca().spines['bottom'].set_visible(False)
#            plt.gca().spines['left'].set_linewidth(2)
#            plt.ylim([-70,-55])
#            plt.xlim([0,limt*10])
#            plt.xticks([0, int((limt*10)/2), int(limt*10)],[0, int(limt/2), int(limt)],fontsize=18,fontweight='bold')
#            plt.yticks(fontsize=18,fontweight='bold')
##            plt.legend(fontsize=8,loc='upper center', bbox_to_anchor=(0.75, 2.5),ncol=1, fancybox=True)
#            plt.xlabel("Simulation time in ms",fontsize=18)
#            plt.ylabel("Membrane potential \nof neuron (mVolt)",fontsize=18)
#            plt.subplot(3,3,3*m+1)
#            for i in range(top_lm):
#                plt.plot(np.array(range(limt*10)),prob_lif[h_lif[i],:(limt*10)],color=colors_set[i],linewidth=2,label=row_lb[i])
#            plt.gca().set_aspect(300)
#            plt.gca().spines['right'].set_visible(False)
#            plt.gca().spines['top'].set_visible(False)
#            plt.gca().spines['bottom'].set_linewidth(2)
#            plt.gca().spines['left'].set_linewidth(2)
#            plt.legend(fontsize=18,bbox_to_anchor=(0.6, 2.),ncol=1, fancybox=True)
#            plt.ylim([0.0,1.0])
#            plt.xlim([0,limt*10])
#            plt.xticks([0, int((limt*10)/2), int(limt*10)],[0, int(limt/2), int(limt)],fontsize=18, fontweight='bold')
#            plt.yticks(fontsize=18,fontweight='bold')
#            plt.xlabel("Simulation time in ms",fontsize=18)
#            plt.ylabel("Conditional probability \nof spiking of neuron",fontsize=18)
#            plt.subplot(3,3,3*m+2)
#            for i in range(top_lm):
#                plt.plot(np.array(range(limt*10)),input_current_lif[h_lif[i],:(limt*10)],color=colors_set[i],linewidth=2,label=row_lb[i])
#            plt.gca().set_aspect(4.2)
##            plt.plot(np.array(range(limt*10)),np.zeros(limt*10),'--k',linewidth=0.5)
#            plt.gca().spines['right'].set_visible(False)
#            plt.gca().spines['top'].set_visible(False)
#            plt.gca().spines['bottom'].set_linewidth(2)
#            plt.gca().spines['left'].set_linewidth(2)
##            plt.legend(fontsize=18,loc='upper left')
#            plt.ylim([10,80])
#            plt.xlim([0,limt*10])
#            plt.xticks([0, int((limt*10)/2), int(limt*10)],[0, int(limt/2), int(limt)],fontsize=18,fontweight='bold')
#            plt.yticks(fontsize=18,fontweight='bold')
#            plt.xlabel("Simulation time in ms",fontsize=18)
#            plt.ylabel("Input current injected \nto neuron (mVolt)",fontsize=18)
#            plt.suptitle("Membrane potential and conditional probability of spiking \nfor highest FR neurons in LIF simulation of " + str(neurons_set[nn]) + " neurons for " + simulation_cases[sm])
#
#            
#             ## Figure showing input current and psp for LIF simulations
#            plt.figure(im*cnt+10)
#            plt.subplot(2,3,m+1)
#            str_gen = 'highest FR'
#            row_lb = np.flip(np.array(['5th '+str_gen, '4th '+str_gen, '3rd '+str_gen, '2nd '+str_gen, '1st '+str_gen]))
#            for i in range(top_lm):
#                txt = row_lb[i]
#                plt.plot(np.array(range(limt*10)),input_current_lif[h_lif[i],:(limt*10)],color=colors_set[i],label=txt)
##            plt.gca().set_aspect(0.7)
#            plt.plot(np.array(range(limt*10)),np.zeros(limt*10),'--k',linewidth=0.5)
#            plt.gca().spines['right'].set_visible(False)
#            plt.gca().spines['top'].set_visible(False)
##            plt.ylim([-0.1,0.1])
#            plt.xlim([0,limt*10])
#            plt.xticks([0, int((limt*10)/2), int(limt*10)],[0, int(limt/2), int(limt)])
#            plt.legend(fontsize=8,loc='upper center', bbox_to_anchor=(0.75, 2.5),ncol=1, fancybox=True)
#            plt.xlabel("Simulation time in ms",fontsize=12)
#            plt.ylabel("Input current \ninjected to neuron (mVolt)",fontsize=12)
#            plt.subplot(2,3,3+m+1)
#            for i in range(top_lm):
#                plt.plot(np.array(range(limt*10)),psp_lif[h_lif[i],:(limt*10)],color=colors_set[i],label=txt)
##            plt.gca().set_aspect(0.7)
#            plt.gca().spines['right'].set_visible(False)
#            plt.gca().spines['top'].set_visible(False)
#            plt.ylim([-1,1])
#            plt.xlim([0,limt*10])
#            plt.xticks([0, int((limt*10)/2), int(limt*10)],[0, int(limt/2), int(limt)])
#            plt.xlabel("Simulation time in ms",fontsize=12)
#            plt.ylabel("PSP input \nto spiking of neuron",fontsize=12)
##            plt.subplots_adjust(wspace=0.1, hspace=0.0)
#            plt.suptitle("Input current and psp inputs to highest FR neurons \nin LIF simulation of " + str(neurons_set[nn]) + " neurons for " + simulation_cases[sm])
#            
#            
#            
#            Sampling_pairwise_joint_prob,_,_,_,_ = pairwise_prob_compute(samples,params.N,1) 
#            LIF_pairwise_joint_prob,_,_,_,_ = pairwise_prob_compute(spike_array_binnedNat,params.N,1) 
#            plt.figure(im*cnt+11)
#            plt.subplot(2,3,m+1)
#            plt.scatter(marg_p_sampling,marg_p_lif,s=20, color=colors_set[0])
#            plt.plot(np.linspace(0,1 ,100),np.linspace(0,1,100),'k')
#            plt.xlim(-0.1,1.1)
#            plt.ylim(-0.1,1.1)
#            plt.xlabel("Sample marg",fontsize=10)
#            plt.ylabel("Implied marg",fontsize=10)
#            
#            plt.subplot(2,3,3+m+1)
#            plt.scatter(Sampling_pairwise_joint_prob,LIF_pairwise_joint_prob,s=20, color=colors_set[0])
#            plt.plot(np.linspace(0,1 ,100),np.linspace(0,1,100),'k')
#            plt.xlim(-0.1,1.1)
#            plt.ylim(-0.1,1.1)
#            plt.xlabel("Sample pairwise joint",fontsize=10)
#            plt.ylabel("Implied pairwise joint",fontsize=10)
#            plt.suptitle("LIF-Sampling match of " + str(neurons_set[nn]) + " \nneurons for " + simulation_cases[sm])
#        