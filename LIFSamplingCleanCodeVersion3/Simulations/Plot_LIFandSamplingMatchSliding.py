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
#colors1 = np.flip(np.array(['greenyellow','limegreen','darkgreen']))
#colors2 = np.flip(np.array(['cornflowerblue','dodgerblue','darkblue']))
#colors3 = np.flip(np.array(['lightcoral','indianred','darkred']))

## Initialize variables    
path = '../Results/'
model_set = ['NN', 'IN','ISN']
neurons_set = [128]
pix = 64 #image pixels
dimension = 8 #sqrt(image pixels)
simulation_cases = ['NaturalImage']#,'Grating']#[]#
scenarios = ['Bad','Good']
#sf_case = 1
colors1 = np.array(['greenyellow','limegreen','darkgreen'])
colors2 = np.array(['dodgerblue','cornflowerblue','darkblue'])
colors3 = np.array(['lightcoral','red','darkred'])

#%%
top_lm = 5
for sm in range(len(simulation_cases)):
    if sm==0:
        relevant_txt = 'NaturalImages'
    else:
        relevant_txt = 'GratingImages'
    for nn in range(len(neurons_set)):
        for sc in range(len(scenarios)):
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
                
                Sampling_marginal_prob = np.load(path + 'SlidingSampling_marginal_prob' + relevant_txt + '_' + str(neurons) + '_' + model + '.npy')
                LIF_marginal_prob = np.load(path + 'SlidingLIF_marginal_prob' + relevant_txt + '_' + str(neurons) + '_' + model + '.npy')
                Multi_info = np.load(path + 'SlidingMulti_info_' + relevant_txt + '_' + str(neurons) + '_' + 'NN' + '.npy')
                H_indx = np.flip(np.argsort(Multi_info))
                if scenarios[sc]=='Bad':
                    H_indx_use = np.array([16, 40, 61, 82, 14])#H_indx[:top_lm]
                else:
                    H_indx_use = H_indx[len(H_indx)-top_lm-1:-1]
                num_im = top_lm
                sampling_p_marg = np.reshape(Sampling_marginal_prob[H_indx_use,:],(top_lm*neurons))
                implied_p_marg = np.reshape(LIF_marginal_prob[H_indx_use,:],(top_lm*neurons))
                slp, b = np.polyfit(sampling_p_marg,implied_p_marg, 1)
                corr = np.corrcoef(sampling_p_marg,implied_p_marg)[0,1]
                
                plt.subplot(3,2,2*m+1)
                plt.scatter(sampling_p_marg, implied_p_marg, s=50, color=colors)#, edgecolors='k')
                plt.plot(np.linspace(0,1 ,100),np.linspace(0,1 ,100),'k',linewidth=2)
#                plt.xlim(-0.1,1.1)
#                plt.ylim(-0.1,1.1)
                plt.xlim(0.0,1.1)
                plt.ylim(0.0,1.1)
                plt.xlabel("Sampling based marginal",fontsize=15)
                plt.ylabel("Implied marginal",fontsize=15)
                plt.xticks([0.0,0.5,1.0],fontsize=15, fontweight='bold') 
                plt.yticks([0.0,0.5,1.0],fontsize=15, fontweight='bold') 
                plt.text(0.1, 0.9, 'Slope='+str(np.round(slp,2)), fontsize=15)
                plt.text(0.1, 0.7, 'Corr='+str(np.round(corr,2)), fontsize=15)
                plt.gca().spines['right'].set_visible(False)
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['bottom'].set_linewidth(2)
                plt.gca().spines['left'].set_linewidth(2)
                plt.gca().set_aspect('equal')
                
            
                Sampling_pairwise_joint_prob00 = np.load(path + 'SlidingSampling_pairwise_joint_prob00' + relevant_txt + '_' + str(neurons) + '_' + model + '.npy')
                combs = np.shape(Sampling_pairwise_joint_prob00)[1]
                sampling_p_pairwise00 = np.reshape(Sampling_pairwise_joint_prob00[H_indx_use,:],top_lm*combs)
                Sampling_pairwise_joint_prob01 = np.load(path + 'SlidingSampling_pairwise_joint_prob01' + relevant_txt + '_' + str(neurons) + '_' + model + '.npy')
                sampling_p_pairwise01 = np.reshape(Sampling_pairwise_joint_prob01[H_indx_use,:],top_lm*combs)
                Sampling_pairwise_joint_prob10 = np.load(path + 'SlidingSampling_pairwise_joint_prob10' + relevant_txt + '_' + str(neurons) + '_' + model + '.npy')
                sampling_p_pairwise10 = np.reshape(Sampling_pairwise_joint_prob10[H_indx_use,:],top_lm*combs)
                Sampling_pairwise_joint_prob11 = np.load(path + 'SlidingSampling_pairwise_joint_prob11' + relevant_txt + '_' + str(neurons) + '_' + model + '.npy')
                sampling_p_pairwise11 = np.reshape(Sampling_pairwise_joint_prob11[H_indx_use,:],top_lm*combs)
                
                
                LIF_pairwise_joint_prob00 = np.load(path + 'SlidingLIF_pairwise_joint_prob00' + relevant_txt + '_' + str(neurons) + '_' + model + '.npy')            
                implied_p_pairwise00 = np.reshape(LIF_pairwise_joint_prob00[H_indx_use,:],top_lm*combs)
                LIF_pairwise_joint_prob01 = np.load(path + 'SlidingLIF_pairwise_joint_prob01' + relevant_txt + '_' + str(neurons) + '_' + model + '.npy')            
                implied_p_pairwise01 = np.reshape(LIF_pairwise_joint_prob01[H_indx_use,:],top_lm*combs)
                LIF_pairwise_joint_prob10 = np.load(path + 'SlidingLIF_pairwise_joint_prob10' + relevant_txt + '_' + str(neurons) + '_' + model + '.npy')            
                implied_p_pairwise10 = np.reshape(LIF_pairwise_joint_prob10[H_indx_use,:],top_lm*combs)
                LIF_pairwise_joint_prob11 = np.load(path + 'SlidingLIF_pairwise_joint_prob11' + relevant_txt + '_' + str(neurons) + '_' + model + '.npy')            
                implied_p_pairwise11 = np.reshape(LIF_pairwise_joint_prob11[H_indx_use,:],top_lm*combs)
                
                sampling_all = np.hstack((sampling_p_pairwise00,sampling_p_pairwise01,sampling_p_pairwise10,sampling_p_pairwise11))
                implied_all = np.hstack((implied_p_pairwise00,implied_p_pairwise01,implied_p_pairwise10,implied_p_pairwise11))
                slp, b = np.polyfit(sampling_all,implied_all, 1)
                corr = np.corrcoef(sampling_all,implied_all)[0,1]
                
                
                plt.subplot(3,2,2*m+2)
                plt.scatter(sampling_p_pairwise00, implied_p_pairwise00, s=50, color='w',edgecolors=colors_set[0],label='00')#, edgecolors='k')
                plt.scatter(sampling_p_pairwise01, implied_p_pairwise01, s=50, color=colors_set[1],label='01 or 10')#, edgecolors='k')
                plt.scatter(sampling_p_pairwise10, implied_p_pairwise10, s=50, color=colors_set[1])#, edgecolors='k')
                plt.scatter(sampling_p_pairwise11, implied_p_pairwise11, s=50, color=colors_set[2],label='11')#, edgecolors='k')
            
                plt.plot(np.linspace(0,1 ,100),np.linspace(0,1,100),'k',linewidth=2)
#                plt.xlim(-0.1,1.1)
#                plt.ylim(-0.1,1.1)
                plt.xlim(0.0,1.1)
                plt.ylim(0.0,1.1)
                plt.xlabel("Sampling based pairwise joint",fontsize=15)
                plt.ylabel("Implied pairwise joint",fontsize=15)
                plt.hold('on')
                plt.xticks([0.0,0.5,1.0],fontsize=15,fontweight='bold') 
                plt.yticks([0.0,0.5,1.0],fontsize=15,fontweight='bold') 
                plt.text(0.1, 0.9, 'Slope='+str(np.round(slp,2)), fontsize=15)
                plt.text(0.1, 0.7, 'Corr='+str(np.round(corr,2)), fontsize=15)
                plt.gca().spines['right'].set_visible(False)
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['bottom'].set_linewidth(2)
                plt.gca().spines['left'].set_linewidth(2)
                plt.gca().set_aspect('equal')
                plt.legend(fontsize=15,loc="lower right")
            
            plt.suptitle('LIF-Sampling match for ' + scenarios[sc] + ' ' + relevant_txt + ' for ' + str(neurons) + ' neurons')
            
            
            
            
            
            
#            
#            
#            
#            
#            
#            
#            
#            
#            
#            
#            
#            
#            
#            
#            
##        plt.subplot(3,3,4+m)
##        for im in range(len(im_selected)):
##            Sampling_pairwise_joint_prob = np.load(path + 'Sampling_pairwise_joint_probNaturalImages' + '_' + str(neurons) + '_' + model + '.npy')
##            sampling_p_pairwise = Sampling_pairwise_joint_prob[im_selected[im],:]
##            
##            LIF_pairwise_joint_prob = np.load(path + 'LIF_pairwise_joint_probNaturalImages' + '_' + str(neurons) + '_' + model + '.npy')            
##            implied_p_pairwise = LIF_pairwise_joint_prob[im_selected[im],:]
##            
##            
##            plt.scatter(sampling_p_pairwise, implied_p_pairwise, s=50, color=colors, edgecolors='k')
##            plt.plot(np.linspace(0,1,100),np.linspace(0,1,100),'k')
##            plt.xlim(-0.1,1.1)
##            plt.ylim(-0.1,1.1)
##            plt.xlabel("Sample pairwise joint",fontsize=10)
##            plt.ylabel("Implied pairwise joint",fontsize=10)
##            plt.xticks([0.0,0.5,1.0]) 
##            plt.xticks(fontsize=10)
##            plt.yticks([0.0,0.5,1.0]) 
##            plt.yticks(fontsize=10)
##            plt.hold('on')
###            plt.axis('tight')
##        plt.gca().spines['right'].set_visible(False)
##        plt.gca().spines['top'].set_visible(False)  
#        
#        plt.subplot(2,3,4+m)
#        xx = np.array([])
#        yy = np.array([])
#        for im in range(len(im_selected)):   
#            Sampling_pairwise_joint_prob = np.load(path1 + 'Sampling_pairwise_joint_probNaturalImages' + '_' + str(neurons) + '_' + model + '.npy')
#            sampling_p_pairwise = Sampling_pairwise_joint_prob[im_selected[im],:]
#            
#            LIF_pairwise_joint_prob = np.load(path + 'SlidingLIF_pairwise_joint_probNaturalImages' + '_' + str(neurons) + '_' + model + '.npy')            
#            implied_p_pairwise = LIF_pairwise_joint_prob[im_selected[im],:]
#            
#            xx = np.append(xx,sampling_p_pairwise)
#            yy = np.append(yy,implied_p_pairwise)
#     
#            Sampling_pairwise_joint_prob00 = np.load(path1 + 'Sampling_pairwise_joint_prob00NaturalImages' + '_' + str(neurons) + '_' + model + '.npy')
#            sampling_p_pairwise00 = Sampling_pairwise_joint_prob00[im_selected[im],:]
#            Sampling_pairwise_joint_prob01 = np.load(path1 + 'Sampling_pairwise_joint_prob01NaturalImages' + '_' + str(neurons) + '_' + model + '.npy')
#            sampling_p_pairwise01 = Sampling_pairwise_joint_prob01[im_selected[im],:]
#            Sampling_pairwise_joint_prob10 = np.load(path1 + 'Sampling_pairwise_joint_prob10NaturalImages' + '_' + str(neurons) + '_' + model + '.npy')
#            sampling_p_pairwise10 = Sampling_pairwise_joint_prob10[im_selected[im],:]
#            Sampling_pairwise_joint_prob11 = np.load(path1 + 'Sampling_pairwise_joint_prob11NaturalImages' + '_' + str(neurons) + '_' + model + '.npy')
#            sampling_p_pairwise11 = Sampling_pairwise_joint_prob11[im_selected[im],:]
#            
#            
#            LIF_pairwise_joint_prob00 = np.load(path + 'SlidingLIF_pairwise_joint_prob00NaturalImages' + '_' + str(neurons) + '_' + model + '.npy')            
#            implied_p_pairwise00 = LIF_pairwise_joint_prob00[im_selected[im],:]
#            LIF_pairwise_joint_prob01 = np.load(path + 'SlidingLIF_pairwise_joint_prob01NaturalImages' + '_' + str(neurons) + '_' + model + '.npy')            
#            implied_p_pairwise01 = LIF_pairwise_joint_prob01[im_selected[im],:]
#            LIF_pairwise_joint_prob10 = np.load(path + 'SlidingLIF_pairwise_joint_prob10NaturalImages' + '_' + str(neurons) + '_' + model + '.npy')            
#            implied_p_pairwise10 = LIF_pairwise_joint_prob10[im_selected[im],:]
#            LIF_pairwise_joint_prob11 = np.load(path + 'SlidingLIF_pairwise_joint_prob11NaturalImages' + '_' + str(neurons) + '_' + model + '.npy')            
#            implied_p_pairwise11 = LIF_pairwise_joint_prob11[im_selected[im],:]
#            
#            if im==len(im_selected)-1:
#                plt.scatter(sampling_p_pairwise00, implied_p_pairwise00, s=20, color='w',edgecolors=colors_set[0],label='00')#, edgecolors='k')
#                plt.scatter(sampling_p_pairwise01, implied_p_pairwise01, s=20, color=colors_set[1],label='01 or 10')#, edgecolors='k')
#                plt.scatter(sampling_p_pairwise10, implied_p_pairwise10, s=20, color=colors_set[1])#, edgecolors='k')
#                plt.scatter(sampling_p_pairwise11, implied_p_pairwise11, s=20, color=colors_set[2],label='11')#, edgecolors='k')
#            
#                plt.plot(np.linspace(0,1 ,100),np.linspace(0,1,100),'k')
#                plt.xlim(-0.1,1.1)
#                plt.ylim(-0.1,1.1)
#                plt.xlabel("Sample pairwise joint",fontsize=10)
#                plt.ylabel("Implied pairwise joint",fontsize=10)
#                plt.hold('on')
#                plt.xticks([0.0,0.5,1.0]) 
#                plt.xticks(fontsize=10)
#                plt.yticks([0.0,0.5,1.0]) 
#                plt.yticks(fontsize=10)
##                plt.axis('tight')
#                
#                plt.legend(loc="lower right")
#
#            else:
#                plt.scatter(sampling_p_pairwise00, implied_p_pairwise00, s=20, color='w',edgecolors=colors_set[0])#, edgecolors='k')
#                plt.scatter(sampling_p_pairwise01, implied_p_pairwise01, s=20, color=colors_set[1])#, edgecolors='k')
#                plt.scatter(sampling_p_pairwise10, implied_p_pairwise10, s=20, color=colors_set[1])#, edgecolors='k')
#                plt.scatter(sampling_p_pairwise11, implied_p_pairwise11, s=20, color=colors_set[2])#, edgecolors='k')
#            
#                plt.plot(np.linspace(0,1 ,100),np.linspace(0,1 ,100),'k')
#                plt.xlim(-0.1,1.1)
#                plt.ylim(-0.1,1.1)
#                plt.xlabel("Sample pairwise joint",fontsize=10)
#                plt.ylabel("Implied pairwise joint",fontsize=10)
#                plt.hold('on')
#                plt.xticks([0.0,0.5,1.0]) 
#                plt.xticks(fontsize=10)
#                plt.yticks([0.0,0.5,1.0]) 
#                plt.yticks(fontsize=10)
##                plt.axis('tight')
#        slp1, b1 = np.polyfit(xx,yy, 1)
#        corr1 = np.corrcoef(xx,yy)[0,1]
#        plt.text(0.1, 0.9, 'Slope='+str(np.round(slp1,2)), fontsize=10)
#        plt.text(0.1, 0.7, 'Corr='+str(np.round(corr1,2)), fontsize=10)
#        plt.gca().spines['right'].set_visible(False)
#        plt.gca().spines['top'].set_visible(False)
#        plt.gca().set_aspect('equal')
#            
##        plt.subplot(4,3,10+m)
##        for im in range(len(im_selected)):
##            Sampling_pairwise_diff = np.load(path + 'Sampling_pairwise_joint_diffprobNaturalImages' + '_' + str(neurons) + '_' + model + '.npy')
##            sampling_p_pairwise_diff = Sampling_pairwise_diff[im_selected[im],:]
##        
##            LIF_pairwise_diff = np.load(path + 'LIF_pairwise_joint_diffprobNaturalImages' + '_' + str(neurons) + '_' + model + '.npy')            
##            implied_p_pairwise_diff = LIF_pairwise_diff[im_selected[im],:]
##        
##        
##            plt.scatter(sampling_p_pairwise_diff, implied_p_pairwise_diff, s=50, color=colors, edgecolors='k')
###            plt.plot(np.linspace(-5,15,100),np.linspace(-5,15,100),'k')
##            plt.plot(np.linspace(-0.2,0.2,100),np.linspace(-0.2,0.2,100),'k')
##
###                plt.xlim(-0.1,1.1)
###                plt.ylim(-0.1,1.1)
##            plt.xlabel("Sample pairwise joint diff",fontsize=10)
##            plt.ylabel("Implied pairwise joint diff",fontsize=10)
###                plt.xticks([0.0,0.5,1.0]) 
##            plt.xticks(fontsize=10)
###                plt.yticks([0.0,0.5,1.0]) 
##            plt.yticks(fontsize=10)
##            plt.hold('on')
###            plt.axis('tight')
##        plt.gca().spines['right'].set_visible(False)
##        plt.gca().spines['top'].set_visible(False)  
##    plt.suptitle("LIF-Sampling Match for Natural Images for " + str(neurons) + " neurons")
#
#
##%%
#for nn in range(len(neurons_set)):
#    plt.figure()
#    for m in range(len(model_set)):
#        if m==0:
#            colors = 'green'
#            colors_set = colors1
#        elif m==1:
#            colors = 'blue'
#            colors_set = colors2
#        elif m==2:
#            colors = 'red' 
#            colors_set = colors3
#            
#        neurons = neurons_set[nn]
#        model = model_set[m]
#        params = load_simulation_parameters(neurons,model)
#        sample_hertz = 1.0/(params.sampling_bin_s)
#        
##        plt.subplot(2,3,1+m)
##        xx = np.array([])
##        yy = np.array([])
##        for im in range(len(im_selected)):
##            Sampling_pairwise_diff = np.load(path + 'SlidingSampling_pairwise_joint_logdiffprobNaturalImages' + '_' + str(neurons) + '_' + model + '.npy')
##            sampling_p_pairwise_diff = Sampling_pairwise_diff[im_selected[im],:]
##        
##            LIF_pairwise_diff = np.load(path + 'SlidingLIF_pairwise_joint_logdiffprobNaturalImages' + '_' + str(neurons) + '_' + model + '.npy')            
##            implied_p_pairwise_diff = LIF_pairwise_diff[im_selected[im],:]
##        
##            xx = np.append(xx,sampling_p_pairwise_diff)
##            yy = np.append(yy,implied_p_pairwise_diff)
##            
##            plt.scatter(sampling_p_pairwise_diff, implied_p_pairwise_diff, s=20, color=colors)#, edgecolors='k')
###            plt.plot(np.linspace(-5,15,100),np.linspace(-5,15,100),'k')
##            plt.plot(np.linspace(-0.1,0.1,100),np.linspace(-0.1,0.1,100),'k')
##
##            plt.xlim([-0.1,0.1])
##            plt.ylim(-0.1,0.1)
##            plt.xlabel("Sample: log(pairwise joint)-log(marg prod)",fontsize=10)
##            plt.ylabel("Implied: log(pairwise joint)-log(marg prod)",fontsize=10)
###                plt.xticks([0.0,0.5,1.0]) 
##            plt.xticks(fontsize=10)
###                plt.yticks([0.0,0.5,1.0]) 
##            plt.yticks(fontsize=10)
##            plt.hold('on')
###            plt.axis('tight')
##        slp, b = np.polyfit(xx,yy, 1)
##        corr = np.corrcoef(xx,yy)[0,1]
##        plt.text(0.3, 0.9, 'Slope='+str(np.round(slp,3)), fontsize=10)
##        plt.text(0.3, 0.7, 'Corr='+str(np.round(corr,3)), fontsize=10)
##        plt.gca().spines['right'].set_visible(False)
##        plt.gca().spines['top'].set_visible(False)
##        plt.gca().set_aspect('equal')
#        
#        plt.subplot(1,3,1+m)
#        xx = np.array([])
#        yy = np.array([])
#        for im in range(len(im_selected)):
#            Sampling_pairwise_diff = np.load(path + 'SlidingSampling_pairwise_joint_logdiffprobboundedNaturalImages' + '_' + str(neurons) + '_' + model + '.npy')
#            sampling_p_pairwise_diff = Sampling_pairwise_diff[im_selected[im],:]
#        
#            LIF_pairwise_diff = np.load(path + 'SlidingLIF_pairwise_joint_logdiffprobboundedNaturalImages' + '_' + str(neurons) + '_' + model + '.npy')            
#            implied_p_pairwise_diff = LIF_pairwise_diff[im_selected[im],:]
#        
#            xx = np.append(xx,sampling_p_pairwise_diff)
#            yy = np.append(yy,implied_p_pairwise_diff)
#            
#            plt.scatter(sampling_p_pairwise_diff, implied_p_pairwise_diff, s=20, color=colors)#, edgecolors='k')
##            plt.plot(np.linspace(-5,15,100),np.linspace(-5,15,100),'k')
#            plt.plot(np.linspace(-3,5,100),np.linspace(-3,5,100),'k')
#
#            plt.xlim([-0.2,0.2])
#            plt.ylim([-0.2,0.2])
#            plt.xlabel("Sample(marg>=0.05): log(pairwise joint)-log(marg prod)",fontsize=10)
#            plt.ylabel("Implied(marg>=0.05): log(pairwise joint)-log(marg prod)",fontsize=10)
##                plt.xticks([0.0,0.5,1.0]) 
#            plt.xticks(fontsize=10)
##                plt.yticks([0.0,0.5,1.0]) 
#            plt.yticks(fontsize=10)
#            plt.hold('on')
##            plt.axis('tight')
#        slp, b = np.polyfit(xx,yy, 1)
#        corr = np.corrcoef(xx,yy)[0,1]
#        plt.text(-0.15, 0.15, 'Slope='+str(np.round(slp,3)), fontsize=10)
#        plt.text(-0.15, 0.1, 'Corr='+str(np.round(corr,3)), fontsize=10)
#        plt.gca().spines['right'].set_visible(False)
#        plt.gca().spines['top'].set_visible(False)
#        plt.gca().set_aspect('equal')
##    plt.suptitle("LIF-Sampling Match for Natural Images for " + str(neurons) + " neurons")
#
##%%
#for nn in range(len(neurons_set)):
#    plt.figure()
#    for m in range(len(model_set)):
#        if m==0:
#            colors = 'green'
#            colors_set = colors1
#        elif m==1:
#            colors = 'blue'
#            colors_set = colors2
#        elif m==2:
#            colors = 'red' 
#            colors_set = colors3
#            
#        neurons = neurons_set[nn]
#        model = model_set[m]
#        params = load_simulation_parameters(neurons,model)
#        sample_hertz = 1.0/(params.sampling_bin_s)
#        
#        plt.subplot(2,3,1+m)
#        xx = np.array([])
#        yy = np.array([])
#        for im in range(len(im_selected)):
#            Sampling_pairwise_diff = np.load(path + 'SlidingSampling_pairwise_joint_diffprobNaturalImages' + '_' + str(neurons) + '_' + model + '.npy')
#            sampling_p_pairwise_diff = Sampling_pairwise_diff[im_selected[im],:]
#        
#            LIF_pairwise_diff = np.load(path + 'SlidingLIF_pairwise_joint_diffprobNaturalImages' + '_' + str(neurons) + '_' + model + '.npy')            
#            implied_p_pairwise_diff = LIF_pairwise_diff[im_selected[im],:]
#        
#            xx = np.append(xx,sampling_p_pairwise_diff)
#            yy = np.append(yy,implied_p_pairwise_diff)
#            
#            plt.scatter(sampling_p_pairwise_diff, implied_p_pairwise_diff, s=15, color=colors)#, edgecolors='k')
##            plt.plot(np.linspace(-5,15,100),np.linspace(-5,15,100),'k')
#            plt.plot(np.linspace(-0.1,0.1,100),np.linspace(-0.1,0.1,100),'k')
#            if m==0:
#                plt.xlim([-0.1,0.1])
#                plt.ylim(-0.1,0.1)
#            else:
#                plt.xlim([-0.04,0.04])
#                plt.ylim(-0.04,0.04)
#            plt.xlabel("Sample pairwise and marg diff",fontsize=10)
#            plt.ylabel("Implied pairwise and marg diff",fontsize=10)
##                plt.xticks([0.0,0.5,1.0]) 
#            plt.xticks(fontsize=10)
##                plt.yticks([0.0,0.5,1.0]) 
#            plt.yticks(fontsize=10)
#            plt.hold('on')
##            plt.axis('tight')
#        slp, b = np.polyfit(xx,yy, 1)
#        corr = np.corrcoef(xx,yy)[0,1]
#        if m==0:
#            plt.text(-0.08, 0.08, 'Slope='+str(np.round(slp,2)), fontsize=10)
#            plt.text(-0.08, 0.065, 'Corr='+str(np.round(corr,2)), fontsize=10)
#        else:
#            plt.text(-0.025, 0.025, 'Slope='+str(np.round(slp,2)), fontsize=10)
#            plt.text(-0.025, 0.02, 'Corr='+str(np.round(corr,2)), fontsize=10)
#        plt.gca().spines['top'].set_visible(False)
#        plt.gca().spines['right'].set_visible(False)
#        plt.gca().set_aspect('equal')
#        
#        plt.subplot(2,3,4+m)
#        xx = np.array([])
#        yy = np.array([])
#        for im in range(len(im_selected)):
#            Sampling_pairwise_diff = np.load(path + 'SlidingSampling_pairwise_joint_diffprobboundedNaturalImages' + '_' + str(neurons) + '_' + model + '.npy')
#            sampling_p_pairwise_diff = Sampling_pairwise_diff[im_selected[im],:]
#        
#            LIF_pairwise_diff = np.load(path + 'SlidingLIF_pairwise_joint_diffprobboundedNaturalImages' + '_' + str(neurons) + '_' + model + '.npy')            
#            implied_p_pairwise_diff = LIF_pairwise_diff[im_selected[im],:]
#        
#            xx = np.append(xx,sampling_p_pairwise_diff)
#            yy = np.append(yy,implied_p_pairwise_diff)
#            
#            plt.scatter(sampling_p_pairwise_diff, implied_p_pairwise_diff, s=15, color=colors)#, edgecolors='k')
##            plt.plot(np.linspace(-5,15,100),np.linspace(-5,15,100),'k')
#            plt.plot(np.linspace(-3,5,100),np.linspace(-3,5,100),'k')
#
#            if m==0:
#                plt.xlim([-0.1,0.1])
#                plt.ylim(-0.1,0.1)
#            else:
#                plt.xlim([-0.03,0.03])
#                plt.ylim(-0.03,0.03)
#                
#            plt.xlabel("Sample pairwise and marg diff (marg>=0.05)",fontsize=10)
#            plt.ylabel("Implied pairwise and marg diff (marg>=0.05)",fontsize=10)
##                plt.xticks([0.0,0.5,1.0]) 
#            plt.xticks(fontsize=10)
##                plt.yticks([0.0,0.5,1.0]) 
#            plt.yticks(fontsize=10)
#            plt.hold('on')
##            plt.axis('tight')
#        slp, b = np.polyfit(xx,yy, 1)
#        corr = np.corrcoef(xx,yy)[0,1]
#        if m==0:
#            plt.text(-0.08, 0.08, 'Slope='+str(np.round(slp,2)), fontsize=10)
#            plt.text(-0.08, 0.065, 'Corr='+str(np.round(corr,2)), fontsize=10)
#        else:
#            plt.text(-0.025, 0.025, 'Slope='+str(np.round(slp,2)), fontsize=10)
#            plt.text(-0.025, 0.02, 'Corr='+str(np.round(corr,2)), fontsize=10)
#        plt.gca().spines['right'].set_visible(False)
#        plt.gca().spines['top'].set_visible(False)
#        plt.gca().set_aspect('equal')
##    plt.suptitle("LIF-Sampling Match for Natural Images for " + str(neurons) + " neurons")
#        
#
#
#
#
#
#
#
#
#
#
##%%
##    plt.figure()
##    for m in range(len(model_set)):
##        
##        if m==0:
##            colors = 'green'
##        elif m==1:
##            colors = 'blue'
##        elif m==2:
##            colors = 'red'  
##            
##        neurons = neurons_set[nn]
##        model = model_set[m]
##        params = load_simulation_parameters(neurons,model)
##        sample_hertz = 1.0/(params.sampling_bin_s)
##        
##        Sampling_marginal_prob = np.load(path + 'Sampling_marginal_probGratingImages' + str(sf_case) + '_' + str(contrast) + str(neurons) + '_' + model + '.npy')
##        sampling_p_marg = Sampling_marginal_prob[im_selected,:]
##        Sampling_pairwise_joint_prob = np.load(path + 'Sampling_pairwise_joint_probGratingImages' + str(sf_case) + '_' + str(contrast) + str(neurons) + '_' + model + '.npy')
##        sampling_p_pairwise = Sampling_pairwise_joint_prob[im_selected,:]
##        
##        LIF_marginal_prob = np.load(path + 'LIF_marginal_probGratingImages' + str(sf_case) + '_' + str(contrast) + str(neurons) + '_' + model + '.npy')
##        implied_p_marg = LIF_marginal_prob[im_selected,:]
##        LIF_pairwise_joint_prob = np.load(path + 'LIF_pairwise_joint_probGratingImages' + str(sf_case) + '_' + str(contrast) + str(neurons) + '_' + model + '.npy')            
##        implied_p_pairwise = LIF_pairwise_joint_prob[im_selected,:]
##
##        plt.subplot(2,3,4+m)
##        plt.scatter(sampling_p_pairwise, implied_p_pairwise, s=50, color=colors, edgecolors='k')
##        plt.plot(np.linspace(0,1 + 0.1,100),np.linspace(0,1 + 0.1,100),'k')
##        plt.xlim(-0.01,1.01)
##        plt.ylim(-0.01,1.01)
##        plt.xlabel("Sample pairwise joint for Grating Images")
##        plt.ylabel("Implied pairwise joint for Grating Images")
##        plt.axis('tight')
##        
##        subplot(2,3,1+m)
##        plt.scatter(sampling_p_marg, implied_p_marg, s=50, color=colors, edgecolors='k')
##        plt.plot(np.linspace(0,1 + 0.1,100),np.linspace(0,1 + 0.1,100),'k')
##        plt.xlim(-0.01,1.01)
##        plt.ylim(-0.01,1.01)
##        plt.xlabel("Sample marginal for Grating Images")
##        plt.ylabel("Implied marginal for Grating Images")
##        plt.axis('tight')
##    plt.suptitle("LIF-Sampling Match for Grating Images for " + str(neurons) + " neurons")
##   
