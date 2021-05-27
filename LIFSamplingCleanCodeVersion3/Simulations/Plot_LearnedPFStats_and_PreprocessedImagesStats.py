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
import h5py

from Tools.load_params import *
from Tools.get_PFstats import *

neurons_set = [128]#, 64]
pix = 64 #image pixels
dimension = 8 #sqrt(image pixels)

#%% Plots related to learned parameters
for nn in range(len(neurons_set)):
    filename = 'weights_'+ str(neurons_set[nn]) +'_0.00'
    PFs, FF, Rec = show_PFs(filename)
    k = int(128/8) ##int(neurons_set[nn]/8)
    sv_name_sig_all_NN = 'coverging_sigma_' + str(neurons_set[nn]) + '_0.00.npy'
    sv_name_pi_all_NN = 'coverging_pi_' + str(neurons_set[nn]) + '_0.00.npy'
    converging_sig_NN = np.load('../Data/' + sv_name_sig_all_NN)
    converging_pi_NN = np.load('../Data/' + sv_name_pi_all_NN)
    sv_name_sig_all_IN = 'coverging_sigma_' + str(neurons_set[nn]) + '_0.05.npy'
    sv_name_pi_all_IN = 'coverging_pi_' + str(neurons_set[nn]) + '_0.05.npy'
    converging_sig_IN = np.load('../Data/' + sv_name_sig_all_IN)
    converging_pi_IN = np.load('../Data/' + sv_name_pi_all_IN)
        
    plt.figure(figsize=(10,20))
    for i in range(neurons_set[nn]):
        plt.subplot(8,k,i+1)
        plt.imshow(PFs[i,:,:],cmap='gray')
        plt.axis('off')
    plt.suptitle('PFs for ' + str(neurons_set[nn]) + ' neurons')
    
    plt.figure(figsize=(10,20))
    plt.subplot(1,2,2)
#    sns.distplot(Rec, hist=True, fit=norm, kde=False, norm_hist=True, 
#             bins=8, color = 'black', 
#             hist_kws={'color': 'gray','edgecolor':'black'},
#             kde_kws={'linewidth': 4})
    sns.distplot(Rec, hist=True, kde=True, norm_hist=True,  
             bins=100, color = 'black', 
             hist_kws={'color': 'gray','edgecolor':'black'},
             kde_kws={'linewidth': 3})
    plt.yticks([])
    plt.xlim([-1,1])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.xticks([-1,0.0,1]) 
    plt.xticks(fontsize=20,fontweight = 'bold')
    plt.title('Recurrent Synapses for ' + str(neurons_set[nn]) + ' neurons')
    plt.subplot(1,2,1)
#    sns.distplot(FF, hist=True, fit=norm, kde=False, norm_hist=True,
#             bins=7, color = 'black', 
#             hist_kws={'color': 'gray','edgecolor':'black'},
#             kde_kws={'linewidth': 4})
    sns.distplot(FF, hist=True,  kde=True, norm_hist=True, 
             bins=30, color = 'black', 
             hist_kws={'color': 'gray','edgecolor':'black'},
             kde_kws={'linewidth': 3})
    plt.yticks([])
    plt.xlim([-1,1])
    
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.xticks([-1,0.0,1]) 
    plt.xticks(fontsize=20,fontweight = 'bold')
    plt.title('Feedforward Synapses for ' + str(neurons_set[nn]) + ' neurons')
    plt.suptitle('FF and Rec for ' + str(neurons_set[nn]) + ' neurons')
    
    plt.figure(figsize=(10,20))
    plt.subplot(2,1,1)
    plt.plot(range(np.shape(converging_pi_NN)[0]),converging_pi_NN,'-og',label = '$\pi$ for $\Theta_{\mathrm{NN}}$')
    plt.plot(range(np.shape(converging_pi_IN)[0]),converging_pi_IN,'-ob',label = '$\pi$ for $\Theta_{\mathrm{IN}}$')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Iterations',fontsize=20)
    plt.ylabel('$\pi$ for ' + str(neurons_set[nn]) + ' neurons',fontsize=20)
    plt.legend(fontsize=20,loc='best')
    plt.subplot(2,1,2)
    plt.plot(range(np.shape(converging_sig_NN)[0]),converging_sig_NN,'-og',label = '$\sigma$ for $\Theta_{\mathrm{NN}}$')
    plt.plot(range(np.shape(converging_sig_IN)[0]),converging_sig_IN,'-ob',label = '$\sigma$ for $\Theta_{\mathrm{IN}}$')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Iterations',fontsize=20)
    plt.ylabel('$\sigma$ for ' + str(neurons_set[nn]) + ' neurons',fontsize=20)
    plt.suptitle('Convergence of $\pi$ and $\sigma$ for ' + str(neurons_set[nn]) + ' neurons')
    plt.legend(fontsize=20,loc='best')
    

#%% Plots related to image preprocessing
filename1 = '../Data/patches-'+ str(dimension) +'-zca-norm.h5'
filename2 = '../Data/patches-'+ str(dimension) +'.h5'
f_processed = h5py.File(filename1, 'r')
f_original = h5py.File(filename2, 'r')
grating_processed = np.load('../Data/PreprocessedGratings-8-zca-norm.npy')
grating_original = np.load('../Data/Gratings.npy')
selected_indx_nat = np.array([0, 5, 10, 15]) +10000
selected_indx = 5
sf_chosen = 0
phase_chosen = 5
contrasts = np.load('../Data/Contrasts.npy')
a_group_key = 'patches'
data_processed = np.array(f_processed[a_group_key])
data_original = np.array(f_original[a_group_key])
#U = f['eig_vec']
S = f_processed['eig_val']
epsilon = 1e-3
trans_mat = np.load('../Data/ZCA_filter_NaturalImages.npy')
trans_mat = np.transpose(trans_mat)

#plt.figure()
#plt.plot(range(np.shape(S)[0]),S,'o')
#plt.ylabel('Eigen Values')
#plt.title('Eigen Values for 8 x 8 natural image patches')

#plt.figure()
#plt.imshow(trans_mat,cmap='gray')
#plt.axis('image')
#plt.axis('off')
#plt.colorbar()
#plt.suptitle('ZCA matrix for 8 x 8 natural image patches')

fig,axes = plt.subplots(nrows=2,ncols=16,figsize=(5,2))
k = 0
for i in range(2):
    for j in range(16):
        im = axes[i][j].imshow(np.reshape(trans_mat[:,k],(8,8)),cmap='gray',vmin=-20, vmax = 20)
        axes[i][j].axis('image')
        axes[i][j].axis('off')
        k = k + 1
plt.subplots_adjust(wspace=0.1,hspace=0.01)
fig.colorbar(im,ax=axes.ravel().tolist(),shrink=1.0)
plt.suptitle('Center-surround ZCA filters')

fig1,axes1 = plt.subplots(nrows=2,ncols=len(contrasts),figsize=(20,5))
for i in range(len(contrasts)):
    im = axes1[0][i].imshow((np.reshape(grating_original[sf_chosen,selected_indx,phase_chosen,:],(8,8)) * contrasts[i]),cmap='gray',vmin=-1,vmax=1)#    plt.imshow(np.reshape(data_original[sf_chosen,selected_indx,phase_chosen,:],(8,8)),cmap='gray')
    axes1[0][i].axis('image')
    axes1[0][i].axis('off')
    axes1[0][i].set_title('Original Grating \nx Contrast = '+ str(contrasts[i]),fontsize=10)   
    im = axes1[1][i].imshow((np.reshape(contrasts[i]*grating_processed[sf_chosen,selected_indx,phase_chosen,:],(8,8))),cmap='gray',vmin=-1,vmax=1)#    plt.imshow(np.reshape(data_original[sf_chosen,selected_indx,phase_chosen,:],(8,8)),cmap='gray')
    axes1[1][i].axis('image')
    axes1[1][i].axis('off')
    axes1[1][i].set_title('Processed Grating \nx Contrast = '+ str(contrasts[i]),fontsize=10)     
plt.subplots_adjust(wspace=0.65)
fig1.colorbar(im,ax=axes1.ravel().tolist(),shrink=1.0)
plt.suptitle('Comparing original and processed grating images')

fig2,axes2 = plt.subplots(nrows=2,ncols=len(selected_indx_nat),figsize=(20,5))
for i in range(len(selected_indx_nat)):
    im = axes2[0][i].imshow(np.squeeze(data_original[selected_indx_nat[i],:,:]),cmap='gray',vmin=-1,vmax=1)#    plt.imshow(np.reshape(data_original[sf_chosen,selected_indx,phase_chosen,:],(8,8)),cmap='gray')
    axes2[0][i].axis('image')
    axes2[0][i].axis('off')
    axes2[0][i].set_title('Original natural image \npatch number ' + str(i+1),fontsize=10)   
    im = axes2[1][i].imshow(np.squeeze(data_processed[selected_indx_nat[i],:,:]),cmap='gray',vmin=-1,vmax=1)#    plt.imshow(np.reshape(data_original[sf_chosen,selected_indx,phase_chosen,:],(8,8)),cmap='gray')
    axes2[1][i].axis('image')
    axes2[1][i].axis('off')
    axes2[1][i].set_title('Processed natural image \npatch number ' + str(i+1),fontsize=10)     
plt.subplots_adjust(wspace=0.65)
fig2.colorbar(im,ax=axes2.ravel().tolist(),shrink=1.0)
plt.suptitle('Comparing original and processed 8 x 8 natural image images')