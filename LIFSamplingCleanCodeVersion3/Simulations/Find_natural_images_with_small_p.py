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
neurons_set = [128]#, 64]#, 64]
pix = 64 #image pixels
dimension = 8 #sqrt(image pixels)
path = '../Data/'
NaturalImage_data = np.load('../Data/PreprocessedImagePatches-8-zca-norm.npy') # load natural image reprocessed patches 
max_elms = 100
for nn in range(len(neurons_set)):
    selected_natural_images = np.zeros((max_elms,8,8))
    selected_natural_image_indices = np.zeros(max_elms)
    k = 0
    flag = 1
    t = 0
    while flag==1:
        chosen_natural_im = t
        print('Chosen Natural Image number:' + str(chosen_natural_im))
        NaturalImage_chosen = np.squeeze(NaturalImage_data[chosen_natural_im,:,:])
        p_mx = np.zeros(len(model_set))
#        p_mn = np.zeros(len(model_set))
        p = np.zeros((len(model_set),neurons_set[nn]))
        cnt = np.zeros(len(model_set))
        for m in range(len(model_set)):
            params = load_simulation_parameters(neurons_set[nn],model_set[m])
            sample_hertz = 1.0/(params.sampling_bin_s)
            natural_image_duration = 500
            n_samples = int(natural_image_duration/params.sampling_bin_ms)
            
            random_init = 1
            init_sample = np.zeros(params.N)
            samplesNat,_,_,_ = Inference_Gibbs(params, NaturalImage_chosen, n_samples, random_init, init_sample)
            print ('Sampling Done for model ' + str(model_set[m]))
            
            p[m,:] = np.sum(samplesNat,1)/n_samples#marg_prob_compute(samplesNat,params.N)
            cnt[m] = np.sum(p[m,:]>=0.1)
#            p_mn[m] = np.sum(p[m,:]>0.1)
            p_mx[m] = np.max(p[m,:])
        print('Active for 3 models: ' + str(cnt))
        print('Found images = ' + str(k))
        t = t + 1
#        if(np.min(p_mn)>=5 and np.max(p_mx)<=0.4):
        print('Probs for 3 models: ' + str(p_mx))
        if(np.min(p_mx)>=0.1 and np.max(p_mx)<=0.5 and np.min(cnt)>=5):
            
           selected_natural_images[k,:,:] = NaturalImage_chosen
           selected_natural_image_indices[k] = t - 1
           k = k + 1
#           print('Probs for 3 models: ' + str(p_mx))
           print('Active for 3 models: ' + str(cnt))
           print('Found Natural Image number '+str(k))
           if k==max_elms:
               flag=0
        print("==================================================================================")
    filename = 'SuitableNaturalImagesNewSet' + '_' + str(neurons_set[nn]) + '.npy'
    np.save(path + filename,selected_natural_images)
    filename1 = 'SuitableNaturalImageIndicesNewSet' + '_' + str(neurons_set[nn]) + '.npy'
    np.save(path + filename1,selected_natural_image_indices)
           
           
           
           
           
           