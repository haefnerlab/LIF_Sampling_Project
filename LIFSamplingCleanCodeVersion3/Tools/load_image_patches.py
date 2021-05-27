## loads and returns preprocessed natural image patches
import h5py
import numpy as np 
def load_natural_image_patches(dimension):
    filename = '../Data/patches-'+ str(dimension) +'-zca-norm.h5'
    f = h5py.File(filename, 'r')
    # Get the data
    a_group_key = 'patches'
    data = np.array(f[a_group_key])
    return data
#%%
dimension = 8
PreprocessedImagePatches = load_natural_image_patches(dimension)
np.save('../Data/PreprocessedImagePatches-8-zca-norm.npy',PreprocessedImagePatches) 

