#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 12:55:15 2018

@author: achattoraj
"""

"""
Created on Wed Oct 10 07:41:12 2018

@author: achattoraj
"""

import h5py
import numpy as np 
import matplotlib.pyplot as plt
def load_grating_images(dimension):
    filename = '../Data/Gratings-' + str(dimension) + '-zca-norm.h5'
    f = h5py.File(filename, 'r')
    pix = dimension * dimension
    # Get the data
    a_group_key = 'patches'
    Gratings = np.array(f[a_group_key])
    angles = np.load('../Data/Grating_angles.npy')
    phases = np.load('../Data/Grating_phases.npy')
    sf_cases = np.load('../Data/Grating_sf.npy')
    Gratings = np.reshape(Gratings,(len(sf_cases),len(angles),len(phases),pix))
    return Gratings
    
#%% Test gratings generated
angles = np.load('../Data/Grating_angles.npy')
phases = np.load('../Data/Grating_phases.npy')
sf_cases = np.load('../Data/Grating_sf.npy')
dimension = 8
PreprocessedGratings = load_grating_images(dimension)
np.save('../Data/PreprocessedGratings-8-zca-norm.npy',PreprocessedGratings) 

plt.figure()
for i in range(len(sf_cases)):
    plt.subplot(1,len(sf_cases),i+1)
    plt.imshow(np.reshape(PreprocessedGratings[i,0,0,:],(8,8)),cmap='gray', interpolation='none')
    plt.axis('off')
plt.suptitle("Testing spatial frequency changes \nfor preprocessed gratings")

plt.figure()
for i in range(len(phases)):
    plt.subplot(3,7,i+1)
    plt.imshow(np.reshape(PreprocessedGratings[0,0,i,:],(8,8)),cmap='gray', interpolation='none')
    plt.axis('off')
plt.suptitle("Testing phase changes \nfor preprocessed gratings")

plt.figure()
for i in range(len(angles)):
    plt.subplot(3,7,i+1)
    plt.imshow(np.reshape(PreprocessedGratings[0,i,0,:],(8,8)),cmap='gray', interpolation='none')
    plt.axis('off')
plt.suptitle("Testing angle changes \nfor preprocessed gratings")