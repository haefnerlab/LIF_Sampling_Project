#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 09:52:44 2020

@author: achattoraj
"""
import numpy as np
from Tools.generate_gratings import *
angles = np.load('/Users/achattoraj/Desktop/Projects/LIF_Sampling_Project/LIFSamplingCleanCodeVersion1/Data/Grating_angles.npy')
phases = np.load('/Users/achattoraj/Desktop/Projects/LIF_Sampling_Project/LIFSamplingCleanCodeVersion1/Data/Grating_phases.npy')
k_str = np.array(['np.pi/2','np.pi/3','np.pi/4', 'np.pi/5', 'np.pi/8'])
#k = np.pi/8
#for i in range(len(k)):
gratings = generate_Im(angles, phases, 64)

plt.figure(5)
for j in range(len(phases)):
    subplot(7,3,j+1)
    plt.imshow(np.reshape(gratings[0,j,:],(8,8)),cmap = 'gray')

np.save('/Users/achattoraj/Desktop/Projects/LIF_Sampling_Project/LIFSamplingCleanCodeVersion1/Data/' + 'Gratings_latest5.npy', gratings)