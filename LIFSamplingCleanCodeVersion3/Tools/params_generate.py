#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 00:29:49 2018

@author: achattoraj
"""
import numpy as np
class params_generate(object):
    def __init__(self, N, G, prior_prob=0.1, sigma_I=0.5, duration=1000,
                 sampling_bin_ms = 5.0, membrane_noise = 0.0, photo_noise=0.0, photo_noise_refresh_ms=10, rec_noise=0.0):
        self.G = G
        ## Initialize other variables of interest
        self.N = N  # number of neurons
        self.sampling_bin_ms = 5.0 # sampling bin size in ms
        self.sampling_bin_s = sampling_bin_ms/ 1000.0
        self.duration = duration # duration in ms for which the model simulation runs
        self.pix = G.shape[0] # dimension of the PF vector
        self.prior = prior_prob # initial firing probability of each neuron
        self.sigma_I = sigma_I # pixel_wise noise
        self.R = -np.dot(np.transpose(G), G) # Recurrent connection matrix
        self.membrane_noise = membrane_noise # Noise in membrane of neurons
        self.photo_noise = photo_noise # Noise in photoreceptors making image stochastic
        self.rec_noise = rec_noise # Noise in photoreceptors making image stochastic
        self.photo_noise_refresh_ms = photo_noise_refresh_ms # Rate at which noise gets added to image  

        