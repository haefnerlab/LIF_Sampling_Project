## Simulates Gibbs Sampling on neuron population
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import random

def Inference_Gibbs(params, Im, n_samples, random_init, init_samples):
    N = params.N
    G = params.G
    pixel_std = params.sigma_I
    prior = params.prior
    photo_noise = params.photo_noise 
    rec_noise = params.rec_noise
    # Flatten the image
    image = Im.flatten()
    R = -1.0 * np.dot(G.T ,G)
    pix_var = pixel_std**2
    log_prior = np.log(prior) - np.log(1 - prior)
    pixels = G.shape[0]
    x_samples = np.zeros((N, n_samples))
    prob_n = np.zeros((N, n_samples))
    ff_arr = np.zeros((N, n_samples))
    R_arr = np.zeros((N, n_samples))
    
    # No outer loop over repeats (vectorized)
    # Sample 1st value from the prior
    if (random_init==1):
        initial_s = np.random.random(N) <= prior
        x_samples[:, 0] = initial_s.astype(int)
    else:
        x_samples[:, 0] = init_samples
    
    # Loop sequentially over samples
    for s in range(1,n_samples):
        image_noisy = image +  np.random.normal(0.0,photo_noise,pixels)
        feedforward = np.dot(image_noisy, G)
        
        inp_syn_noise = np.exp(rec_noise * np.random.randn(N) - rec_noise**2/2)
        feedforward = feedforward *  inp_syn_noise 
        rec_syn_noise = np.exp(rec_noise * np.random.randn((N)) - rec_noise**2/2)
        R_new = (R * rec_syn_noise)
        
        #Copy state s-1 to state s
        x_samples[:, s] = x_samples[:, s-1]
        #Loop random permutation over cells (note: this is the same random permutation per repeat due
        #to vectorization, but different for each sample)
        for k in np.random.permutation(N):
            #Sample x_k
            M_1 = np.append(x_samples[0:k, s],x_samples[k+1:N, s]).T
            M_2 = np.append(R_new[0:k, k],R_new[k+1:N,k])
            drive = (feedforward[k] + R[k, k] / 2.0  + np.dot(M_1, M_2))/ pix_var
            ff_arr[k,s] = feedforward[k]/pix_var 
            R_arr[k,s] = (R[k, k] / 2.0  + np.dot(M_1, M_2))/ pix_var
            sigmoid_input = log_prior + drive 
            prob = sigmoid_x(sigmoid_input)
            temp = np.squeeze(np.random.rand(1) < prob  )
            x_samples[k, s] = temp.astype(int)
            prob_n[k,s] = prob
    # Returns samples, feedforward inputs, Recurrent inputs and Gibbs probabilities
    return x_samples,ff_arr,R_arr,prob_n 

from math import *
def sigmoid_x(x):
    return (1.0)/(1.0 + np.exp(-x))
