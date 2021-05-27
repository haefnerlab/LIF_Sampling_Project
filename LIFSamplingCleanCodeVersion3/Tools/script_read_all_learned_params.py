#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 00:23:33 2020

@author: achattoraj
"""
## Script to read from learning performed by Learning Bronschein code of Bornschein et al. 
## Save the PFs and sigma and pi. 
## Change fl to indicate result filenamestoring all parameters in .h5 format
## sv_name, sv_name_sig, sv_name_pi are all names used to save PFs, sigma and pi respectively 
## corresponding to particular fl file.

import h5py
import numpy as np 
import matplotlib.pyplot as plt
fl = 'result_overcomplete.h5'
sv_name = 'weights_128_0.00.npy'
sv_name_sig = 'sigma_128_0.00.npy'
sv_name_pi = 'pi_128_0.00.npy'
sv_name_sig_all = 'coverging_sigma_128_0.00.npy'
sv_name_pi_all = 'coverging_pi_128_0.00.npy'
#
#fl = 'result_overcomplete_noisy.h5'
#sv_name = 'weights_128_0.05.npy'
#sv_name_sig = 'sigma_128_0.05.npy'
#sv_name_pi = 'pi_128_0.05.npy'
#sv_name_sig_all = 'coverging_sigma_128_0.05.npy'
#sv_name_pi_all = 'coverging_pi_128_0.05.npy'


#fl = 'result_complete.h5'
#sv_name = 'weights_64_0.00.npy'
#sv_name_sig = 'sigma_64_0.00.npy'
#sv_name_pi = 'pi_64_0.00.npy'
#sv_name_sig_all = 'coverging_sigma_64_0.00.npy'
#sv_name_pi_all = 'coverging_pi_64_0.00.npy'

#fl = 'result_complete_noisy.h5'
#sv_name = 'weights_64_0.05.npy'
#sv_name_sig = 'sigma_64_0.05.npy'
#sv_name_pi = 'pi_64_0.05.npy'
#sv_name_sig_all = 'coverging_sigma_64_0.05.npy'
#sv_name_pi_all = 'coverging_pi_64_0.05.npy'

pix = 8

#%% 
##Getting parameters

filename = '../Data/' + fl
f = h5py.File(filename, 'r')
temp_wt = f['W']
PFs = np.squeeze(temp_wt[-1,:,:])
PFs = np.reshape(PFs,(np.shape(PFs)[0],pix,pix))
pi = f['pi'][-1]
sigma = f['sigma'][-1]

print("Pi = " + str(pi))
print("Sigma = " + str(sigma))
plt.figure()
for i in range(np.shape(PFs)[0]):
    plt.subplot(8,int(np.shape(PFs)[0]/8),i+1)
    plt.imshow(np.squeeze(PFs[i,:,:]),cmap='gray')
    plt.axis('off')
plt.show()

plt.figure()
plt.subplot(1,2,1)
plt.plot(range(np.shape(f['pi'])[0]),f['pi'],'-k')
plt.xlabel('Iterations')
plt.ylabel('$\pi$')
plt.subplot(1,2,2)
plt.plot(range(np.shape(f['sigma'])[0]),f['pi'],'-k')
plt.xlabel('Iterations')
plt.ylabel('$\sigma$')

np.save('../Data/' + sv_name,PFs)
np.save('../Data/' + sv_name_sig,sigma)
np.save('../Data/' + sv_name_pi,pi)
np.save('../Data/' + sv_name_sig_all,f['sigma'])
np.save('../Data/' + sv_name_pi_all,f['pi'])
    