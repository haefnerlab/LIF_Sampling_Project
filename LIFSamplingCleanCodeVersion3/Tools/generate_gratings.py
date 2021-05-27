# Function to generate grating patches
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from numpy import exp,cos,mean,sin,transpose

def grating(x,y,k,phi,theta=0):
    return np.sin(phi+k*(y*np.cos(theta)+x*np.sin(theta)))

def generate_Im(angles,sf,phase,pix_fixed = 0):
    if pix_fixed == 0:
        X1 = np.arange(-5, 5, 0.2)
        Y1 = np.arange(-5, 5, 0.2)
        X, Y = np.meshgrid(X1, Y1)
        pix = len(X1) * len(X1)    
        num_ph = len(phase)
        Images = np.zeros((len(angles),num_ph,pix))
        for i in range(len(angles)):
            for j in range(num_ph):
                Z = grating(X,Y,sf,phase[j],angles[i])
                Images[i,j,:] = np.squeeze(Z.reshape(pix,1)) 
                Images[i,j,:] = Images[i,j,:]/np.sqrt(np.dot(Images[i,j,:].T,Images[i,j,:])) 
        return Images
    else:
        isf = 3
        n = np.sqrt(pix_fixed)
        X1 = np.arange(1, (1+(2*(n)/(isf-1))), 2/(isf-1))
        Y1 = np.arange(1, (1+(2*(n)/(isf-1))), 2/(isf-1))
        X, Y = np.meshgrid(X1, Y1)
        pix = pix_fixed 
        num_ph = len(phase)
        Images = np.zeros((len(angles),num_ph,pix))
        for i in range(len(angles)):
            for j in range(num_ph):
                Z = grating(X,Y,sf,phase[j],angles[i])
                Images[i,j,:] = np.squeeze(Z.reshape(pix,1)) 
                Images[i,j,:] = Images[i,j,:]/np.sqrt(np.dot(Images[i,j,:].T,Images[i,j,:])) 
        return Images

#%% Generates and saves gratings
pix = 64 # Number of pixels in grating
sf_cases = [np.pi/2, np.pi/4, np.pi/6, np.pi/8, np.pi/12] # spatial frequency of gratings
angles = np.linspace(0,np.pi,21) #the angles of gratings
phases = np.linspace(0,np.pi,21) #the phases of gratings
np.save('../Data/Grating_angles.npy',angles)
np.save('../Data/Grating_phases.npy',phases)
np.save('../Data/Grating_sf.npy',sf_cases)

Gratings = np.zeros((len(sf_cases),len(angles),len(phases),pix))
for i in range(len(sf_cases)):
    Gratings[i,:,:,:] = generate_Im(angles,sf_cases[i],phases,pix)
np.save('../Data/Gratings.npy',Gratings)

#%% Test gratings generated
angles = np.load('../Data/Grating_angles.npy')
phases = np.load('../Data/Grating_phases.npy')
sf_cases = np.load('../Data/Grating_sf.npy')
Gratings = np.load('../Data/Gratings.npy')

plt.figure()
for i in range(len(sf_cases)):
    plt.subplot(1,len(sf_cases),i+1)
    plt.imshow(np.reshape(Gratings[i,0,0,:],(8,8)),cmap='gray', interpolation='none')
    plt.axis('off')
plt.suptitle("Testing spatial frequency changes")

plt.figure()
for i in range(len(phases)):
    plt.subplot(3,7,i+1)
    plt.imshow(np.reshape(Gratings[0,0,i,:],(8,8)),cmap='gray', interpolation='none')
    plt.axis('off')
plt.suptitle("Testing phase changes")

plt.figure()
for i in range(len(angles)):
    plt.subplot(3,7,i+1)
    plt.imshow(np.reshape(Gratings[0,i,0,:],(8,8)),cmap='gray', interpolation='none')
    plt.axis('off')
plt.suptitle("Testing angle changes")