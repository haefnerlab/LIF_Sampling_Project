
import numpy as np
import matplotlib.pyplot as plt 
def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    return A[mask]

def show_PFs(filename):
    PFs = np.load('../Data/' + filename + '.npy')
    N = np.shape(PFs)[0]
    PFs = np.reshape(PFs,(N,8,8))
    if N==64:
		k = 8
    if(N==128):
		k = 16
    shp = np.shape(PFs)
    G = np.transpose(np.reshape(PFs,(shp[0],shp[1]*shp[2])))
    R = np.dot(G.T ,G)
    R_up = upper_tri_masking(R)

#    plt.figure()
#    for i in range(N):
#        plt.subplot(8,k,i+1)
#        plt.imshow(PFs[i,:,:],cmap='gray')
#        plt.axis('off')
#    
#    
#    plt.figure()
#    plt.subplot(1,2,1)
#    plt.hist(R_up.flatten(),100)
#    plt.title('Recurrent Synapses')
#    plt.subplot(1,2,2)
#    plt.hist(G.flatten(),100)
#    plt.title('Feedforward Synapses')
#    
#    print(len(G.flatten()))
#    print(len(R.flatten()))
#    print(len(R_up.flatten()))
    
    return PFs, G.flatten(), R_up.flatten()
    
    
    