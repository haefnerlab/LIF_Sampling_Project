import sys
sys.path.insert(0, "lib/")
sys.path.append('../pulp/utils')
sys.path.append('../pulp/preproc')
import os
import os.path
import numpy as np
import tables
#import pylab

from datalog import dlog
from autotable import AutoTable
import image as pri
import matplotlib.pyplot as plt



extracted_patch = tables.open_file('patches-20.h5','r')
preprocessed_patch = tables.open_file('patches-20-dog-norm.h5','r')


true_patches =  extracted_patch.root.patches
oversized_patches = extracted_patch.root.oversized
preprocessed_patches = preprocessed_patch.root.patches



print true_patches.shape
print oversized_patches.shape
print preprocessed_patches.shape


plt.subplot(1,3,1)
plt.imshow(true_patches[0])

plt.subplot(1,3,2)
plt.imshow(oversized_patches[0])

plt.subplot(1,3,3)
plt.imshow(preprocessed_patches[0])

plt.show()