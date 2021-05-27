#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import h5py as h5
import numpy as np
from sys import argv, exit

if len(argv) < 2:
    print("Usage: python extract_vanhateren_to_bsc.py path/to/vanhateren_iml/ [npatches] [patchsize]")
    exit(1)

images_dataset = argv[1]
images = [name for name in os.listdir(images_dataset) if name[-4:] == ".iml"]
(im_height, im_width) = (1024, 1536)
im_dtype = 'uint16'

n_patches = 1000 if len(argv) < 2 else int(argv[2])
size = 16 if len(argv) < 3 else int(argv[3])
patch_size = (size, size)
dest_name = "patches_%dx%d_%d" % (patch_size + (n_patches,))
dest_file = h5.File(dest_name + ".tmp.h5", "w")

idxs = np.zeros(shape=(n_patches, 3), dtype='int32')
idxs[:, 0] = np.random.randint(len(images), size=(n_patches,))  # which image
idxs[:, 1] = np.random.randint(im_height - size, size=(n_patches,))  # bottom edge
idxs[:, 2] = np.random.randint(im_width - size, size=(n_patches,))  # left edge

# Sort idxs so that same images are adjacent
idxs = idxs[idxs[:, 0].argsort()]


def read_image(iml_filename):
    with open(iml_filename, "rb") as f:
        data = np.fromfile(f, dtype=im_dtype)
        data.byteswap(True)
        return data.reshape((im_height, im_width)).astype('float32') / np.iinfo(im_dtype).max


try:
    patches_array = dest_file.require_dataset("patches", (n_patches,) + patch_size, dtype='f')

    last_im_idx = -1
    for i in range(n_patches):
        im_idx = idxs[i, 0]

        if im_idx != last_im_idx:
            last_im_idx = im_idx
            image = read_image(os.path.join(images_dataset, images[im_idx]))

        bottom, left = idxs[i, 1:]
        patches_array[i, :, :] = image[bottom:bottom + size, left:left + size]

    dest_file.close()
    os.rename(dest_name + ".tmp.h5", dest_name + ".h5")
except Exception as e:
    os.remove(dest_name + ".tmp.h5")
    raise(e)
