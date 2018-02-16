#!/usr/bin/env python

import sys
import numpy as np
from time import time
import nibabel as nib
from dipy.denoise.nlmeans import nlmeans
import os


import matplotlib.pyplot as plt

#os.chdir('/Users/tamag/data/original_data/full/errsm_03/t2')
os.chdir('/Users/tamag/data/sct_testing_data/data/t2')

#os.remove('/Users/tamag/.dipy/sherbrooke_3shell')

#img = nib.load('data.nii.gz')
img = nib.load('t2.nii.gz')

data = img.get_data()
aff = img.get_affine()

mask = data[:, :, :] > 80

data = data[:, :, :]

print("vol size", data.shape)

t = time()

sigma = np.std(data[~mask])

den = nlmeans(data, sigma=sigma, mask=mask)

print("total time", time() - t)
print("vol size", den.shape)


axial_middle = data.shape[2] / 2

before = data[:, :, axial_middle].T
after = den[:, :, axial_middle].T
difference = np.absolute(after.astype('f8') - before.astype('f8'))
difference[~mask[:, :, axial_middle].T] = 0

fig, ax = plt.subplots(1, 3)
ax[0].imshow(before, cmap='gray', origin='lower')
ax[0].set_title('before')
ax[1].imshow(after, cmap='gray', origin='lower')
ax[1].set_title('after')
ax[2].imshow(difference, cmap='gray', origin='lower')
ax[2].set_title('difference')
for i in range(3):
    ax[i].set_axis_off()

plt.show()
plt.savefig('denoised_S0.png', bbox_inches='tight')