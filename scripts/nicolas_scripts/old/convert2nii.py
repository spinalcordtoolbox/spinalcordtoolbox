# This is script will be moved elsewhere (not supposed to be present in sct)
# converts image to nii

import matplotlib.image as mpimg
from nibabel import load, Nifti1Image, save
import numpy as np
import os

img=mpimg.imread('mask_rotated110.png')
img = np.mean(img, axis=2)
imgnif = Nifti1Image(img, np.eye(4))
save(imgnif, "mask_rotated110.nii")
