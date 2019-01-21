# This is script will be moved elsewhere (not supposed to be present in sct)
# converts image to nii

import matplotlib.image as mpimg
from nibabel import load, Nifti1Image, save
import numpy as np
import os

os.chdir("/home/nicolas/Sample_images")
img=mpimg.imread('test2.png')
imgnif = Nifti1Image(img, np.eye(4))
save(imgnif, "test2.nii")