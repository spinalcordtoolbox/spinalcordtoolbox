# This is script will be moved elsewhere (not supposed to be present in sct)
# converts image to nii

import matplotlib.image as mpimg
from nibabel import load, Nifti1Image, save
import numpy as np
import os

os.chdir("/Users/nipin_local")
img=mpimg.imread('circle2.png')
imgnif = Nifti1Image(img, np.eye(4))
save(imgnif, "circle2.nii")