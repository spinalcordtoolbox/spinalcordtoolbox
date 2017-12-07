#!/usr/bin/env python

import os, sys, commands

# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, "scripts"))


import nibabel

from scipy.io import loadmat
from msct_smooth import smoothing_window
from numpy import asarray, zeros
import sct_utils as sct
from math import cos, sin
import matplotlib.pyplot as plt





os.chdir('/Users/tamag/data/work_on_registration')

f_1 = '/Users/tamag/data/work_on_registration/line_0degre.nii.gz'
f_2 = ''

im_1 = nibabel.load(f_1)
nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(f_1)
data_1 = im_1.get_data()
hdr_1 = im_1.get_header()

theta = 3*0.0872664626  # 5 degres

data = zeros((nx,ny))
max_param = int(round(ny/(2*cos(theta))))

for i in range(-max_param, max_param):
    if int(round(i*sin(theta)) + nx/2) < nx and int(round(i*cos(theta)) + ny/2) < ny:
        data[int(round(i*sin(theta)) + nx/2), int(round(i*cos(theta)) + ny/2)] = 1

X,Y = data.nonzero()

# plt.plot(X,Y)
# plt.show()

img = nibabel.Nifti1Image(data, None, hdr_1)
nibabel.save(img, 'line_15degre.nii.gz')
