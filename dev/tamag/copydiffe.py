#!/usr/bin/env python


import commands, sys, os
from glob import glob
import nibabel

# Get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# Append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')
sys.path.append(path_sct + '/dev/tamag')

from numpy import mean, append, isnan, array
import sct_utils as sct
from scipy import ndimage
from msct_register import register_slicereg2d_translation

path = '/Users/tamag/Desktop/test_register_multimodal/diff_size'
os.chdir(path)
src = 'T2_crop.nii.gz'
dest ='data_T2_RPI_crop.nii.gz'


# status, output = sct.run('sct_label_utils -i labels_vertebral.nii.gz -t display-voxel')
# nb = output.find('notation')
# int_nb = nb + 10
# labels = output[int_nb:]

register_slicereg2d_translation(dest, src, remove_temp_files=0, verbose=1)


