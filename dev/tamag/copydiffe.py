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

path = '/Users/tamag/data/data_template/info/template_subjects'
os.chdir(path)

status, output = sct.run('sct_label_utils -i labels_vertebral.nii.gz -t display-voxel')
nb = output.find('notation')
int_nb = nb + 10
labels = output[int_nb:]

bla


