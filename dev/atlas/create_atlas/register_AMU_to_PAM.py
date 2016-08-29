#!/usr/bin/env python
#########################################################################################
#
# Register AMU_wm atlas to the PAM_cord segmentation, and also add missing slices at the top and bottom.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2016 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: jcohen@polymtl.ca
# Created: 2016-08-26
#
# About the license: see the file LICENSE.TXT
#########################################################################################


# Import common Python libraries
import os
import sys
import numpy as np
# append path that contains scripts, to be able to load modules
path_script = os.path.dirname(__file__)
path_sct = os.path.abspath(path_script+'/../../../')
sys.path.append(path_sct+'/scripts')
import sct_utils as sct
from msct_image import Image


# parameters
fname_wm = '/Users/julien/Dropbox/Public/sct/PAM50/template/PAM50_wm.nii.gz'
fname_gm = '/Users/julien/Dropbox/Public/sct/PAM50/template/PAM50_gm.nii.gz'
fname_cord = '/Users/julien/data/sct_dev/PAM50/template/PAM50_cord.nii.gz'

# create temporary folder
path_tmp = sct.tmp_create()

# go to temp folder
os.chdir(path_tmp)

# open volumes
im_wm = Image(fname_wm)
data_wm = im_wm.data
im_gm = Image(fname_gm)
data_gm = im_gm.data
im_cord = Image(fname_cord)
data_cord = im_cord.data
dim = im_cord.dim

# sum wm/gm
data_wmgm = data_wm + data_gm

# get min/max z slices from wm/gm
zsum = np.sum(np.sum(data_wmgm, 0), 0)
zmin_wm = np.min(np.nonzero(zsum))
zmax_wm = np.max(np.nonzero(zsum))

# get min/max z slices from cord
zsum = np.sum(np.sum(data_cord, 0), 0)
zmin_cord = np.min(np.nonzero(zsum))
zmax_cord = np.max(np.nonzero(zsum))

# duplicate WM and GM atlas towards the top and bottom slices to match the cord template
# bottom slices
for iz in range(zmin_cord, zmin_wm):
    data_wm[:, :, iz] = data_wm[:, :, zmin_wm]
    data_gm[:, :, iz] = data_gm[:, :, zmin_wm]
# top slices
for iz in range(zmax_wm, zmax_cord):
    data_wm[:, :, iz] = data_wm[:, :, zmax_wm]
    data_gm[:, :, iz] = data_gm[:, :, zmax_wm]

# save modified atlases
im_wm.setFileName('wm_ext.nii.gz')
im_wm.data = data_wm
im_wm.save()
im_gm.setFileName('gm_ext.nii.gz')
im_gm.data = data_gm
im_gm.save()

# sum modified wm/gm
data_wmgm = data_wm + data_gm

# save wm/gm
im_wm.setFileName('wmgm_ext.nii.gz')
im_wm.data = data_wmgm
im_wm.save()

# register wmgm --> cord
sct.run('cp '+fname_cord+' cord.nii.gz')
# sct.run('sct_maths -i '+fname_cord+' -laplacian 1 -o cord.nii.gz')
sct.run('sct_maths -i wmgm_ext.nii.gz -bin 0.5 -o wmgm_ext.nii.gz')
# sct.run('sct_maths -i wmgm_ext.nii.gz -laplacian 1 -o wmgm_ext.nii.gz')
#sct.run('sct_register_multimodal -i wmgm_ext.nii.gz -d cord.nii.gz -iseg wmgm_ext.nii.gz -dseg cord.nii.gz -param step=1,type=im,algo=bsplinesyn,iter=10,slicewise=1,metric=MeanSquares -x linear -r 0')
sct.run('sct_register_multimodal -i wmgm_ext.nii.gz -d cord.nii.gz -iseg wmgm_ext.nii.gz -dseg cord.nii.gz -param step=1,type=im,algo=affine,iter=100,slicewise=1,metric=MeanSquares,smooth=1:step=2,type=im,algo=bsplinesyn,iter=5,slicewise=1,metric=MeanSquares,smooth=1 -x linear -r 0')
sct.run('sct_apply_transfo -i wm_ext.nii.gz -d cord.nii.gz -w warp_wmgm_ext2cord.nii.gz -x linear')

# crop below a certain point
sct.run('sct_crop_image -i wm_ext_reg.nii.gz -dim 2 -start 62 -end 1099 -b 0 -o wm_ext_reg_crop.nii.gz')

# rename new file
sct.run('mv wm_ext_reg_crop.nii.gz PAM50_wm.nii.gz')

# go back to previous folder
#os.chdir('../')

