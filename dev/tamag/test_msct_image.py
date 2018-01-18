#!/usr/bin/env python

import os, sys

# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, "scripts"))
sys.path.append(path_sct + '/dev/tamag')

import nibabel
from msct_image import Image

PATH_OUTPUT= '/Users/tamag/data/data_template/Results_template' #folder where you want the results to be stored
PATH_INFO = '/Users/tamag/data/data_template/info/template_subjects/T1'  # to be replaced by URL from github


import sct_utils as sct
from numpy import asarray, array
from msct_smooth import smoothing_window
from msct_register_regularized import register_images,register_seg,generate_warping_field
from msct_image import Image
from scipy.ndimage.filters import laplace

#
#
# # Register seg to create first warping field and apply it
# print '\nUsing register_seg with centerlines to create first warping field and applying it...'
# x_disp, y_disp = register_seg('centerline_T1.nii.gz', 'centerline_T2.nii.gz')
# x_disp_a = asarray(x_disp)
# y_disp_a = asarray(y_disp)
# x_disp_smooth = smoothing_window(x_disp_a, window_len=31, window='hanning')
# y_disp_smooth = smoothing_window(y_disp_a, window_len=31, window='hanning')
#
# generate_warping_field('centerline_T2.nii.gz', x_disp_smooth, y_disp_smooth, fname='warping_field_seg.nii.gz')
# sct.run('sct_apply_transfo -i data_RPI_registered.nii.gz -d data_T2_RPI.nii.gz -w warping_field_seg.nii.gz -o data_RPI_registered_reg1.nii.gz -x spline')
#
#
# # Register_image to create second warping field and apply it
# print'\nUsing register_image with images to create second warping field and applying it...'
# x_disp_2, y_disp_2 = register_images('data_RPI_registered_reg1.nii.gz', 'data_T2_RPI.nii.gz')
# x_disp_2_a = asarray(x_disp_2)
# y_disp_2_a = asarray(y_disp_2)
# x_disp_2_smooth = smoothing_window(x_disp_2_a, window_len=31, window='hanning')
# y_disp_2_smooth = smoothing_window(y_disp_2_a, window_len=31, window='hanning')
#
# generate_warping_field('data_T2_RPI.nii.gz', x_disp_2_smooth, y_disp_2_smooth, fname='warping_field_im_trans.nii.gz')
# sct.run('sct_apply_transfo -i data_RPI_registered_reg1.nii.gz -d data_T2_RPI.nii.gz -w warping_field_im_trans.nii.gz -o data_RPI_registered_reg2.nii.gz -x spline')


f_1 = "/Users/tamag/data/data_template/independant_templates/Results_magma/t2_avg_RPI.nii.gz"
f_2 = "/Users/tamag/data/data_template/independant_templates/Results_magma/t1_avg.independent_RPI_reg1_unpad.nii.gz"
f_3 = "/Users/tamag/data/data_template/independant_templates/Results_magma/t1_avg.independent_RPI.nii.gz"

os.chdir("/Users/tamag/data/data_template/independant_templates/Results_magma")

im_1 = Image(f_1)
im_2 = Image(f_2)

data_1 = im_1.data

coord_test1 = [[1,1,1]]
coord_test = [[1,1,1],[2,2,2],[3,3,3]]

coordi_phys = im_1.transfo_pix2phys(coordi=coord_test)
coordi_pix = im_1.transfo_phys2pix(coordi = coordi_phys)
bla

# im_3 = nibabel.load(f_3)
# data_3 = im_3.get_data()
# hdr_3 = im_3.get_header()
#
# data_f = data_3 - laplace(data_3)
#
# img_f = nibabel.Nifti1Image(data_f, None, hdr_3)
# nibabel.save(img_f, "rehauss.nii.gz")