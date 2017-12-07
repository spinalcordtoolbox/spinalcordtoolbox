#!/usr/bin/env python

import os, sys, commands

# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, "scripts"))


import nibabel
from msct_image import Image

from msct_register_regularized import generate_warping_field, register_images, register_seg
from scipy.io import loadmat
from msct_smooth import smoothing_window
from numpy import asarray
from sct_register_multimodal import Paramreg
import sct_utils as sct
from math import asin
from numpy.linalg import inv


# csv_file = '/Users/tamag/data/data_template/subjects_test_T1/subjects/errsm_11/T1/tmp.150519154253/step2TxTy_poly.csv'
#
#
# with open(csv_file) as f:
#     reader = csv.reader(f)
#     count = 0
#     for row in reader:
#         count += 1
#         if count == 2:
#             print row
#
#     print reader[1]


# f = '/Users/tamag/data/data_template/subjects_test_T1/split/tmp.150522154549/transform_00000GenericAffine.mat'
# matfile = loadmat(f, struct_as_record=True)
# array_transfo = matfile['AffineTransform_double_2_2']


os.chdir('/Users/tamag/data/work_on_registration')
#os.chdir('/Users/tamag/data/data_template/subjects_test_T1/subjects/errsm_11/T1')
seg_T1 = '/Users/tamag/data/work_on_registration/centerline_T1_mask_cyl_7.nii.gz'
seg_T2 = '/Users/tamag/data/work_on_registration/centerline_T2_mask_cyl_7.nii.gz'
seg_T1_crop = '/Users/tamag/data/work_on_registration/T1_seg_crop.nii.gz'
seg_T2_crop = '/Users/tamag/data/work_on_registration/T2_seg_crop.nii.gz'

im_T2 = '/Users/tamag/data/work_on_registration/data_T2_RPI_crop.nii.gz'
im_T2_crop = '/Users/tamag/data/work_on_registration/T2_crop.nii.gz'
im_T1 = '/Users/tamag/data/work_on_registration/data_RPI_registered.nii.gz'
im_T1_crop = '/Users/tamag/data/work_on_registration/T1_crop.nii.gz'

im_d_3 = '/Users/tamag/data/work_on_registration/pad_carre.nii.gz'
im_d_5 = '/Users/tamag/data/work_on_registration/pad_carre_after_rotation.nii.gz'
mask = 'mask_cyl_50.nii.gz'

# #seg
# x_disp, y_disp = register_seg(im_i_test, im_d_test)
#
# x_disp_a = asarray(x_disp)
# y_disp_a = asarray(y_disp)
#
# x_disp_smooth = smoothing_window(x_disp_a, window_len=31, window='hanning', verbose = 2)
# y_disp_smooth = smoothing_window(y_disp_a, window_len=31, window='hanning', verbose = 2)
#
#
# generate_warping_field(seg_d_2, x_disp_smooth, y_disp_smooth, fname='warping_field_seg.nii.gz')


# #im and algo trans
# x_disp, y_disp = register_images(im_i, im_i, paramreg=Paramreg(step='0', type='im', algo='Translation', metric='MI', iter='5', shrink='1', smooth='0', gradStep='0.5'), remove_tmp_folder=0)
#
# x_disp_a = asarray(x_disp)
# y_disp_a = asarray(y_disp)
#
# x_disp_smooth = smoothing_window(x_disp_a, window_len=31, window='hanning', verbose = 2)
# y_disp_smooth = smoothing_window(y_disp_a, window_len=31, window='hanning', verbose = 2)
#
# generate_warping_field(im_i, x_disp_smooth, y_disp_smooth, fname='warping_field_im_trans.nii.gz')


# # test
# theta = [0.57]#[1.57079] #10 degres
# x_ = [0]
# y_ = [0]
# generate_warping_field(im_d_3, x_, y_, theta, center_rotation=None, fname='warping_field_15transx.nii.gz')
# sct.run('sct_apply_transfo -i '+im_d_3+' -d '+im_d_3+' -w warping_field_15transx.nii.gz -o ' + im_d_5 + ' -x nn')



# im and algo rigid
im_i = im_T1
im_d = im_T2
window_size = 31
x_disp, y_disp, theta = register_images(im_i, im_d, paramreg=Paramreg(step='0', type='im', algo='Rigid', metric='MI', iter='100', shrink='1', smooth='0', gradStep='3'), remove_tmp_folder=1)

x_disp_a = asarray(x_disp)
y_disp_a = asarray(y_disp)
theta_a = asarray(theta)

x_disp_smooth = smoothing_window(x_disp_a, window_len=window_size, window='hanning', verbose = 2)
y_disp_smooth = smoothing_window(y_disp_a, window_len=window_size, window='hanning', verbose = 2)
theta_smooth = smoothing_window(theta_a, window_len=window_size, window='hanning', verbose = 2)
theta_smooth_inv = -theta_smooth

generate_warping_field(im_d, x_disp_smooth, y_disp_smooth, theta_rot=theta_smooth, fname='warping_field_seg_rigid.nii.gz')
generate_warping_field(im_i, -x_disp_smooth, -y_disp_smooth, theta_rot=theta_smooth_inv, fname='warping_field_seg_rigid_inv.nii.gz')

sct.run('sct_apply_transfo -i '+ im_i +' -d ' +im_d+ ' -w warping_field_seg_rigid.nii.gz -o output_im_rigid.nii.gz -x nn')
sct.run('sct_apply_transfo -i '+ im_d +' -d ' +im_i+ ' -w warping_field_seg_rigid_inv.nii.gz -o output_im_rigid_inv.nii.gz -x nn')



# # im and algo affine
# im_i = im_T1
# im_d = im_T2
# window_size = 31
# x_disp, y_disp, matrix_def = register_images(im_i, im_d, paramreg=Paramreg(step='0', type='im', algo='Affine', metric='MI', iter='100', shrink='1', smooth='0', gradStep='3'), remove_tmp_folder=1)
#
#
# x_disp_a = asarray(x_disp)
# y_disp_a = asarray(y_disp)
# matrix_def_0_a = asarray([matrix_def[j][0][0] for j in range(len(matrix_def))])
# matrix_def_1_a = asarray([matrix_def[j][0][1] for j in range(len(matrix_def))])
# matrix_def_2_a = asarray([matrix_def[j][1][0] for j in range(len(matrix_def))])
# matrix_def_3_a = asarray([matrix_def[j][1][1] for j in range(len(matrix_def))])
#
# x_disp_smooth = smoothing_window(x_disp_a, window_len=window_size, window='hanning', verbose = 2)
# y_disp_smooth = smoothing_window(y_disp_a, window_len=window_size, window='hanning', verbose = 2)
# matrix_def_smooth_0 = smoothing_window(matrix_def_0_a, window_len=window_size, window='hanning', verbose = 2)
# matrix_def_smooth_1 = smoothing_window(matrix_def_1_a, window_len=window_size, window='hanning', verbose = 2)
# matrix_def_smooth_2 = smoothing_window(matrix_def_2_a, window_len=window_size, window='hanning', verbose = 2)
# matrix_def_smooth_3 = smoothing_window(matrix_def_3_a, window_len=window_size, window='hanning', verbose = 2)
# # matrix_def_smooth = [[matrix_def_smooth_0, matrix_def_smooth_1], [matrix_def_smooth_2, matrix_def_smooth_3]]
# matrix_def_smooth = [[[matrix_def_smooth_0[iz], matrix_def_smooth_1[iz]], [matrix_def_smooth_2[iz], matrix_def_smooth_3[iz]]] for iz in range(len(matrix_def_smooth_0))]
# matrix_def_smooth_inv = inv(asarray(matrix_def_smooth)).tolist()
#
# generate_warping_field(im_d, x_disp_smooth, y_disp_smooth, theta_rot=None, matrix_def=matrix_def_smooth, fname='warping_field_seg_affine.nii.gz')
# generate_warping_field(im_i, -x_disp_smooth, -y_disp_smooth, theta_rot=None, matrix_def=matrix_def_smooth_inv, fname='warping_field_seg_affine_inv.nii.gz')
#
# sct.run('sct_apply_transfo -i '+ im_i +' -d ' +im_d+ ' -w warping_field_seg_affine.nii.gz -o output_im_affine.nii.gz -x nn')
# sct.run('sct_apply_transfo -i '+ im_d +' -d ' +im_i+ ' -w warping_field_seg_affine_inv.nii.gz -o output_im_affine_inv.nii.gz -x nn')