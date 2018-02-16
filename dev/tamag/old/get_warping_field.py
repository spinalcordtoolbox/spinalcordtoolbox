#!/usr/bin/env python

# Create a warping field to register a segmentation onto another by calculating precisely the displacement.
# The segmentations can be of different size but the output segmentation must be smaller thant the input segmentation
# INPUT:
#   volume_src
#   volume_dest
#   paramreg (class inherited in sct_register_multimodal)
#   (mask)
#   fname_warp
# OUTPUT:
#   none
#   (write warping field)


import os, sys

from scipy import ndimage
from numpy import array, asarray, zeros, int8, mean, std, sqrt, convolve, hanning
from copy import copy
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.interpolate import splrep,splev
import nibabel

# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, 'scripts'))

import sct_utils as sct
from sct_orientation import get_orientation, set_orientation
from msct_image import Image

os.chdir('/Users/tamag/data/data_template/subjects_test_T1/creating_warping_field')
seg_input = nibabel.load('/Users/tamag/data/data_template/subjects_test_T1/creating_warping_field/centerline_T1_mask_cyl_7.nii.gz')
seg_output = nibabel.load('/Users/tamag/data/data_template/subjects_test_T1/creating_warping_field/centerline_T2_mask_cyl_7.nii.gz')

seg_input_data = seg_input.get_data()
seg_output_data = seg_output.get_data()
hdr_seg_input = seg_input.get_header()
hdr_seg_output = seg_output.get_header()
hdr_warp = hdr_seg_output.copy()

# ## Get physical coordinates of seg output into seg input
# # 1st: get physical coordinates of origin of seg output (f:pix to phys)
# m_p2f_output = hdr_seg_output.get_sform()
# coord_origin_output = m_p2f_output[:, 3]
#
# # 2nd: find pixel coordinates of this point in seg input (f:phys to pix)
# m_p2f_input = hdr_seg_input.get_sform()
# m_f2p_input = inv(m_p2f_output[0:3,0:3])
# coord_origin_input = array([[m_p2f_input[0, 3]],[m_p2f_input[1, 3]], [m_p2f_input[2, 3]]])
#
# # 3rd: keep zmin and zmax of interest
#
# img_input = Image('/Users/tamag/data/data_template/subjects_test_T1/creating_warping_field/centerline_T1_mask_cyl_7.nii.gz')
# #data_phys_output = img_input.transfo_pix2phys(seg_input_data)
# data_phys_output = zeros((((seg_output_data.shape[0], seg_output_data.shape[1], seg_output_data.shape[2], 3))))
# data_pix_output = img_input.tranfo_phys2pix(data_phys_output)
#
# img_input





# FUNCTION
# register_seg


x_center_of_mass_input = [0 for i in range(seg_input_data.shape[2])]
y_center_of_mass_input = [0 for i in range(seg_input_data.shape[2])]
print '\nGet center of mass of the input segmentation for each slice (corresponding to a slice int the output segmentation)...' #different if size of the two seg are different
#TO DO: select only the slices corresponding to the output segmentation
for iz in xrange(seg_output_data.shape[2]):
    x_center_of_mass_input[iz], y_center_of_mass_input[iz] = ndimage.measurements.center_of_mass(array(seg_input_data[:,:,iz]))


x_center_of_mass_output = [0 for i in range(seg_input_data.shape[2])]
y_center_of_mass_output = [0 for i in range(seg_input_data.shape[2])]
print '\nGet center of mass of the output segmentation for each slice ...'
for iz in xrange(seg_output_data.shape[2]):
    x_center_of_mass_output[iz], y_center_of_mass_output[iz] = ndimage.measurements.center_of_mass(array(seg_output_data[:,:,iz]))

x_displacement = [0 for i in range(seg_input_data.shape[2])]
y_displacement = [0 for i in range(seg_input_data.shape[2])]
print '\nGet displacement by voxel...'
for iz in xrange(seg_output_data.shape[2]):
    x_displacement[iz] = x_center_of_mass_output[iz] - x_center_of_mass_input[iz]
    y_displacement[iz] = y_center_of_mass_output[iz] - y_center_of_mass_input[iz]

    # return x_displacement,y_displacement



# Creating warping fields
# They must be of 5 dimensions



# FUNCTION
# register_image

# split volumes along z
# retrieve nz
# loop across nz
    # # set masking
    # if param.fname_mask:
    #     masking = '-x mask.nii.gz'
    # else:
    #     masking = ''
    #
    # cmd = ('isct_antsRegistration '
    #        '--dimensionality 3 '
    #        '--transform '+paramreg.steps[i_step_str].algo+'['+paramreg.steps[i_step_str].gradStep +
    #        ants_registration_params[paramreg.steps[i_step_str].algo.lower()]+'] '
    #        '--metric '+paramreg.steps[i_step_str].metric+'['+dest+','+src+',1,'+metricSize+'] '
    #        '--convergence '+paramreg.steps[i_step_str].iter+' '
    #        '--shrink-factors '+paramreg.steps[i_step_str].shrink+' '
    #        '--smoothing-sigmas '+paramreg.steps[i_step_str].smooth+'mm '
    #        '--restrict-deformation 1x1x0 '
    #        '--output --> file.txt (contains Tx,Ty)
    #        '--interpolation BSpline[3] '
    #        +masking)

# read file.txt to extract x_displacement,y_displacement
# return x_displacement,y_displacement




# FUNCTION REGULARIZE
# input: x_displacement,y_displacement
# output: x_displacement_reg,y_displacement_reg

## Implementation of a 2D fitting
z_index = [i for i in range(seg_output_data.shape[2])]

# # with 2D B-spline
# # For x
# m_x =mean(x_displacement)
# sigma_x = std(x_displacement)
# smoothing_param_x = (((m_x + sqrt(2*m_x))*(sigma_x**2))+((m_x - sqrt(2*m_x))*(sigma_x**2)))/2 #Equivalent to : m*sigma**2
# tck_x = splrep(z_index, x_displacement, s=smoothing_param_x)
# x_disp_smooth = splev(z_index, tck_x)
# # For y
# y_for_fitting = [-i for i in y_displacement]
# m_y =mean(y_for_fitting)
# sigma_y = std(y_for_fitting)
# smoothing_param_y = m_y * sigma_y**2 #Equivalent to : m*sigma**2
# tck_y = splrep(z_index, y_for_fitting, s=smoothing_param_y)
# y_disp_smooth = splev(z_index, tck_y)
# y_disp_smooth *= -1

#with hanning and mirror
from msct_smooth import smoothing_window
x_displacement_array = asarray(x_displacement)
y_displacement_array = asarray(y_displacement)
x_disp_smooth = smoothing_window(x_displacement_array, window_len=31)
y_disp_smooth = smoothing_window(y_displacement_array, window_len=31)

# #with hanning and no mirror
# window = 'hanning'
# window_len_int = 30
# w = eval(window+'(window_len_int)')
# x_disp_smooth = convolve(x_displacement, w/w.sum(), mode='same')
# y_disp_smooth = convolve(y_displacement, w/w.sum(), mode='same')

plt.figure()
plt.subplot(2,1,1)
plt.plot(z_index,x_displacement, "ro")
plt.plot(z_index,x_disp_smooth)
plt.subplot(2,1,2)
plt.plot(z_index,y_displacement, "ro")
plt.plot(z_index,y_disp_smooth)
plt.title("Smoothing of x (top) and y (bottom) displacement")
plt.show()





# FUNCTION: generate warping field
#input:
#   x_displacement
#   y_displacement
#   z_displacement
#   fname_warp

data_warp = zeros(((((49, 424, 546, 1, 3)))))
for i in range(seg_output_data.shape[0]):
    for j in range(seg_output_data.shape[1]):
        for k in range(seg_output_data.shape[2]):
            data_warp[i, j, k, 0, 0] = -x_disp_smooth[k]
            data_warp[i, j, k, 0, 1] = y_disp_smooth[k]
            data_warp[i, j, k, 0, 2] = 0


hdr_warp.set_intent('vector', (), '')
hdr_warp.set_data_dtype('float32')
img = nibabel.Nifti1Image(data_warp, None, hdr_warp)
nibabel.save(img, 'warping_field.nii.gz')

os.system('sct_apply_transfo -i centerline_T1_mask_cyl_7.nii.gz -d centerline_T2_mask_cyl_7.nii.gz -w warping_field.nii.gz -x nn')
os.system('sct_apply_transfo -i data_RPI_registered.nii.gz -d centerline_T2_mask_cyl_7.nii.gz -w warping_field.nii.gz')





