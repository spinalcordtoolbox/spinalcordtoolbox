#!/usr/bin/env python
#
# Pre-process data for template creation.
#
# N.B.:
# - Edit variable: 'PATH_INFO': corresponds to the variable 'path_results' in file preprocess_data_template.py
# - Edit variable: 'PATH_OUTPUT': results of fully-preprocessed T1 and T2 to be used for generating the template.
#
# Details of what this script does:
#   1. Import dicom files and convert to NIFTI format (``dcm2nii) (output: data_RPI.nii.gz``).
#   2. Change orientation to RPI (sct_orientation).
#   3. Crop image a little above the brainstem and a little under L2/L3 vertebral disk (``sct_crop_image``)(output: ``data_RPI_crop.nii.gz``).
#   4. Process segmentation of the spinal cord (``sct_propseg -i data_RPI_crop.nii.gz -init-centerline centerline_propseg_RI.nii.gz``)(output: ``data_RPI_crop_seg.nii.gz``)
#   5. Erase three bottom and top slices from the segmentation to avoid edge effects from propseg (output: ``data_RPI_crop_seg_mod.nii.gz``)
#   6. Check segmentation results and crop if needed (``sct_crop_image``)(output: ``data_RPI_crop_seg_mod_crop.nii.gz``)
#   7. Concatenation of segmentation and original label file centerline_propseg_RPI.nii.gz (``fslmaths -add``)(output: ``seg_and_labels.nii.gz``).
#   8. Extraction of the centerline for normalizing intensity along the spinalcord before straightening (``sct_get_centerline_from_labels``)(output: ``generated_centerline.nii.gz``)
#   9. Normalize intensity along z (``sct_normalize -c generated_centerline.nii.gz``)(output: ``data_RPI_crop_normalized.nii.gz``)
#   10. Straighten volume using this concatenation (``sct_straighten_spinalcord -c seg_and_labels.nii.gz -a nurbs``)(output: ``data_RPI_crop_normalized_straight.nii.gz``).
#   11. Apply those transformation to labels_vertebral.nii.gz:
#     * crop with zmin_anatomic and zmax_anatomic (``sct_crop_image``)(output: ``labels_vertebral_crop.nii.gz``)
#     * dilate labels before applying warping fields to avoid the disapearance of a label (``fslmaths -dilF)(output: labels_vertebral_crop_dilated.nii.gz``)
#     * apply warping field curve2straight (``sct_apply_transfo -x nn) (output: labels_vertebral_crop_dialeted_reg.nii.gz``)
#     * select center of mass of labels volume due to past dilatation (``sct_label_utils -t cubic-to-point)(output: labels_vertebral_crop_dilated_reg_2point.nii.gz``)
#   12. Apply transfo to seg_and_labels.nii.gz (``sct_apply_transfo)(output: seg_and_labels_reg.nii.gz``).
#   13. Crop volumes one more time to erase the blank spaces due to the straightening. To do this, the pipeline uses your straight centerline as input and returns the slices number of the upper and lower nonzero points. It then crops your volume (``sct_crop_image)(outputs: data_RPI_crop_normalized_straight_crop.nii.gz, labels_vertebral_crop_dilated_reg_crop.nii.gz``).
#   14. For each subject of your list, the pipeline creates a cross of 5 mm at the top label from labels_vertebral_crop_dilated_reg_crop.nii.gz in the center of the plan xOy and a point at the bottom label from labels_vertebral.nii.gz in the center of the plan xOy (``sct_create_cross)(output:landmark_native.nii.gz``).
#   15. Calculate mean position of top and bottom labels from your list of subjects to create cross on a template shape file (``sct_create_cross``)
#   16. Push the straightened volumes into the template space. The template space has crosses in it for registration. (``sct_push_into_template_space)(outputs: data_RPI_crop_straight_normalized_crop_2temp.nii.gz, labels_vertebral_crop_dilated_reg_crop_2temp.nii.gz``)
#   17. Apply cubic to point to the label file as it now presents cubic group of labels instead of discrete labels (``sct_label_utils -t cubic-to-point) (output: labels_vertebral_dilated_reg_2point_crop_2temp.nii.gz``)
#   18. Use sct_average_levels to calculate the mean landmarks for vertebral levels in the template space. This scripts take the folder containing all the masks created in previous step and for a given landmark it averages values across all subjects and put a landmark at this averaged value. You only have to do this once for a given preprocessing process. If you change the preprocessing or if you add subjects you have 2 choices : assume that it will not change the average too much and use the previous mask, or generate a new one. (``sct_average_levels) (output: template_landmarks.nii.gz``)
#   19. Use sct_align_vertebrae -t SyN (transformation) -w spline (interpolation) to align the vertebrae using transformation along Z (``sct_align_vertebrae -t SyN -w sline -R template_landmarks.nii.gz)(output: <subject>_aligned_normalized.nii.gz``)

# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 NeuroPoly, Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Tanguy Magnan
# Modified: 2015-07-23
#
# License: see the LICENSE.TXT
#=======================================================================================================================

import os, sys

# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, "scripts"))

import sct_utils as sct
import nibabel
from scipy import ndimage
from numpy import array
import matplotlib.pyplot as plt
import time

# add path to scripts
PATH_INFO = '/Users/benjamindeleener/data/template_preprocessing'  # corresponds to the variable 'path_results' in file preprocess_data_template.py
PATH_OUTPUT = '/Users/benjamindeleener/data/template_preprocessing_final'  # folder where you want the results to be stored

# folder to dataset
folder_data_errsm = '/Volumes/data_shared/montreal_criugm/errsm'
folder_data_sct = '/Volumes/data_shared/montreal_criugm/sct'
folder_data_marseille = '/Volumes/data_shared/marseille'
folder_data_pain = '/Volumes/data_shared/montreal_criugm/simon'

# removed because movement artefact:
# ['errsm_22', folder_data_errsm+'/errsm_22/29-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_22/25-SPINE_T2'],\
# removed because of low contrast:
# ['errsm_02', folder_data_errsm+'/errsm_02/22-SPINE_T1', folder_data_errsm+'/errsm_02/28-SPINE_T2'],
# removed because of stitching issue
# ['TM', folder_data_marseille+'/TM_T057c/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', folder_data_marseille+'/TM_T057c/01_0105_t2-composing'],

"""
# define subject
SUBJECTS_LIST = [['errsm_04', folder_data_errsm+'/errsm_04/16-SPINE_memprage/echo_2.09', folder_data_errsm+'/errsm_04/18-SPINE_space'],
                 ['errsm_05', folder_data_errsm+'/errsm_05/23-SPINE_MEMPRAGE/echo_2.09', folder_data_errsm+'/errsm_05/24-SPINE_SPACE'],
                 ['errsm_09', folder_data_errsm+'/errsm_09/34-SPINE_MEMPRAGE2/echo_2.09', folder_data_errsm+'/errsm_09/33-SPINE_SPACE'],
                 ['errsm_10', folder_data_errsm+'/errsm_10/13-SPINE_MEMPRAGE/echo_2.09', folder_data_errsm+'/errsm_10/20-SPINE_SPACE'],
                 ['errsm_12', folder_data_errsm+'/errsm_12/19-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_12/18-SPINE_T2'],
                 ['errsm_13', folder_data_errsm+'/errsm_13/33-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_13/34-SPINE_T2'],
                 ['errsm_14', folder_data_errsm+'/errsm_14/5002-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_14/5003-SPINE_T2'],
                 ['errsm_16', folder_data_errsm+'/errsm_16/23-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_16/39-SPINE_T2'],
                 ['errsm_17', folder_data_errsm+'/errsm_17/41-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_17/42-SPINE_T2'],
                 ['errsm_18', folder_data_errsm+'/errsm_18/36-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_18/33-SPINE_T2'],
                 ['errsm_11', folder_data_errsm+'/errsm_11/24-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_11/09-SPINE_T2'],
                 ['errsm_21', folder_data_errsm+'/errsm_21/27-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_21/30-SPINE_T2'],
                 ['errsm_23', folder_data_errsm+'/errsm_23/29-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_23/28-SPINE_T2'],
                 ['errsm_24', folder_data_errsm+'/errsm_24/20-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_24/24-SPINE_T2'],
                 ['errsm_25', folder_data_errsm+'/errsm_25/25-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_25/26-SPINE_T2'],
                 ['errsm_30', folder_data_errsm+'/errsm_30/51-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_30/50-SPINE_T2'],
                 ['errsm_31', folder_data_errsm+'/errsm_31/31-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_31/32-SPINE_T2'],
                 ['errsm_32', folder_data_errsm+'/errsm_32/16-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_32/19-SPINE_T2'],
                 ['errsm_33', folder_data_errsm+'/errsm_33/30-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_33/31-SPINE_T2'],
                 ['sct_001', folder_data_sct+'/sct_001/17-SPINE_T1/echo_2.09', folder_data_sct+'/sct_001/16-SPINE_T2'],
                 ['sct_002', folder_data_sct+'/sct_002/12-SPINE_T1/echo_2.09', folder_data_sct+'/sct_002/18-SPINE_T2'],
                 ['ED', folder_data_marseille+'/ED/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-101', folder_data_marseille+'/ED/01_0008_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-65'],
                 ['ALT', folder_data_marseille+'/ALT/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-15', folder_data_marseille+'/ALT/01_0100_space-composing'],
                 ['JD', folder_data_marseille+'/JD/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-23', folder_data_marseille+'/JD/01_0100_compo-space'],
                 ['JW', folder_data_marseille+'/JW/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', folder_data_marseille+'/JW/01_0100_compo-space'],
                 ['MLL', folder_data_marseille+'/MLL_1016/01_0008_sc-mprage-1mm-2palliers-fov384-comp-sp-7', folder_data_marseille+'/MLL_1016/01_0100_t2-compo'],
                 ['MT', folder_data_marseille+'/MT/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', folder_data_marseille+'/MT/01_0100_t2composing'],
                 ['T045', folder_data_marseille+'/T045/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', folder_data_marseille+'/T045/01_0101_t2-3d-composing'],
                 ['T047', folder_data_marseille+'/T047/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', folder_data_marseille+'/T047/01_0100_t2-3d-composing'],
                 ['VC', folder_data_marseille+'/VC/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-23', folder_data_marseille+'/VC/01_0008_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-113'],
                 ['VG', folder_data_marseille+'/VG/T1/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-15', folder_data_marseille+'/VG/T2/01_0024_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-11'],
                 ['VP', folder_data_marseille+'/VP/01_0011_sc-mprage-1mm-2palliers-fov384-comp-sp-25', folder_data_marseille+'/VP/01_0100_space-compo'],
                 ['pain_pilot_1', folder_data_pain+'/d_sp_pain_pilot1/24-SPINE_T1/echo_2.09', folder_data_pain+'/d_sp_pain_pilot1/25-SPINE'],
                 ['pain_pilot_2', folder_data_pain+'/d_sp_pain_pilot2/13-SPINE_T1/echo_2.09', folder_data_pain+'/d_sp_pain_pilot2/30-SPINE_T2'],
                 ['pain_pilot_4', folder_data_pain+'/d_sp_pain_pilot4/33-SPINE_T1/echo_2.09', folder_data_pain+'/d_sp_pain_pilot4/32-SPINE_T2'],
                 ['errsm_20', folder_data_errsm+'/errsm_20/12-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_20/34-SPINE_T2'],
                 ['pain_pilot_3', folder_data_pain+'/d_sp_pain_pilot3/16-SPINE_T1/echo_2.09', folder_data_pain+'/d_sp_pain_pilot3/31-SPINE_T2'],
                 ['errsm_34', folder_data_errsm+'/errsm_34/41-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_34/40-SPINE_T2'],
                 ['errsm_35', folder_data_errsm+'/errsm_35/37-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_35/38-SPINE_T2'],
                 ['pain_pilot_7', folder_data_pain+'/d_sp_pain_pilot7/32-SPINE_T1/echo_2.09', folder_data_pain+'/d_sp_pain_pilot7/33-SPINE_T2'],
                 ['errsm_03', folder_data_errsm+'/errsm_03/32-SPINE_all/echo_2.09', folder_data_errsm+'/errsm_03/38-SPINE_all_space'],
                 ['FR', folder_data_marseille+'/FR_T080/01_0039_sc-mprage-1mm-3palliers-fov384-comp-sp-13', folder_data_marseille+'/FR_T080/01_0104_spine2'],
                 ['GB', folder_data_marseille+'/GB_T083/01_0029_sc-mprage-1mm-2palliers-fov384-comp-sp-5', folder_data_marseille+'/GB_T083/01_0033_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-7'],
                 ['errsm_36', folder_data_errsm+'/errsm_36/30-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_36/31-SPINE_T2'],
                 ['errsm_37', folder_data_errsm+'/errsm_37/19-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_37/20-SPINE_T2'],
                 ['errsm_43', folder_data_errsm+'/errsm_43/22-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_43/18-SPINE_T2'],
                 ['errsm_44', folder_data_errsm+'/errsm_44/18-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_44/19-SPINE_T2'],
                 ['AM', folder_data_marseille+'/AM/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', folder_data_marseille+'/AM/01_0100_compo-t2-spine'],
                 ['HB', folder_data_marseille+'/HB/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-29', folder_data_marseille+'/HB/01_0100_t2-compo'],
                 ['PA', folder_data_marseille+'/PA/01_0034_sc-mprage-1mm-2palliers-fov384-comp-sp-5', folder_data_marseille+'/PA/01_0038_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-7']
                 ]

['errsm_04', new_folder + '/errsm_04/T1', new_folder + '/errsm_04/T2'],
    ['errsm_05', new_folder + '/errsm_05/T1', new_folder + '/errsm_05/T2'],
    ['errsm_09', new_folder + '/errsm_09/T1', new_folder + '/errsm_09/T2'],
    ['errsm_10', new_folder + '/errsm_10/T1', new_folder + '/errsm_10/T2'],
    ['errsm_12', new_folder + '/errsm_12/T1', new_folder + '/errsm_12/T2'],
    ['errsm_13', new_folder + '/errsm_13/T1', new_folder + '/errsm_13/T2'],
    ['errsm_14', new_folder + '/errsm_14/T1', new_folder + '/errsm_14/T2'],
    ['errsm_16', new_folder + '/errsm_16/T1', new_folder + '/errsm_16/T2'],
    ['errsm_17', new_folder + '/errsm_17/T1', new_folder + '/errsm_17/T2'],
    ['errsm_18', new_folder + '/errsm_18/T1', new_folder + '/errsm_18/T2'],
    ['errsm_11', new_folder + '/errsm_11/T1', new_folder + '/errsm_11/T2'],
    ['errsm_21', new_folder + '/errsm_21/T1', new_folder + '/errsm_21/T2'],
    ['errsm_23', new_folder + '/errsm_23/T1', new_folder + '/errsm_23/T2'],
    ['errsm_24', new_folder + '/errsm_24/T1', new_folder + '/errsm_24/T2'],
    ['errsm_25', new_folder + '/errsm_25/T1', new_folder + '/errsm_25/T2'],
    ['errsm_30', new_folder + '/errsm_30/T1', new_folder + '/errsm_30/T2'],
    ['errsm_31', new_folder + '/errsm_31/T1', new_folder + '/errsm_31/T2'],
    ['errsm_32', new_folder + '/errsm_32/T1', new_folder + '/errsm_32/T2'],
    ['errsm_33', new_folder + '/errsm_33/T1', new_folder + '/errsm_33/T2'],
    ['sct_001', new_folder + '/sct_001/T1', new_folder + '/sct_001/T2'],
    ['sct_002', new_folder + '/sct_002/T1', new_folder + '/sct_002/T2'],
    ['ED', new_folder + '/ED/T1', new_folder + '/ED/T2'],
    ['ALT', new_folder + '/ALT/T1', new_folder + '/ALT/T2'],
    ['JD', new_folder + '/JD/T1', new_folder + '/JD/T2'],
    ['JW', new_folder + '/JW/T1', new_folder + '/JW/T2'],
    ['MLL', new_folder + '/MLL/T1', new_folder + '/MLL/T2'],
    ['MT', new_folder + '/MT/T1', new_folder + '/MT/T2'],
    ['T045', new_folder + '/T045/T1', new_folder + '/T045/T2'],
    ['T047', new_folder + '/T047/T1', new_folder + '/T047/T2'],
    ['VC', new_folder + '/VC/T1', new_folder + '/VC/T2'],
    ['VG', new_folder + '/VG/T1', new_folder + '/VG/T2'],
    ['VP', new_folder + '/VP/T1', new_folder + '/VP/T2'],
    ['pain_pilot_1', new_folder + '/pain_pilot_1/T1', new_folder + '/pain_pilot_1/T2'],
    ['pain_pilot_2', new_folder + '/pain_pilot_2/T1', new_folder + '/pain_pilot_2/T2'],
    ['pain_pilot_4', new_folder + '/pain_pilot_4/T1', new_folder + '/pain_pilot_4/T2'],
    ['errsm_20', new_folder + '/errsm_20/T1', new_folder + '/errsm_20/T2'],
    ['pain_pilot_3', new_folder + '/pain_pilot_3/T1', new_folder + '/pain_pilot_3/T2'],
    ['errsm_34', new_folder + '/errsm_34/T1', new_folder + '/errsm_34/T2'],
    ['errsm_35', new_folder + '/errsm_35/T1', new_folder + '/errsm_35/T2'],
    ['pain_pilot_7', new_folder + '/pain_pilot_7/T1', new_folder + '/pain_pilot_7/T2'],
    ['errsm_03', new_folder + '/errsm_03/T1', new_folder + '/errsm_03/T2'],
    ['FR', new_folder + '/FR/T1', new_folder + '/FR/T2'],
    ['GB', new_folder + '/GB/T1', new_folder + '/GB/T2'],
    ['errsm_36', new_folder + '/errsm_36/T1', new_folder + '/errsm_36/T2'],
    ['errsm_37', new_folder + '/errsm_37/T1', new_folder + '/errsm_37/T2'],
    ['errsm_43', new_folder + '/errsm_43/T1', new_folder + '/errsm_43/T2'],
    ['errsm_44', new_folder + '/errsm_44/T1', new_folder + '/errsm_44/T2'],
    ['AM', new_folder + '/AM/T1', new_folder + '/AM/T2'],
    ['HB', new_folder + '/HB/T1', new_folder + '/HB/T2'],
    ['PA', new_folder + '/PA/T1', new_folder + '/PA/T2']

                 """

new_folder = "/Users/benjamindeleener/data/template_data"
SUBJECTS_LIST = [
    ['errsm_35', new_folder + '/errsm_35/T1', new_folder + '/errsm_35/T2'],
    ['pain_pilot_7', new_folder + '/pain_pilot_7/T1', new_folder + '/pain_pilot_7/T2'],
    ['errsm_03', new_folder + '/errsm_03/T1', new_folder + '/errsm_03/T2'],
    ['FR', new_folder + '/FR/T1', new_folder + '/FR/T2'],
    ['GB', new_folder + '/GB/T1', new_folder + '/GB/T2'],
    ['errsm_36', new_folder + '/errsm_36/T1', new_folder + '/errsm_36/T2'],
    ['errsm_37', new_folder + '/errsm_37/T1', new_folder + '/errsm_37/T2'],
    ['errsm_43', new_folder + '/errsm_43/T1', new_folder + '/errsm_43/T2'],
    ['errsm_44', new_folder + '/errsm_44/T1', new_folder + '/errsm_44/T2'],
    ['AM', new_folder + '/AM/T1', new_folder + '/AM/T2'],
    ['HB', new_folder + '/HB/T1', new_folder + '/HB/T2'],
    ['PA', new_folder + '/PA/T1', new_folder + '/PA/T2']
]

propseg_parameters = {
                        'errsm_36/T2': ' -init 0.55',
                        'MLL/T2': ' -init 0.6',
                        'T047/T2': ' -init 0.6',
                        'pain_pilot_7/T2': ' -init 0.6',
                        'VG/T2': ' -init 0.55'
                     }

#Parameters:
height_of_template_space = 1100
x_size_of_template_space = 201
y_size_of_template_space = 201
spacing = 0.5
number_labels_for_template = 20  # vertebral levels
straightening_parameters = '-params algo_fitting=nurbs'
"""
# generate template space
from msct_image import Image
from numpy import zeros
template = Image('/Users/benjamindeleener/code/spinalcordtoolbox/dev/template_creation/template_landmarks-mm.nii.gz')
template_space = Image([x_size_of_template_space, y_size_of_template_space, height_of_template_space])
template_space.data = zeros((x_size_of_template_space, y_size_of_template_space, height_of_template_space))
template_space.hdr = template.hdr
template_space.hdr.set_data_dtype('float32')
origin = [(x_size_of_template_space-1.0)/4.0, -(y_size_of_template_space-1.0)/4.0, -((height_of_template_space/4.0)-spacing)]
template_space.hdr.structarr['dim'] = [3.0, x_size_of_template_space, y_size_of_template_space, height_of_template_space, 1.0, 1.0, 1.0, 1.0]
template_space.hdr.structarr['pixdim'] = [-1.0, spacing, spacing, spacing, 1.0, 1.0, 1.0, 1.0]
template_space.hdr.structarr['qoffset_x'] = origin[0]
template_space.hdr.structarr['qoffset_y'] = origin[1]
template_space.hdr.structarr['qoffset_z'] = origin[2]
template_space.hdr.structarr['srow_x'][-1] = origin[0]
template_space.hdr.structarr['srow_y'][-1] = origin[1]
template_space.hdr.structarr['srow_z'][-1] = origin[2]
template_space.hdr.structarr['srow_x'][0] = -spacing
template_space.hdr.structarr['srow_y'][1] = spacing
template_space.hdr.structarr['srow_z'][2] = spacing
template_space.setFileName('/Users/benjamindeleener/code/spinalcordtoolbox/dev/template_creation/template_landmarks-mm.nii.gz')
template_space.save()
"""

class TimeObject:
    def __init__(self, number_of_subjects=1):
        self.start_timer = 0
        self.time_list = []
        self.total_number_of_subjects = number_of_subjects
        self.number_of_subjects_done = 0
        self.is_started = False

    def start(self):
        self.start_timer = time.time()
        self.is_started = True

    def one_subject_done(self):
        self.number_of_subjects_done += 1
        self.time_list.append(time.time() - self.start_timer)
        remaining_subjects = self.total_number_of_subjects - self.number_of_subjects_done
        time_one_subject = self.time_list[-1] / self.number_of_subjects_done
        remaining_time = remaining_subjects * time_one_subject
        hours, rem = divmod(remaining_time, 3600)
        minutes, seconds = divmod(rem, 60)
        sct.printv('Remaining time: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))

    def stop(self):
        self.time_list.append(time.time() - self.start_timer)
        hours, rem = divmod(self.time_list[-1], 3600)
        minutes, seconds = divmod(rem, 60)
        sct.printv('Total time: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))
        self.is_started = False

    def printRemainingTime(self):
        remaining_subjects = self.total_number_of_subjects - self.number_of_subjects_done
        time_one_subject = self.time_list[-1] / self.number_of_subjects_done
        remaining_time = remaining_subjects * time_one_subject
        hours, rem = divmod(remaining_time, 3600)
        minutes, seconds = divmod(rem, 60)
        if self.is_started:
            sct.printv('Remaining time: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))
        else:
            sct.printv('Total time: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))

    def printTotalTime(self):
        hours, rem = divmod(self.time_list[-1], 3600)
        minutes, seconds = divmod(rem, 60)
        if self.is_started:
            sct.printv('Remaining time: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))
        else:
            sct.printv('Total time: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))

timer = dict()
timer['Total'] = TimeObject(number_of_subjects=1)
timer['T1_do_preprocessing'] = TimeObject(number_of_subjects=len(SUBJECTS_LIST))
timer['T1_create_cross'] = TimeObject(number_of_subjects=len(SUBJECTS_LIST))
timer['T1_push_into_templace_space'] = TimeObject(number_of_subjects=len(SUBJECTS_LIST))
timer['T1_average_levels'] = TimeObject(number_of_subjects=1)
timer['T1_align'] = TimeObject(number_of_subjects=len(SUBJECTS_LIST))
timer['T2'] = TimeObject(number_of_subjects=1)
timer['T2_do_preprocessing'] = TimeObject(number_of_subjects=len(SUBJECTS_LIST))
timer['T2_create_cross'] = TimeObject(number_of_subjects=len(SUBJECTS_LIST))
timer['T2_push_into_templace_space'] = TimeObject(number_of_subjects=len(SUBJECTS_LIST))
timer['T2_average_levels'] = TimeObject(number_of_subjects=1)
timer['T2_align'] = TimeObject(number_of_subjects=len(SUBJECTS_LIST))


def main():
    timer['Total'].start()

    """
    # Processing of T1 data for template
    timer['T1_do_preprocessing'].start()
    #do_preprocessing('T1')
    timer['T1_do_preprocessing'].stop()
    timer['T1_create_cross'].start()
    #create_cross('T1')
    timer['T1_create_cross'].stop()
    timer['T1_push_into_templace_space'].start()
    push_into_templace_space('T1')
    timer['T1_push_into_templace_space'].stop()
    timer['T1_average_levels'].start()
    average_levels('T1')
    timer['T1_average_levels'].stop()"""

    """
    # Processing of T2 data for template
    timer['T2_do_preprocessing'].start()
    #do_preprocessing('T2')
    timer['T2_do_preprocessing'].stop()
    timer['T2_create_cross'].start()
    #create_cross('T2')
    timer['T2_create_cross'].stop()
    timer['T2_push_into_templace_space'].start()
    push_into_templace_space('T2')
    timer['T2_push_into_templace_space'].stop()
    timer['T2_average_levels'].start()
    average_levels('T2')
    timer['T2_average_levels'].stop()"""


    #average_levels('both')
    timer['T1_align'].start()
    #align_vertebrae('T1')
    timer['T1_align'].stop()
    timer['T2_align'].start()
    #align_vertebrae('T2')
    timer['T2_align'].start()

    #qc('T1')
    qc('T2')

    timer['Total'].stop()

    sct.printv('Total time:')
    timer['Total'].printTotalTime()

    """
    sct.printv('T1_do_preprocessing time:')
    timer['T1_do_preprocessing'].printTotalTime()
    sct.printv('T1_create_cross time:')
    timer['T1_create_cross'].printTotalTime()
    sct.printv('T1_push_into_templace_space time:')
    timer['T1_push_into_templace_space'].printTotalTime()
    sct.printv('T1_average_levels time:')
    timer['T1_average_levels'].printTotalTime()"""


    #sct.printv('T1_align time:')
    #timer['T1_align'].printTotalTime()

    """
    sct.printv('T2_do_preprocessing time:')
    timer['T2_do_preprocessing'].printTotalTime()
    sct.printv('T2_create_cross time:')
    timer['T2_create_cross'].printTotalTime()
    sct.printv('T2_push_into_templace_space time:')
    timer['T2_push_into_templace_space'].printTotalTime()
    sct.printv('T2_average_levels time:')
    timer['T2_average_levels'].printTotalTime()"""


    sct.printv('T2_align time:')
    timer['T2_align'].printTotalTime()



def do_preprocessing(contrast):
    # Loop across subjects
    for i in range(0,len(SUBJECTS_LIST)):
        subject = SUBJECTS_LIST[i][0]

        # Should check all inputs before starting the processing of the data

        # Create and go to output folder
        print '\nCreate -if not existing- and go to output folder '+ PATH_OUTPUT + '/subjects/'+subject+'/'+contrast
        if not os.path.isdir(PATH_OUTPUT + '/subjects/'+subject):
            os.makedirs(PATH_OUTPUT + '/subjects/'+subject)
        if not os.path.isdir(PATH_OUTPUT + '/subjects/'+subject+'/'+contrast):
            os.makedirs(PATH_OUTPUT + '/subjects/'+subject+'/'+contrast)
        os.chdir(PATH_OUTPUT + '/subjects/'+subject+'/'+contrast)

        # convert to nii
        print '\nChecking if dicoms have already been imported...'
        list_file = os.listdir(PATH_OUTPUT + '/subjects/'+subject+'/'+contrast)
        if 'data.nii.gz' not in list_file:
            print '\nImporting dicoms and converting to nii...'
            if contrast == 'T1':
                sct.run('dcm2nii -o . -r N ' + SUBJECTS_LIST[i][1] + '/*.dcm')
            if contrast == 'T2':
                sct.run('dcm2nii -o . -r N ' + SUBJECTS_LIST[i][2] + '/*.dcm')

            # change file name
            print '\nChanging file name to data.nii.gz...'
            sct.run('mv *.nii.gz data.nii.gz')

        # Convert to RPI
        # Input:
        # - data.nii.gz
        # - data_RPI.nii.gz
        print '\nConverting to RPI...'
        sct.run('sct_image -i data.nii.gz -setorient RPI')

        # Get info from txt file
        print '\nRecover infos from text file' + PATH_INFO + '/' + contrast + '/' + subject+ '/' + 'crop.txt'
        file_name = 'crop.txt'
        os.chdir(PATH_INFO + '/' + contrast + '/' + subject)

        file_results = open(PATH_INFO + '/' + contrast + '/' +subject+ '/' +file_name, 'r')
        ymin_anatomic = None
        ymax_anatomic = None
        for line in file_results:
            line_list = line.split(',')
            zmin_anatomic = line.split(',')[0]
            zmax_anatomic = line.split(',')[1]
            zmin_seg = line.split(',')[2]
            zmax_seg = line.split(',')[3]
            if len(line_list) == 6:
                ymin_anatomic = line.split(',')[4]
                ymax_anatomic = line.split(',')[5]
        file_results.close()

        os.chdir(PATH_OUTPUT + '/subjects/'+subject+ '/' + contrast)

        # Crop image
        print '\nCropping image at L2-L3 and a little above brainstem...'
        if ymin_anatomic == None and ymax_anatomic == None:
            sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start ' + zmin_anatomic + ' -end ' + zmax_anatomic )
        else: sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 1,2 -start ' + ymin_anatomic +','+zmin_anatomic+ ' -end ' + ymax_anatomic+','+zmax_anatomic )

        # propseg
        # input:
        # - data_RPI_crop.nii.gz
        # - labels_propseg.nii.gz
        # output:
        # - data_RPI_crop_seg.nii.gz
        print '\nExtracting segmentation...'
        list_dir = os.listdir(PATH_INFO + '/' + contrast + '/'+subject)
        centerline_proseg = False
        for k in range(len(list_dir)):
            if list_dir[k] == 'centerline_propseg_RPI.nii.gz':
                centerline_proseg = True
        if centerline_proseg:
            cmd = ''
            if contrast == 'T1':
                cmd = 'sct_propseg -i data_RPI_crop.nii.gz -c t1 -init-centerline ' + PATH_INFO + '/' + contrast + '/' + subject + '/centerline_propseg_RPI.nii.gz'
            if contrast == 'T2':
                cmd = 'sct_propseg -i data_RPI_crop.nii.gz -c t2 -init-centerline ' + PATH_INFO + '/' + contrast + '/' + subject + '/centerline_propseg_RPI.nii.gz'
            if subject + '/' + contrast in propseg_parameters:
                cmd += propseg_parameters[subject + '/' + contrast]

            sct.run(cmd)
        else:
            if contrast == 'T1':
                sct.run('sct_propseg -i data_RPI_crop.nii.gz -c t1')
            if contrast == 'T2':
                sct.run('sct_propseg -i data_RPI_crop.nii.gz -c t2')

        # Erase 3 top and 3 bottom slices of the segmentation to avoid edge effects  (Done because propseg tends to diverge on edges)
        print '\nErasing 3 top and 3 bottom slices of the segmentation to avoid edge effects of propseg...'
        path_seg, file_seg, ext_seg = sct.extract_fname('data_RPI_crop_seg.nii.gz')
        image_seg = nibabel.load('data_RPI_crop_seg.nii.gz')
        from msct_image import Image
        nx, ny, nz, nt, px, py, pz, pt = Image('data_RPI_crop_seg.nii.gz').dim
        data_seg = image_seg.get_data()
        hdr_seg = image_seg.get_header()
           # List slices that contain non zero values
        z_centerline = [iz for iz in range(0, nz, 1) if data_seg[:,:,iz].any() ]
        for k in range(0,3):
            data_seg[:,:,z_centerline[-1]-k] = 0
            if z_centerline[0]+k < nz:
                data_seg[:,:,z_centerline[0]+k] = 0
        img_seg = nibabel.Nifti1Image(data_seg, None, hdr_seg)
        nibabel.save(img_seg, file_seg + '_mod' + ext_seg)

        # crop segmentation (but keep same dimension)
        # input:
        # - data_crop_denoised_seg_mod.nii.gz
        # - crop.txt
        # output:
        # - data_crop_denoised_seg_mod_crop.nii.gz
        print '\nCropping segmentation...'
        if zmax_seg == 'max':
            nx, ny, nz, nt, px, py, pz, pt = Image('data_RPI_crop_seg.nii.gz').dim
            sct.run('sct_crop_image -i data_RPI_crop_seg_mod.nii.gz -o data_RPI_crop_seg_mod_crop.nii.gz -start ' + zmin_seg + ' -end ' + str(nz) + ' -dim 2 -b 0')
        else:
            sct.run('sct_crop_image -i data_RPI_crop_seg_mod.nii.gz -o data_RPI_crop_seg_mod_crop.nii.gz -start ' + zmin_seg + ' -end ' + zmax_seg + ' -dim 2 -b 0')

        # Concatenate segmentation and labels_updown if labels_updown is inputed. If not, it concatenates the segmentation and centerline_propseg_RPI.
        print '\nConcatenating segmentation and label files...'
        labels_updown = False
        list_file_info = os.listdir(PATH_INFO+ '/' + contrast + '/' + subject)
        for k in range(0,len(list_file_info)):
            if list_file_info[k] == 'labels_updown.nii.gz':
                labels_updown = True
        if centerline_proseg == False and labels_updown == False:
            print '\nERROR: No label file centerline_propseg_RPI.nii.gz or labels_updown.nii.gz in '+PATH_INFO+ '/' + contrast + '/' + subject +'. There must be at least one. Check '+ path_sct+'/dev/template_preprocessing/Readme.md for necessary inputs.'
            sys.exit(2)
        if labels_updown:
            # Creation of centerline from seg and labels for intensity normalization.
            print '\nExtracting centerline for intensity normalization...'
            sct.run('sct_get_centerline.py -i data_RPI_crop_seg_mod_crop.nii.gz -method labels -l ' + PATH_INFO + '/' + contrast + '/' + subject + '/labels_updown.nii.gz')
            sct.run('fslmaths data_RPI_crop_seg_mod_crop.nii.gz -add '+ PATH_INFO + '/' + contrast + '/' + subject + '/labels_updown.nii.gz seg_and_labels.nii.gz')
        else:
            sct.run('sct_get_centerline.py -i data_RPI_crop_seg_mod_crop.nii.gz -method labels -l ' + PATH_INFO + '/' + contrast + '/' + subject + '/centerline_propseg_RPI.nii.gz')
            sct.run('fslmaths data_RPI_crop_seg_mod_crop.nii.gz -add '+ PATH_INFO + '/' + contrast + '/' + subject + '/centerline_propseg_RPI.nii.gz seg_and_labels.nii.gz')

        # Normalisation of intensity with centerline before straightening (pb of brainstem with bad centerline)
        print '\nNormalizing intensity...'
        sct.run('sct_normalize.py -i data_RPI_crop.nii.gz -c generated_centerline.nii.gz')

        # straighten image using the concatenation of the segmentation and the labels
        # function: sct_straighten_spinalcord (option: nurbs)
        # input:
        # - data_crop_normalized.nii.gz
        # output:
        # - warp_curve2straight.nii.gz
        # - data_RPI_crop_normalized_straight.nii.gz
        print '\nStraightening image using centerline...'
        cmd_straighten = ('sct_straighten_spinalcord -i data_RPI_crop_normalized.nii.gz -s ' + PATH_OUTPUT + '/subjects/' + subject + '/' + contrast + '/seg_and_labels.nii.gz -o data_RPI_crop_normalized_straight.nii.gz '+straightening_parameters)
        #sct.printv(cmd_straighten)
        sct.run(cmd_straighten)

        # # # normalize intensity
        # print '\nNormalizing intensity of the straightened image...'
        # sct.run('sct_normalize.py -i data_RPI_crop_straight.nii.gz')

        # Crop labels_vertebral file
        print '\nCropping labels_vertebral file...'
        if ymin_anatomic == None and ymax_anatomic == None:
            sct.run('sct_crop_image -i '+PATH_INFO + '/' + contrast + '/' + subject+ '/labels_vertebral.nii.gz -o labels_vertebral_crop.nii.gz -start ' + zmin_anatomic + ' -end ' + zmax_anatomic + ' -dim 2')
        else: sct.run('sct_crop_image -i '+PATH_INFO + '/' + contrast + '/' + subject+ '/labels_vertebral.nii.gz -o labels_vertebral_crop.nii.gz -start ' + ymin_anatomic+','+zmin_anatomic + ' -end ' + ymax_anatomic+','+ zmax_anatomic + ' -dim 1,2')
        # Dilate labels from labels_vertebral file before straightening
        print '\nDilating labels from labels_vertebral file...'
        sct.run('fslmaths '+ PATH_OUTPUT + '/subjects/' + subject+ '/' + contrast + '/labels_vertebral_crop.nii.gz -dilF labels_vertebral_dilated.nii.gz')

        # apply straightening to labels_vertebral_dilated.nii.gz and to seg_and_labels.nii.gz
        # function: sct_apply_transfo
        # input:
        # - labels_vertebral_dilated.nii.gz
        # - warp_curve2straight.nii.gz
        # output:
        # - labels_vertebral_dilated_reg.nii.gz
        print '\nApplying straightening to labels_vertebral_dilated.nii.gz...'
        sct.run('sct_apply_transfo -i labels_vertebral_dilated.nii.gz -d data_RPI_crop_normalized_straight.nii.gz -w warp_curve2straight.nii.gz -x nn')

        # Select center of mass of labels volume due to past dilatation
        # REMOVE IF NOT REQUIRED
        print '\nSelecting center of mass of labels volume due to past dilatation...'
        sct.run('sct_label_utils -i labels_vertebral_dilated_reg.nii.gz -o labels_vertebral_dilated_reg_2point.nii.gz -t cubic-to-point')

        # Apply straightening to seg_and_labels.nii.gz
        print'\nApplying transfo to seg_and_labels.nii.gz ...'
        sct.run('sct_apply_transfo -i seg_and_labels.nii.gz -d data_RPI_crop_normalized_straight.nii.gz -w warp_curve2straight.nii.gz -x nn')

        ##Calculate the extrem non zero points of the straightened centerline file to crop image one last time
        """file = nibabel.load('seg_and_labels_reg.nii.gz')
        data_c = file.get_data()

        X,Y,Z = (data_c>0).nonzero()

        z_max = max(Z)

        z_min = min(Z)

        # Crop image one last time
        print'\nCrop image one last time and create cross to push into template space...'
        sct.run('sct_crop_image -i data_RPI_crop_normalized_straight.nii.gz -o data_RPI_crop_normalized_straight_crop.nii.gz -dim 2 -start '+ str(z_min)+' -end '+ str(z_max))

        # Crop labels_vertebral_reg.nii.gz
        print'\nCrop labels_vertebral_reg.nii.gz and use cross to push into template space...'
        sct.run('sct_crop_image -i labels_vertebral_dilated_reg_2point.nii.gz -o labels_vertebral_dilated_reg_2point_crop.nii.gz -dim 2 -start '+ str(z_min)+' -end '+ str(z_max))
        """
        timer[contrast+'_do_preprocessing'].one_subject_done()

# Create cross at the first and last labels for each subject. This cross will be used to push the subject into the template space using affine transfo.
def create_cross(contrast):
    # Define list to gather all distances
    list_distances_1 = []
    list_distances_2 = []
    for i in range(0,len(SUBJECTS_LIST)):
        subject = SUBJECTS_LIST[i][0]

        # go to output folder
        print '\nGo to output folder '+ PATH_OUTPUT + '/subjects/'+ subject+ '/' + contrast
        os.chdir(PATH_OUTPUT + '/subjects/' + subject + '/' + contrast )

        #Calculate distances between : (last_label and bottom)  and (first label and top)
        print '\nCalculating distances between : (last_label and bottom)  and (first label and top)...'
        img_label = nibabel.load('labels_vertebral_dilated_reg_2point.nii.gz')
        data_labels = img_label.get_data()
        from msct_image import Image
        nx, ny, nz, nt, px, py, pz, pt = Image('labels_vertebral_dilated_reg_2point.nii.gz').dim
        X, Y, Z = (data_labels > 0).nonzero()
        list_coordinates = [([X[i], Y[i], Z[i], data_labels[X[i], Y[i], Z[i]]]) for i in range(0, len(X))]
        for i in range(len(list_coordinates)):
            if list_coordinates[i][3] == 1:
                coordinates_first_label = list_coordinates[i]
            if list_coordinates[i][3] == 20:
                coordinates_last_label = list_coordinates[i]
        # Distance 1st label top
        distance_1 = nz - 1 - coordinates_first_label[2]
        distance_2 = nz - 1 - coordinates_last_label[2]

        # Complete list to gather all distances
        list_distances_1.append(distance_1)
        list_distances_2.append(distance_2)

        # Create a cross on each subject at first and last labels
        print '\nCreating a cross at first and last labels...'
        os.system('sct_create_cross.py -i data_RPI_crop_normalized_straight.nii.gz -x ' +str(int(round(nx/2.0)))+' -y '+str(int(round(ny/2.0)))+ ' -s '+str(coordinates_last_label[2])+ ' -e '+ str(coordinates_first_label[2]))

        # Write into a txt file the list of distances
        # os.chdir('../')
        # f_distance = open('list_distances.txt', 'w')
        # f_distance.write(str(distance_1))
        # f_distance.write(' ')
        # f_distance.write(str(distance_2))
        # f_distance.write('\n')

        timer[contrast + '_create_cross'].one_subject_done()

    # Calculate mean cross height for template and create file of reference
    print '\nCalculating mean cross height for template and create file of reference'
    mean_distance_1 = int(round(sum(list_distances_1)/len(list_distances_1)))
    mean_distance_2 = int(round(sum(list_distances_2)/len(list_distances_2)))
    L = height_of_template_space - 2 * mean_distance_2
    H = height_of_template_space - 2 * mean_distance_1
    os.chdir(path_sct+'/dev/template_creation')
    os.system('sct_create_cross.py -i template_landmarks-mm.nii.gz -x ' +str(int(x_size_of_template_space/2))+' -y '+str(int(y_size_of_template_space/2))+ ' -s '+str(L)+ ' -e '+ str(H))



# push into template space
def push_into_templace_space(contrast):
    for i in range(0,len(SUBJECTS_LIST)):
        subject = SUBJECTS_LIST[i][0]

        # go to output folder
        print '\nGo to output folder '+ PATH_OUTPUT + '/subjects/' + subject + '/' + contrast
        os.chdir(PATH_OUTPUT + '/subjects/' + subject + '/' + contrast )

        # Push into template space
        print'\nPush into template space...'
        sct.run('sct_push_into_template_space.py -i data_RPI_crop_normalized_straight.nii.gz -n landmark_native.nii.gz')
        sct.run('sct_push_into_template_space.py -i labels_vertebral_dilated_reg_2point.nii.gz -n landmark_native.nii.gz -a nn')

        # Change image type from float64 to uint16
        sct.run('sct_change_image_type.py -i data_RPI_crop_normalized_straight_2temp.nii.gz -o data_RPI_crop_normalized_straight_2temp.nii.gz -t uint16')

        # get center of mass of each label group
        print '\nGet center of mass of each label group due to affine transformation...'
        sct.run('sct_label_utils -i labels_vertebral_dilated_reg_2point_2temp.nii.gz -o labels_vertebral_dilated_reg_2point_2temp.nii.gz -t cubic-to-point')

        # Copy labels_vertebral_straight_in_template_space.nii.gz into a folder that will contain each subject labels_vertebral_straight_in_template_space.nii.gz file and rename them
        print'\nCheck if folder '+PATH_OUTPUT +'/labels_vertebral_' + contrast+ ' exists and if not creates it ...'
        # check if folder exists and if not create it
        if not os.path.isdir(PATH_OUTPUT +'/labels_vertebral_' + contrast):
            os.makedirs(PATH_OUTPUT + '/labels_vertebral_' + contrast)
        sct.run('cp labels_vertebral_dilated_reg_2point_2temp.nii.gz '+PATH_OUTPUT +'/labels_vertebral_' + contrast + '/'+subject+'.nii.gz')

        timer[contrast + '_push_into_templace_space'].one_subject_done()

# Check position of labels crop_2temp with image crop_2temp
# if no good: check position of labels reg with image normalized_straight
# if no good: check position of labels dilated with image crop


# Calculate mean labels and save it into folder "labels_vertebral"
def average_levels(contrast):
    if contrast == 'both':
        print 'Averaging levels from T1 and T2 contrasts...'

        from numpy import mean, zeros, array
        n_i, n_l = 2, number_labels_for_template
        average = zeros((n_i, n_l))
        compteur = 0

        img_T1 = nibabel.load(PATH_OUTPUT + '/labels_vertebral_T1/template_landmarks.nii.gz')
        data_T1 = img_T1.get_data()
        X, Y, Z = (data_T1 > 0).nonzero()
        Z = [Z[i] for i in Z.argsort()]
        Z.reverse()

        for i in xrange(n_l):
            if i < len(Z):
                average[compteur][i] = Z[i]

        compteur = compteur + 1

        img_T2 = nibabel.load(PATH_OUTPUT + '/labels_vertebral_T2/template_landmarks.nii.gz')
        data_T2 = img_T2.get_data()
        X, Y, Z = (data_T1 > 0).nonzero()
        Z = [Z[i] for i in Z.argsort()]
        Z.reverse()

        for i in xrange(n_l):
            if i < len(Z):
                average[compteur][i] = Z[i]

        average = array([int(round(mean([average[average[:, i] > 0, i]]))) for i in xrange(n_l)])

        template_absolute_path = path_sct + '/dev/template_creation/template_landmarks-mm.nii.gz'
        print template_absolute_path
        print '\nGet dimensions of template...'
        from msct_image import Image
        nx, ny, nz, nt, px, py, pz, pt = Image(template_absolute_path).dim
        print '.. matrix size: ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz)
        print '.. voxel size:  ' + str(px) + 'mm x ' + str(py) + 'mm x ' + str(pz) + 'mm'

        img = nibabel.load(template_absolute_path)
        data = img.get_data()
        hdr = img.get_header()
        data[:, :, :] = 0
        compteur = 1
        for i in average:
            print int(nx / 2.0), int(ny / 2.0), int(round(i)), int(round(compteur))
            data[int(nx / 2.0), int(ny / 2.0), int(round(i))] = int(round(compteur))
            compteur = compteur + 1

        print '\nSave volume ...'
        # hdr.set_data_dtype('float32') # set imagetype to uint8
        # save volume
        # data = data.astype(float32, copy =False)
        img = nibabel.Nifti1Image(data, None, hdr)
        file_name = PATH_OUTPUT + '/labels_vertebral_T1/template_landmarks.nii.gz'
        nibabel.save(img, file_name)
        print '\nFile created : ' + file_name
        file_name = PATH_OUTPUT + '/labels_vertebral_T2/template_landmarks.nii.gz'
        nibabel.save(img, file_name)
        print '\nFile created : ' + file_name

    else:
        print '\nGo to output folder '+ PATH_OUTPUT + '/labels_vertebral_' + contrast + '\n'
        os.chdir(PATH_OUTPUT +'/labels_vertebral_' + contrast)
        print'\nCalculate mean along subjects of files labels_vertebral and save it into '+PATH_OUTPUT +'/labels_vertebral_' + contrast +' as template_landmarks.nii.gz'
        template_shape = path_sct + '/dev/template_creation/template_landmarks-mm.nii.gz'
        # this function looks at all files inside the folder "labels_vertebral_T*" and find the average vertebral levels across subjects
        sct.run('sct_average_levels.py -i ' +PATH_OUTPUT +'/labels_vertebral_' + contrast + ' -t '+ template_shape +' -n '+ str(number_labels_for_template))


# Aligning vertebrae for all subjects and copy results into "Final_results". Plus: save png images of all subject into a folder named Image_results.
def align_vertebrae(contrast):
    for i in range(0,len(SUBJECTS_LIST)):
        subject = SUBJECTS_LIST[i][0]

        # go to output folder
        print '\nGo to output folder '+ PATH_OUTPUT + '/subjects/'+subject+ '/' + contrast + '\n'
        os.chdir(PATH_OUTPUT + '/subjects/' + subject + '/' + contrast)

        print '\nAligning vertebrae for subject '+subject+'...'
        sct.printv('\nsct_align_vertebrae.py -i data_RPI_crop_normalized_straight_2temp.nii.gz -l ' + PATH_OUTPUT + '/subjects/' + subject + '/' + contrast + '/labels_vertebral_dilated_reg_2point_2temp.nii.gz -R ' +PATH_OUTPUT +'/labels_vertebral_' + contrast + '/template_landmarks.nii.gz -o '+ subject+'_aligned.nii.gz -t nurbs -w spline')
        os.system('sct_align_vertebrae.py -i data_RPI_crop_normalized_straight_2temp.nii.gz -l ' + PATH_OUTPUT + '/subjects/' + subject + '/' + contrast + '/labels_vertebral_dilated_reg_2point_2temp.nii.gz -R ' +PATH_OUTPUT +'/labels_vertebral_' + contrast + '/template_landmarks.nii.gz -o '+ subject+'_aligned.nii.gz -t nurbs -w spline')

        timer[contrast + '_align'].one_subject_done()


def qc(contrast):
    for i in range(0, len(SUBJECTS_LIST)):
        subject = SUBJECTS_LIST[i][0]

        # go to output folder
        print '\nGo to output folder ' + PATH_OUTPUT + '/subjects/' + subject + '/' + contrast + '\n'
        os.chdir(PATH_OUTPUT + '/subjects/' + subject + '/' + contrast)

        print '\nApplying transformations to segmentation to compute accuracy of preprocessing and center the spinal cord.'
        sct.run('sct_apply_transfo -i seg_and_labels_reg.nii.gz -o seg_and_labels_reg_2temp.nii.gz -d data_RPI_crop_normalized_straight_2temp.nii.gz -w native2temp.txt -x nn')
        sct.run('sct_apply_transfo -i seg_and_labels_reg_2temp.nii.gz -o seg_and_labels_reg_2temp_aligned.nii.gz -d data_RPI_crop_normalized_straight_2temp.nii.gz -w warp_subject2template.nii.gz -x nn')
        from msct_image import Image
        image_centerline_aligned = Image('seg_and_labels_reg_2temp_aligned.nii.gz')
        nx, ny, nz, nt, px, py, pz, pt = image_centerline_aligned.dim
        points = image_centerline_aligned.getNonZeroCoordinates()
        center = None
        for coord in points:
            if not center:
                center = coord
            else:
                center += coord
        center /= float(len(points))
        distance_from_center = [(int(round(nx / 2)) - center.x)*px, (int(round(ny / 2)) - center.y)*py]
        print 'Position of center: ', center
        print 'Distance from center: ', distance_from_center
        text_file = open("centering_transformation.txt", "w")
        text_file.write("#Insight Transform File V1.0\n")
        text_file.write("#Transform 0\n")
        text_file.write("Transform: AffineTransform_double_3_3\n")
        text_file.write("Parameters: 1 0 0 0 1 0 0 0 1 %.9f %.9f %.9f\n" % (-distance_from_center[0], distance_from_center[1], 0.0))
        text_file.write("FixedParameters: 0 0 0\n")
        text_file.close()
        sct.run('sct_apply_transfo -i seg_and_labels_reg_2temp_aligned.nii.gz -o ' + subject + '_centerline_centered.nii.gz -d data_RPI_crop_normalized_straight_2temp.nii.gz -w centering_transformation.txt -x nn')
        sct.run('sct_apply_transfo -i ' + subject + '_aligned.nii.gz -o ' + subject + '_final_full.nii.gz -d data_RPI_crop_normalized_straight_2temp.nii.gz -w centering_transformation.txt -x spline')

        # Change image type from float64 to uint16
        sct.run('sct_change_image_type.py -i ' + subject+'_final_full.nii.gz -o ' + subject+'_final_full.nii.gz -t uint16')

        # Crop all subjects to final space
        sct.run('sct_crop_image -i ' + subject + '_final_full.nii.gz -o ' + subject + '_final.nii.gz -start 30,30 -end 170,170 -dim 0,1')
        sct.run('sct_crop_image -i ' + subject + '_centerline_centered.nii.gz -o ' + subject + '_centerline_final.nii.gz -start 30,30 -end 170,170 -dim 0,1')

        # Concatenate transformations
        sct.run('sct_concat_transfo -d ' + subject + '_final.nii.gz -w native2temp.txt,warp_subject2template.nii.gz,centering_transformation.txt -o warp_native2template.nii.gz')
        sct.run('sct_concat_transfo -d ' + subject + '_final.nii.gz -w -centering_transformation.txt,warp_template2subject.nii.gz,-native2temp.txt -o warp_template2native.nii.gz')

        # Inform that results for the subject is ready
        print'\nThe results for subject ' + subject + ' are ready. You can visualize them by tapping: fslview ' + subject + '_final.nii.gz'

        # Copy final results into final results
        if not os.path.isdir(PATH_OUTPUT +'/final_results'):
            os.makedirs(PATH_OUTPUT +'/final_results')
        sct.run('cp ' + subject + '_final.nii.gz ' + PATH_OUTPUT + '/final_results/' + subject + '_final_' + contrast + '.nii.gz')

        # Save png images of the results into a different folder
        print '\nSaving png image of the final result into ' + PATH_OUTPUT + '/Image_results...'
        if not os.path.isdir(PATH_OUTPUT + '/image_results'):
            os.makedirs(PATH_OUTPUT + '/image_results')
        from msct_image import Image
        image_result = Image(PATH_OUTPUT + '/final_results/' + subject + '_final_' + contrast + '.nii.gz')
        nx, ny, nz, nt, px, py, pz, pt = image_result.dim
        data = image_result.data
        sagital_middle = nx / 2
        coronal_middle = ny / 2
        sagittal = data[sagital_middle, :, :].T
        coronal = data[:, coronal_middle, :].T
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(sagittal, cmap='gray', origin='lower')
        ax[0].set_title('sagittal')
        ax[1].imshow(coronal, cmap='gray', origin='lower')
        ax[1].set_title('coronal')
        for i in range(2):
            ax[i].set_axis_off()
        fig1 = plt.gcf()
        fig1.savefig(PATH_OUTPUT + '/image_results' + '/' + subject + '_final_' + contrast + '.png', format='png')



#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
   # call main function
   main()
