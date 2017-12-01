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
# Authors: Benjamin De Leener
# Modified: 2016-10-16
#
# License: see the LICENSE.TXT
#=======================================================================================================================

import os, sys, commands

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

from msct_image import Image

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
    ['ALT', folder_data_marseille+'/ALT/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-15', folder_data_marseille+'/ALT/01_0100_space-composing'],
    ['errsm_11', folder_data_errsm + '/errsm_11/24-SPINE_T1/echo_2.09', folder_data_errsm + '/errsm_11/09-SPINE_T2'],
    ['errsm_18', folder_data_errsm + '/errsm_18/36-SPINE_T1/echo_2.09', folder_data_errsm + '/errsm_18/33-SPINE_T2'],
    ['MLL', folder_data_marseille+'/MLL_1016/01_0008_sc-mprage-1mm-2palliers-fov384-comp-sp-7', folder_data_marseille+'/MLL_1016/01_0100_t2-compo'],
    ['errsm_03', folder_data_errsm+'/errsm_03/32-SPINE_all/echo_2.09', folder_data_errsm+'/errsm_03/38-SPINE_all_space'],
    ['errsm_14', folder_data_errsm+'/errsm_14/5002-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_14/5003-SPINE_T2'],
    ['errsm_25', folder_data_errsm+'/errsm_25/25-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_25/26-SPINE_T2'],
    ['errsm_37', folder_data_errsm+'/errsm_37/19-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_37/20-SPINE_T2'],
    ['sct_001', folder_data_sct+'/sct_001/17-SPINE_T1/echo_2.09', folder_data_sct+'/sct_001/16-SPINE_T2'],
    ['AM', folder_data_marseille+'/AM/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', folder_data_marseille+'/AM/01_0100_compo-t2-spine'],
    ['MT', folder_data_marseille+'/MT/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', folder_data_marseille+'/MT/01_0100_t2composing'],
    ['errsm_04', folder_data_errsm+'/errsm_04/16-SPINE_memprage/echo_2.09', folder_data_errsm+'/errsm_04/18-SPINE_space'],
    ['errsm_16', folder_data_errsm + '/errsm_16/23-SPINE_T1/echo_2.09', folder_data_errsm + '/errsm_16/39-SPINE_T2'],
    ['errsm_30', folder_data_errsm + '/errsm_30/51-SPINE_T1/echo_2.09', folder_data_errsm + '/errsm_30/50-SPINE_T2'],
    ['errsm_43', folder_data_errsm + '/errsm_43/22-SPINE_T1/echo_2.09', folder_data_errsm + '/errsm_43/18-SPINE_T2'],
    ['sct_002', folder_data_sct + '/sct_002/12-SPINE_T1/echo_2.09', folder_data_sct + '/sct_002/18-SPINE_T2'],
    ['ED', folder_data_marseille+'/ED/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-101', folder_data_marseille+'/ED/01_0008_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-65'],
    ['PA', folder_data_marseille+'/PA/01_0034_sc-mprage-1mm-2palliers-fov384-comp-sp-5', folder_data_marseille+'/PA/01_0038_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-7'],
    ['errsm_05', folder_data_errsm+'/errsm_05/23-SPINE_MEMPRAGE/echo_2.09', folder_data_errsm+'/errsm_05/24-SPINE_SPACE'],
    ['errsm_17', folder_data_errsm+'/errsm_17/41-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_17/42-SPINE_T2'],
    ['errsm_31', folder_data_errsm+'/errsm_31/31-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_31/32-SPINE_T2'],
    ['errsm_44', folder_data_errsm+'/errsm_44/18-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_44/19-SPINE_T2'],
    ['FR', folder_data_marseille+'/FR_T080/01_0039_sc-mprage-1mm-3palliers-fov384-comp-sp-13', folder_data_marseille+'/FR_T080/01_0104_spine2'],
    ['T045', folder_data_marseille+'/T045/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', folder_data_marseille+'/T045/01_0101_t2-3d-composing'],
    ['errsm_09', folder_data_errsm+'/errsm_09/34-SPINE_MEMPRAGE2/echo_2.09', folder_data_errsm+'/errsm_09/33-SPINE_SPACE'],
    ['errsm_32', folder_data_errsm+'/errsm_32/16-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_32/19-SPINE_T2'],
    ['pain_pilot_1', folder_data_pain+'/d_sp_pain_pilot1/24-SPINE_T1/echo_2.09', folder_data_pain+'/d_sp_pain_pilot1/25-SPINE'],
    ['pain_pilot_2', folder_data_pain+'/d_sp_pain_pilot2/13-SPINE_T1/echo_2.09', folder_data_pain+'/d_sp_pain_pilot2/30-SPINE_T2'],
    ['pain_pilot_3', folder_data_pain+'/d_sp_pain_pilot3/16-SPINE_T1/echo_2.09', folder_data_pain+'/d_sp_pain_pilot3/31-SPINE_T2'],
    ['pain_pilot_4', folder_data_pain+'/d_sp_pain_pilot4/33-SPINE_T1/echo_2.09', folder_data_pain+'/d_sp_pain_pilot4/32-SPINE_T2'],
    ['pain_pilot_7', folder_data_pain+'/d_sp_pain_pilot7/32-SPINE_T1/echo_2.09', folder_data_pain+'/d_sp_pain_pilot7/33-SPINE_T2'],
    ['GB', folder_data_marseille+'/GB_T083/01_0029_sc-mprage-1mm-2palliers-fov384-comp-sp-5', folder_data_marseille+'/GB_T083/01_0033_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-7'],
    ['HB', folder_data_marseille+'/HB/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-29', folder_data_marseille+'/HB/01_0100_t2-compo'],
    ['VC', folder_data_marseille+'/VC/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-23', folder_data_marseille+'/VC/01_0008_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-113'],
    ['T047', folder_data_marseille+'/T047/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', folder_data_marseille+'/T047/01_0100_t2-3d-composing'],
    ['JD', folder_data_marseille+'/JD/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-23', folder_data_marseille+'/JD/01_0100_compo-space'],
    ['VG', folder_data_marseille+'/VG/T1/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-15', folder_data_marseille+'/VG/T2/01_0024_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-11'],
    ['JW', folder_data_marseille+'/JW/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', folder_data_marseille+'/JW/01_0100_compo-space'],
    ['VP', folder_data_marseille+'/VP/01_0011_sc-mprage-1mm-2palliers-fov384-comp-sp-25', folder_data_marseille+'/VP/01_0100_space-compo'],
    ['errsm_10', folder_data_errsm+'/errsm_10/13-SPINE_MEMPRAGE/echo_2.09', folder_data_errsm+'/errsm_10/20-SPINE_SPACE'],
    ['errsm_20', folder_data_errsm+'/errsm_20/12-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_20/34-SPINE_T2'],
    ['errsm_33', folder_data_errsm+'/errsm_33/30-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_33/31-SPINE_T2'],
    ['errsm_21', folder_data_errsm + '/errsm_21/27-SPINE_T1/echo_2.09', folder_data_errsm + '/errsm_21/30-SPINE_T2'],
    ['errsm_34', folder_data_errsm + '/errsm_34/41-SPINE_T1/echo_2.09', folder_data_errsm + '/errsm_34/40-SPINE_T2'],
    ['errsm_12', folder_data_errsm + '/errsm_12/19-SPINE_T1/echo_2.09', folder_data_errsm + '/errsm_12/18-SPINE_T2'],
    ['errsm_23', folder_data_errsm + '/errsm_23/29-SPINE_T1/echo_2.09', folder_data_errsm + '/errsm_23/28-SPINE_T2'],
    ['errsm_35', folder_data_errsm + '/errsm_35/37-SPINE_T1/echo_2.09', folder_data_errsm + '/errsm_35/38-SPINE_T2'],
    ['errsm_13', folder_data_errsm + '/errsm_13/33-SPINE_T1/echo_2.09', folder_data_errsm + '/errsm_13/34-SPINE_T2'],
    ['errsm_24', folder_data_errsm + '/errsm_24/20-SPINE_T1/echo_2.09', folder_data_errsm + '/errsm_24/24-SPINE_T2'],
    ['errsm_36', folder_data_errsm + '/errsm_36/30-SPINE_T1/echo_2.09', folder_data_errsm + '/errsm_36/31-SPINE_T2']
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

"""
        ['ALT', folder_data_marseille+'/ALT/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-15', folder_data_marseille+'/ALT/01_0100_space-composing'],
        ['AM', folder_data_marseille+'/AM/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', folder_data_marseille+'/AM/01_0100_compo-t2-spine'],
        ['ED', folder_data_marseille+'/ED/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-101', folder_data_marseille+'/ED/01_0008_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-65'],
        ['errsm_03', folder_data_errsm+'/errsm_03/32-SPINE_all/echo_2.09', folder_data_errsm+'/errsm_03/38-SPINE_all_space']
        
"""

SUBJECTS_LIST = [

    ['errsm_11', folder_data_errsm + '/errsm_11/24-SPINE_T1/echo_2.09', folder_data_errsm + '/errsm_11/09-SPINE_T2'],
    ['errsm_21', folder_data_errsm + '/errsm_21/27-SPINE_T1/echo_2.09', folder_data_errsm + '/errsm_21/30-SPINE_T2']

]

def main():


    compare_csa('T1', 'data_RPI_crop_seg.nii.gz')
    sys.exit(1)


    timer['Total'].start()


    # Processing of T1 data for template
    timer['T1_do_preprocessing'].start()
    #do_preprocessing('T1')
    timer['T1_do_preprocessing'].stop()

    average_centerline('T1')
    #create_mask_template()

    #convert_nii2mnc('T1')
    #do_preprocessing('T2')
    #straighten_all_subjects('T2')
    # convert_nii2mnc('T2')

    #straighten_all_subjects('T1')

    timer['T1_create_cross'].start()
    #create_cross('T1')
    timer['T1_create_cross'].stop()
    timer['T1_push_into_templace_space'].start()
    #push_into_templace_space('T1')
    timer['T1_push_into_templace_space'].stop()
    timer['T1_average_levels'].start()
    #average_levels('T1')
    timer['T1_average_levels'].stop()

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

    """
    #average_levels('both')
    timer['T1_align'].start()
    #align_vertebrae('T1')
    timer['T1_align'].stop()
    timer['T2_align'].start()
    #align_vertebrae('T2')
    timer['T2_align'].start()
    """

    #qc('T1')
    #qc('T2')

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


    #sct.printv('T2_align time:')
    #timer['T2_align'].printTotalTime()


def create_mask_template():
    template = Image('/Users/benjamindeleener/code/sct/dev/template_creation/template_space.nii.gz')
    template.data += 1.0
    template.setFileName('/Users/benjamindeleener/code/sct/dev/template_creation/template_mask.nii.gz')
    template.save()


import matplotlib.cm as cmx
import matplotlib.colors as colors

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


def average_centerline(contrast):
    """
    We assume A and B are Centerlines and have the same number of points
    Parameters
    ----------
    other

    Returns
    -------

    """

    labels_regions = {'PONS': 50, 'MO': 51,
                      'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5, 'C6': 6, 'C7': 7,
                      'T1': 8, 'T2': 9, 'T3': 10, 'T4': 11, 'T5': 12, 'T6': 13, 'T7': 14, 'T8': 15, 'T9': 16, 'T10': 17,
                      'T11': 18, 'T12': 19,
                      'L1': 20, 'L2': 21, 'L3': 22, 'L4': 23, 'L5': 24,
                      'S1': 25, 'S2': 26, 'S3': 27, 'S4': 28, 'S5': 29,
                      'Co': 30}

    regions_labels = {'50': 'PONS', '51': 'MO',
                      '1': 'C1', '2': 'C2', '3': 'C3', '4': 'C4', '5': 'C5', '6': 'C6', '7': 'C7',
                      '8': 'T1', '9': 'T2', '10': 'T3', '11': 'T4', '12': 'T5', '13': 'T6', '14': 'T7', '15': 'T8',
                      '16': 'T9', '17': 'T10', '18': 'T11', '19': 'T12',
                      '20': 'L1', '21': 'L2', '22': 'L3', '23': 'L4', '24': 'L5',
                      '25': 'S1', '26': 'S2', '27': 'S3', '28': 'S4', '29': 'S5',
                      '30': 'Co'}
    list_labels = [50, 51, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                   27, 28, 29, 30]

    number_of_points_in_centerline = 4000
    list_dist_disks = []
    list_centerline = []
    from sct_straighten_spinalcord import smooth_centerline
    from msct_image import Image

    for i in range(0, len(SUBJECTS_LIST)):
        subject = SUBJECTS_LIST[i][0]

        # go to output folder
        print '\nGo to output folder ' + PATH_OUTPUT + '/subjects/' + subject + '/' + contrast
        os.chdir(PATH_OUTPUT + '/subjects/' + subject + '/' + contrast)

        im = Image('labels_vertebral_crop.nii.gz')
        coord = im.getNonZeroCoordinates(sorting='z', reverse_coord=True)
        coord_physical = []

        for c in coord:
            c_p = im.transfo_pix2phys([[c.x, c.y, c.z]])[0]
            c_p.append(c.value)
            coord_physical.append(c_p)

        x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline(
            'generated_centerline.nii.gz',
            algo_fitting='nurbs',
            verbose=0, nurbs_pts_number=number_of_points_in_centerline, all_slices=False, phys_coordinates=True, remove_outliers=True)
        from msct_types import Centerline
        centerline = Centerline(x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv,
                                y_centerline_deriv,
                                z_centerline_deriv)

        centerline.compute_vertebral_distribution(coord_physical)
        list_dist_disks.append(centerline.distance_from_C1label)
        list_centerline.append(centerline)

    import numpy as np

    length_vertebral_levels = {}
    for dist_disks in list_dist_disks:
        for disk_label in dist_disks:
            if disk_label == 'C1':
                length = 0.0
            elif disk_label == 'PONS':
                length = abs(dist_disks[disk_label] - dist_disks['MO'])
            elif disk_label == 'MO':
                length = abs(dist_disks[disk_label] - dist_disks['C1'])
            else:
                index_current_label = list_labels.index(labels_regions[disk_label])
                previous_label = regions_labels[str(list_labels[index_current_label - 1])]
                length = dist_disks[disk_label] - dist_disks[previous_label]

            if disk_label in length_vertebral_levels:
                length_vertebral_levels[disk_label].append(length)
            else:
                length_vertebral_levels[disk_label] = [length]
    #print length_vertebral_levels_p

    average_length = {}
    for disk_label in length_vertebral_levels:
        mean = np.mean(length_vertebral_levels[disk_label])
        std = np.std(length_vertebral_levels[disk_label])
        average_length[disk_label] = [disk_label, mean, std]
    print average_length

    distances_disks_from_C1 = {'C1': 0.0, 'MO': -average_length['MO'][1], 'PONS': -average_length['MO'][1] - average_length['PONS'][1]}
    for disk_number in list_labels:
        if disk_number not in [50, 51, 1] and regions_labels[str(disk_number)] in average_length:
            distances_disks_from_C1[regions_labels[str(disk_number)]] = distances_disks_from_C1[regions_labels[str(disk_number - 1)]] + average_length[regions_labels[str(disk_number)]][1]
    print '\n', distances_disks_from_C1

    """
    distances_disks_from_C1 = {}
    for dist_disks in list_dist_disks:
        for disk_label in dist_disks:
            if disk_label in distances_disks_from_C1:
                distances_disks_from_C1[disk_label].append(dist_disks[disk_label])
            else:
                distances_disks_from_C1[disk_label] = [dist_disks[disk_label]]
    """

    average_distances = []
    for disk_label in distances_disks_from_C1:
        mean = np.mean(distances_disks_from_C1[disk_label])
        std = np.std(distances_disks_from_C1[disk_label])
        average_distances.append([disk_label, mean, std])

    # create average space
    from operator import itemgetter
    average_distances = sorted(average_distances, key=itemgetter(1))
    import cPickle as pickle
    import bz2
    with bz2.BZ2File(PATH_OUTPUT + '/final_results_2016/' + 'template_distances_from_C1_'+contrast+'.pbz2', 'w') as f:
        pickle.dump(average_distances, f)

    print '\nAverage distance\n', average_distances

    number_of_points_between_levels = 100
    disk_average_coordinates = {}
    points_average_centerline = []
    label_points = []
    average_positions_from_C1 = {}
    disk_position_in_centerline = {}

    for i in range(len(average_distances)):
        disk_label = average_distances[i][0]
        average_positions_from_C1[disk_label] = average_distances[i][1]

        for j in range(number_of_points_between_levels):
            relative_position = float(j) / float(number_of_points_between_levels)
            if disk_label in ['PONS', 'MO']:
                relative_position = 1.0 - relative_position
            list_coordinates = [[]] * len(list_centerline)
            for k, centerline in enumerate(list_centerline):
                idx_closest = centerline.get_closest_to_relative_position(disk_label, relative_position)
                if idx_closest is not None:
                    coordinate_closest = centerline.get_point_from_index(idx_closest[0])
                    list_coordinates[k] = coordinate_closest.tolist()
                else:
                    list_coordinates[k] = [np.nan, np.nan, np.nan]

            # average all coordinates
            average_coord = np.nanmean(list_coordinates, axis=0)
            # add it to averaged centerline list of points
            points_average_centerline.append(average_coord)
            label_points.append(disk_label)
            if j == 0:
                disk_average_coordinates[disk_label] = average_coord
                disk_position_in_centerline[disk_label] = i*number_of_points_between_levels

    """
    # compute average vertebral level length
    length_vertebral_levels = {}
    for i in range(len(list_labels) - 1):
        number_vert = list_labels[i]
        label_vert = regions_labels[str(number_vert)]
        label_vert_next = regions_labels[str(list_labels[i + 1])]
        if label_vert in average_positions_from_C1 and label_vert_next in average_positions_from_C1:
            length_vertebral_levels[label_vert] = average_positions_from_C1[label_vert_next] - average_positions_from_C1[label_vert]
    print length_vertebral_levels
    """
    with bz2.BZ2File(PATH_OUTPUT + '/final_results_2016/' + 'template_vertebral_length_'+contrast+'.pbz2', 'w') as f:
        pickle.dump(length_vertebral_levels, f)


    cmap = get_cmap(len(list_centerline))
    from matplotlib.pyplot import cm
    color = iter(cm.rainbow(np.linspace(0, 1, len(list_centerline))))

    # generate averaged centerline
    plt.figure(1)
    # ax = plt.subplot(211)
    plt.subplot(211)
    for k, centerline in enumerate(list_centerline):
        col = cmap(k)
        col = next(color)
        position_C1 = centerline.points[centerline.index_disk['C1']]
        plt.plot([coord[2] - position_C1[2] for coord in centerline.points], [coord[0] - position_C1[0] for coord in centerline.points], color=col)
        for label_disk in labels_regions:
            if label_disk in centerline.index_disk:
                point = centerline.points[centerline.index_disk[label_disk]]
                plt.scatter(point[2] - position_C1[2], point[0] - position_C1[0], color=col, s=5)

    position_C1 = disk_average_coordinates['C1']
    plt.plot([coord[2] - position_C1[2] for coord in points_average_centerline], [coord[0] - position_C1[0] for coord in points_average_centerline], color='g', linewidth=3)
    for label_disk in labels_regions:
        if label_disk in disk_average_coordinates:
            point = disk_average_coordinates[label_disk]
            plt.scatter(point[2] - position_C1[2], point[0] - position_C1[0], marker='*', color='green', s=25)

    plt.grid()
    MO_array = [[points_average_centerline[i][0], points_average_centerline[i][1], points_average_centerline[i][2]] for i in range(len(points_average_centerline)) if label_points[i] == 'MO']
    PONS_array = [[points_average_centerline[i][0], points_average_centerline[i][1], points_average_centerline[i][2]] for i in range(len(points_average_centerline)) if label_points[i] == 'PONS']
    #plt.plot([coord[2] for coord in MO_array], [coord[0] for coord in MO_array], 'mo')
    #plt.plot([coord[2] for coord in PONS_array], [coord[0] for coord in PONS_array], 'ko')

    color = iter(cm.rainbow(np.linspace(0, 1, len(list_centerline))))

    plt.title("X")
    # ax.set_aspect('equal')
    plt.xlabel('z')
    plt.ylabel('x')
    # ay = plt.subplot(212)
    plt.subplot(212)
    for k, centerline in enumerate(list_centerline):
        col = cmap(k)
        col = next(color)
        position_C1 = centerline.points[centerline.index_disk['C1']]
        plt.plot([coord[2] - position_C1[2] for coord in centerline.points], [coord[1] - position_C1[1] for coord in centerline.points], color=col)
        for label_disk in labels_regions:
            if label_disk in centerline.index_disk:
                point = centerline.points[centerline.index_disk[label_disk]]
                plt.scatter(point[2] - position_C1[2], point[1] - position_C1[1], color=col, s=5)

    position_C1 = disk_average_coordinates['C1']
    plt.plot([coord[2] - position_C1[2] for coord in points_average_centerline], [coord[1] - position_C1[1] for coord in points_average_centerline], color='g', linewidth=3)
    for label_disk in labels_regions:
        if label_disk in disk_average_coordinates:
            point = disk_average_coordinates[label_disk]
            plt.scatter(point[2] - position_C1[2], point[1] - position_C1[1], marker='*', color='green', s=25)

    plt.grid()
    #plt.plot([coord[2] for coord in MO_array], [coord[1] for coord in MO_array], 'mo')
    #plt.plot([coord[2] for coord in PONS_array], [coord[1] for coord in PONS_array], 'ko')
    plt.title("Y")
    # ay.set_aspect('equal')
    plt.xlabel('z')
    plt.ylabel('y')
    plt.show()

    # create final template space
    coord_C1 = disk_average_coordinates['C1']
    position_template_disks = {}
    for disk in average_length:
        if disk in ['MO', 'PONS', 'C1']:
            position_template_disks[disk] = disk_average_coordinates[disk]
        else:
            coord_disk = coord_C1.copy()
            coord_disk[2] -= average_positions_from_C1[disk]
            position_template_disks[disk] = coord_disk

    # change centerline to be straight below C1
    index_C1 = disk_position_in_centerline['C1']
    for i in range(0, len(points_average_centerline)):
        current_label = label_points[i]
        if current_label in average_length:
            length_current_label = average_length[current_label][1]
            relative_position_from_disk = float(i - disk_position_in_centerline[current_label]) / float(number_of_points_between_levels)
            #print i, coord_C1[2], average_positions_from_C1[current_label], length_current_label
            points_average_centerline[i][0] = coord_C1[0]
            if i >= index_C1:
                points_average_centerline[i][1] = coord_C1[1]
                points_average_centerline[i][2] = coord_C1[2] - average_positions_from_C1[current_label] - relative_position_from_disk * length_current_label
            else:
                points_average_centerline[i][1] = coord_C1[1] + (points_average_centerline[i][1] - coord_C1[1]) * 2.0
        else:
            points_average_centerline[i] = None
    points_average_centerline = [x for x in points_average_centerline if x is not None]

    # generate averaged centerline
    plt.figure(1)
    # ax = plt.subplot(211)
    plt.subplot(211)
    for k, centerline in enumerate(list_centerline):
        plt.plot([coord[2] for coord in centerline.points], [coord[0] for coord in centerline.points], 'r')
    plt.plot([coord[2] for coord in points_average_centerline], [coord[0] for coord in points_average_centerline])
    plt.title("X")
    # ax.set_aspect('equal')
    plt.xlabel('z')
    plt.ylabel('x')
    # ay = plt.subplot(212)
    plt.subplot(212)
    for k, centerline in enumerate(list_centerline):
        plt.plot([coord[2] for coord in centerline.points], [coord[1] for coord in centerline.points], 'r')
    plt.plot([coord[2] for coord in points_average_centerline], [coord[1] for coord in points_average_centerline])
    plt.title("Y")
    # ay.set_aspect('equal')
    plt.xlabel('z')
    plt.ylabel('y')
    plt.show()

    # creating template space
    size_template_z = int(abs(points_average_centerline[0][2] - points_average_centerline[-1][2]) / spacing) + 15

    # saving template centerline and levels
    # generate template space
    from msct_image import Image
    from numpy import zeros
    template = Image('/Users/benjamindeleener/code/sct/dev/template_creation/template_landmarks-mm.nii.gz')
    template_space = Image([x_size_of_template_space, y_size_of_template_space, size_template_z])
    template_space.data = zeros((x_size_of_template_space, y_size_of_template_space, size_template_z))
    template_space.hdr = template.hdr
    template_space.hdr.set_data_dtype('float32')
    #origin = [(x_size_of_template_space - 1.0) / 4.0, -(y_size_of_template_space - 1.0) / 4.0, -((size_template_z / 4.0) - spacing)]
    origin = [points_average_centerline[-1][0] + x_size_of_template_space * spacing / 2.0, points_average_centerline[-1][1] - y_size_of_template_space * spacing / 2.0, (points_average_centerline[-1][2] - spacing)]
    print origin
    template_space.hdr.structarr['dim'] = [3.0, x_size_of_template_space, y_size_of_template_space, size_template_z, 1.0, 1.0, 1.0, 1.0]
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
    template_space.setFileName('/Users/benjamindeleener/code/sct/dev/template_creation/template_space.nii.gz')
    template_space.save()

    # generate template centerline
    image_centerline = template_space.copy()
    for coord in points_average_centerline:
        coord_pix = image_centerline.transfo_phys2pix([coord])[0]
        if 0 <= coord_pix[0] < image_centerline.data.shape[0] and 0 <= coord_pix[1] < image_centerline.data.shape[1] and 0 <= coord_pix[2] < image_centerline.data.shape[2]:
            image_centerline.data[int(coord_pix[0]), int(coord_pix[1]), int(coord_pix[2])] = 1
    image_centerline.setFileName('/Users/benjamindeleener/code/sct/dev/template_creation/template_centerline.nii.gz')
    image_centerline.save(type='uint8')

    # generate template disks position
    image_disks = template_space.copy()
    for disk in position_template_disks:
        label = labels_regions[disk]
        coord = position_template_disks[disk]
        coord_pix = image_disks.transfo_phys2pix([coord])[0]
        if 0 <= coord_pix[0] < image_disks.data.shape[0] and 0 <= coord_pix[1] < image_disks.data.shape[1] and 0 <= coord_pix[2] < image_disks.data.shape[2]:
            image_disks.data[int(coord_pix[0]), int(coord_pix[1]), int(coord_pix[2])] = label
        else:
            print 'ERROR: the disk label ' + str(disk) + ' is not in the template image.'
    image_disks.setFileName('/Users/benjamindeleener/code/sct/dev/template_creation/template_disks.nii.gz')
    image_disks.save(type='uint8')


def compute_csa(fname_segmentation, fname_disks='labels_vertebral_crop.nii.gz'):
    labels_regions = {'PONS': 50, 'MO': 51,
                      'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5, 'C6': 6, 'C7': 7,
                      'T1': 8, 'T2': 9, 'T3': 10, 'T4': 11, 'T5': 12, 'T6': 13, 'T7': 14, 'T8': 15, 'T9': 16,
                      'T10': 17, 'T11': 18, 'T12': 19,
                      'L1': 20, 'L2': 21, 'L3': 22, 'L4': 23, 'L5': 24,
                      'S1': 25, 'S2': 26, 'S3': 27, 'S4': 28, 'S5': 29,
                      'Co': 30}

    # compute csa on the input segmentation
    # this function create a csv file (csa_per_slice.txt) containing csa for each slice in the image
    sct.run('sct_process_segmentation '
            '-i ' + fname_segmentation + ' '
            '-p csa')

    # read csv file to extract csa per slice
    csa_file = open('csa_per_slice.txt', 'r')
    csa = csa_file.read()
    csa_file.close()
    csa_lines = csa.split('\n')[1:-1]
    z_values, csa_values = [], []
    for l in csa_lines:
        s = l.split(',')
        z_values.append(int(s[0]))
        csa_values.append(float(s[1]))

    # compute a lookup table with continuous vertebral levels and slice position
    from sct_straighten_spinalcord import smooth_centerline
    from msct_image import Image
    im = Image(fname_disks)
    coord = im.getNonZeroCoordinates(sorting='z', reverse_coord=True)
    coord_physical = []
    for c in coord:
        c_p = im.transfo_pix2phys([[c.x, c.y, c.z]])[0]
        c_p.append(c.value)
        coord_physical.append(c_p)

    number_of_points_in_centerline = 4000
    x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline(
        'generated_centerline.nii.gz',
        algo_fitting='nurbs',
        verbose=0, nurbs_pts_number=number_of_points_in_centerline, all_slices=False, phys_coordinates=True,
        remove_outliers=True)
    from msct_types import Centerline
    centerline = Centerline(x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv,
                            y_centerline_deriv,
                            z_centerline_deriv)

    centerline.compute_vertebral_distribution(coord_physical)
    x, y, z, xd, yd, zd = centerline.average_coordinates_over_slices(im)
    coordinates = []
    for i in range(len(z)):
        nearest_index = centerline.find_nearest_indexes([[x[i], y[i], z[i]]])[0]
        disk_label = centerline.l_points[nearest_index]
        relative_position = centerline.dist_points_rel[nearest_index]
        if disk_label != 0:
            coordinates.append(float(labels_regions[disk_label]) + relative_position)

    # concatenate results
    result_levels, result_csa = [], []
    import numpy as np
    z_pix = [int(im.transfo_phys2pix([[x[k], y[k], z[k]]])[0][2]) for k in range(len(z))]
    for i, zi in enumerate(z_values):
        try:
            corresponding_values = z_pix.index(int(zi))
        except ValueError as e:
            print 'got exception'
            continue

        if coordinates[corresponding_values] <= 30:
            result_levels.append(coordinates[corresponding_values])
            result_csa.append(csa_values[i])

    #print result_levels, result_csa
    plt.plot(result_levels, result_csa)
    plt.show()

    return result_levels, result_csa


def compare_csa(contrast, fname_segmentation, fname_disks='labels_vertebral_crop.nii.gz'):
    list_csa = []

    for i in range(0, len(SUBJECTS_LIST)):
        subject = SUBJECTS_LIST[i][0]
        print '\nGo to output folder ' + PATH_OUTPUT + '/subjects/' + subject + '/' + contrast
        os.chdir(PATH_OUTPUT + '/subjects/' + subject + '/' + contrast)

        levels, csa = compute_csa(fname_segmentation, fname_disks)
        list_csa.append([subject, levels, csa])

    plt.figure()
    for subject in list_csa:
        plt.plot(subject[1], subject[2])
    plt.show()


def straighten_all_subjects(contrast):
    # straightening of each subject on the new template
    for i in range(0, len(SUBJECTS_LIST)):
        subject = SUBJECTS_LIST[i][0]

        # go to output folder
        print '\nGo to output folder ' + PATH_OUTPUT + '/subjects/' + subject + '/' + contrast
        os.chdir(PATH_OUTPUT + '/subjects/' + subject + '/' + contrast)
        sct.run('sct_straighten_spinalcord'
                ' -i data_RPI_crop_normalized.nii.gz'
                ' -s generated_centerline.nii.gz'
                ' -disks-input labels_vertebral_crop.nii.gz'
                ' -ref /Users/benjamindeleener/code/sct/dev/template_creation/template_centerline.nii.gz'
                ' -disks-ref /Users/benjamindeleener/code/sct/dev/template_creation/template_disks.nii.gz'
                ' -disable-straight2curved', verbose=1)

        sct.run('cp data_RPI_crop_normalized_straight.nii.gz ' + PATH_OUTPUT + '/final_results_2016/' + subject + '_final_' + contrast + '.nii.gz')


def convert_nii2mnc(contrast):
    for i in range(0, len(SUBJECTS_LIST)):
        subject = SUBJECTS_LIST[i][0]

        sct.run('nii2mnc ' + PATH_OUTPUT + '/final_results_2016/' + subject + '_final_' + contrast + '.nii.gz ' + PATH_OUTPUT + '/final_results_mnc_2016/' + subject + '_final_' + contrast + '.mnc')

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
        #sct.run(cmd_straighten)

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
        #sct.run('sct_apply_transfo -i labels_vertebral_dilated.nii.gz -d data_RPI_crop_normalized_straight.nii.gz -w warp_curve2straight.nii.gz -x nn')

        # Select center of mass of labels volume due to past dilatation
        # REMOVE IF NOT REQUIRED
        print '\nSelecting center of mass of labels volume due to past dilatation...'
        #sct.run('sct_label_utils -i labels_vertebral_dilated_reg.nii.gz -o labels_vertebral_dilated_reg_2point.nii.gz -cubic-to-point')

        # Apply straightening to seg_and_labels.nii.gz
        print'\nApplying transfo to seg_and_labels.nii.gz ...'
        #sct.run('sct_apply_transfo -i seg_and_labels.nii.gz -d data_RPI_crop_normalized_straight.nii.gz -w warp_curve2straight.nii.gz -x nn')

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
