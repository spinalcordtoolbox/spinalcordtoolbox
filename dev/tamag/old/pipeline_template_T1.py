#!/usr/bin/env python
#
# WHAT DOES IT DO:
# pre-process data for template.
#
# LOCATION OF pipeline_template.sh
# $SCT_DIR/dev/template_preprocessing
#
# HOW TO USE:
# run: pipeline_template.sh
#
# REQUIRED DATA:
# ~/subject/T1/centerline_propseg_RPI.nii.gz --> a series of binary labels along the cord to help propseg. To be done on the image cropped and in RPI orientation ! (Use command: "matlab_batcher.sh sct_get_centerline "'image_RPI_crop.nii.gz'" if image_RPI_crop.nii.gz is your anatomic image, cropped and in RPI orientation)
# ~/subject/T1/crop.txt --> ASCII txt file that indicates zmin and zmax for cropping the anatomic image and the segmentation . Format: zmin_anatomic,zmax_anatomic,zmin_seg,zmax_seg  If there is a need to crop along y axis the RPI image, please specify as follow: zmin_anatomic,zmax_anatomic,zmin_seg,zmax_seg,ymin_anatomic,ymax_anatomic
# ~/subject/T1/labels_updown.nii.gz --> a series of binary labels to complete the centerline from brainstem to L2/L3.
# ~/subject/T1/labels_vertebral.nii.gz --> a series of labels to identify vertebral level(to be done on the original RPI image i.e. non crop image). These are placed on the left side of the vertebral body, at the edge of the cartilage separating two vertebra. The value of the label corresponds to the level. There are 20 labels: [name of point at top] + PMJ + 18 labels of vertebral level going until the frontier T12/L1 I.e., Brainstem [name first label]=1, (PMJ)=2, C2/C3=3, C3/C4=4, C4/C5=5, C5/C6=6, T1/T2=7, T2/T3=8, T3/T4=9 ... T11/T12=19, T12/L1=20.
# cf snapshot in $SCT_DIR/dev/template_preprocessing/snap1, 2, etc.

import os, sys

# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, "scripts"))

import sct_utils as sct
import nibabel
from sct_orientation import get_orientation, set_orientation
from scipy import ndimage
from numpy import array
from msct_image import Image
import matplotlib.pyplot as plt

# add path to scripts
PATH_DICOM = '/Volumes/data_shared/' #sert a rien
PATH_OUTPUT = '/Users/tamag/data/data_template/Results_template' #folder where you want the results to be stored
PATH_INFO = '/Users/tamag/data/data_template/info/template_subjects/T1'  # eventually to be replaced by URL from github

# define subject
# SUBJECTS_LIST=[['errsm_14', ,'pathtodicomt1', 'pathtodicomt2']
SUBJECTS_LIST_test=[['errsm_14', '/Volumes/data_shared/montreal_criugm/errsm_14/5002-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_14/5003-SPINE_T2'], ['errsm_16', '/Volumes/data_shared/montreal_criugm/errsm_16/23-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_16/39-SPINE_T2'], ['errsm_17', '/Volumes/data_shared/montreal_criugm/errsm_17/41-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_17/42-SPINE_T2'], ['errsm_18', '/Volumes/data_shared/montreal_criugm/errsm_18/36-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_18/33-SPINE_T2']]

SUBJECTS_LIST_total = [['errsm_02', '/Volumes/data_shared/montreal_criugm/errsm_02/22-SPINE_T1', '/Volumes/data_shared/montreal_criugm/errsm_02/28-SPINE_T2'],['errsm_04', '/Volumes/data_shared/montreal_criugm/errsm_04/16-SPINE_memprage/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_04/18-SPINE_space'],\
                       ['errsm_05', '/Volumes/data_shared/montreal_criugm/errsm_05/23-SPINE_MEMPRAGE/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_05/24-SPINE_SPACE'],['errsm_09', '/Volumes/data_shared/montreal_criugm/errsm_09/34-SPINE_MEMPRAGE2/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_09/33-SPINE_SPACE'],\
                       ['errsm_10', '/Volumes/data_shared/montreal_criugm/errsm_10/13-SPINE_MEMPRAGE/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_10/20-SPINE_SPACE'], ['errsm_11', '/Volumes/data_shared/montreal_criugm/errsm_11/24-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_11/09-SPINE_T2'],\
                       ['errsm_12', '/Volumes/data_shared/montreal_criugm/errsm_12/19-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_12/18-SPINE_T2'],['errsm_13', '/Volumes/data_shared/montreal_criugm/errsm_13/33-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_13/34-SPINE_T2'],\
                       ['errsm_14', '/Volumes/data_shared/montreal_criugm/errsm_14/5002-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_14/5003-SPINE_T2'], ['errsm_16', '/Volumes/data_shared/montreal_criugm/errsm_16/23-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_16/39-SPINE_T2'],\
                       ['errsm_17', '/Volumes/data_shared/montreal_criugm/errsm_17/41-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_17/42-SPINE_T2'], ['errsm_18', '/Volumes/data_shared/montreal_criugm/errsm_18/36-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_18/33-SPINE_T2'],\
                       ['errsm_21', '/Volumes/data_shared/montreal_criugm/errsm_21/27-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_21/30-SPINE_T2'],['errsm_22', '/Volumes/data_shared/montreal_criugm/errsm_22/29-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_22/25-SPINE_T2'],\
                       ['errsm_23', '/Volumes/data_shared/montreal_criugm/errsm_23/29-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_23/28-SPINE_T2'],['errsm_24', '/Volumes/data_shared/montreal_criugm/errsm_24/20-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_24/24-SPINE_T2'],\
                       ['errsm_25', '/Volumes/data_shared/montreal_criugm/errsm_25/25-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_25/26-SPINE_T2'],['errsm_30', '/Volumes/data_shared/montreal_criugm/errsm_30/51-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_30/50-SPINE_T2'],\
                       ['errsm_31', '/Volumes/data_shared/montreal_criugm/errsm_31/31-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_31/32-SPINE_T2'],['errsm_32', '/Volumes/data_shared/montreal_criugm/errsm_32/16-SPINE_T1/echo_2.09 ', '/Volumes/data_shared/montreal_criugm/errsm_32/19-SPINE_T2'],\
                       ['errsm_33', '/Volumes/data_shared/montreal_criugm/errsm_33/30-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_33/31-SPINE_T2'],['1','/Volumes/data_shared/marseille/ED/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-101','/Volumes/data_shared/marseille/ED/01_0008_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-65'],\
                       ['ALT','/Volumes/data_shared/marseille/ALT/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-15','/Volumes/data_shared/marseille/ALT/01_0100_space-composing'],['JD','/Volumes/data_shared/marseille/JD/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-23','/Volumes/data_shared/marseille/JD/01_0100_compo-space'],\
                       ['JW','/Volumes/data_shared/marseille/JW/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5','/Volumes/data_shared/marseille/JW/01_0100_compo-space'],['MLL','/Volumes/data_shared/marseille/MLL_1016/01_0008_sc-mprage-1mm-2palliers-fov384-comp-sp-7','/Volumes/data_shared/marseille/MLL_1016/01_0100_t2-compo'],\
                       ['MT','/Volumes/data_shared/marseille/MT/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5','/Volumes/data_shared/marseille/MT/01_0100_t2composing'],['T020b', '/Volumes/data_shared/marseille/T020b/01_0010_sc-mprage-1mm-2palliers-fov384-comp-sp-15', None],\
                       ['T047','/Volumes/data_shared/marseille/T047/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5','/Volumes/data_shared/marseille/T047/01_0100_t2-3d-composing'],['VC','/Volumes/data_shared/marseille/VC/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-23','/Volumes/data_shared/marseille/VC/01_0008_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-113'],\
                       ['VG','/Volumes/data_shared/marseille/VG/T1/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-15','/Volumes/data_shared/marseille/VG/T2/01_0024_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-11'],['VP','/Volumes/data_shared/marseille/VP/01_0011_sc-mprage-1mm-2palliers-fov384-comp-sp-25','/Volumes/data_shared/marseille/VP/01_0100_space-compo'],\
                       ['pain_pilot_1','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot1/24-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot1/25-SPINE'],['pain_pilot_2','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot2/13-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot2/30-SPINE_T2'],\
                       ['pain_pilot_4','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot4/33-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot4/32-SPINE_T2'],\
                       ['errsm_20', '/Volumes/data_shared/montreal_criugm/errsm_20/12-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_20/34-SPINE_T2'],['pain_pilot_3','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot3/16-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot3/31-SPINE_T2'],\
                       ['errsm_34','/Volumes/data_shared/montreal_criugm/errsm_34/41-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/errsm_34/40-SPINE_T2'],['errsm_35','/Volumes/data_shared/montreal_criugm/errsm_35/37-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/errsm_35/38-SPINE_T2'],\
                       ['pain_pilot_7', '/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot7/32-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot7/33-SPINE_T2'], ['errsm_03', '/Volumes/data_shared/montreal_criugm/errsm_03/32-SPINE_all/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_03/38-SPINE_all_space'],\
                       ['AP', '/Volumes/data_shared/marseille/AP_T077/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-7', '/Volumes/data_shared/marseille/AP_T077/01_0102_t2comp'],\
                       ['FR', '/Volumes/data_shared/marseille/FR_T080/01_0039_sc-mprage-1mm-3palliers-fov384-comp-sp-13', '/Volumes/data_shared/marseille/FR_T080/01_0104_spine2'],['GB', '/Volumes/data_shared/marseille/GB_T083/01_0029_sc-mprage-1mm-2palliers-fov384-comp-sp-5', '/Volumes/data_shared/marseille/GB_T083/01_0033_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-7'],\
                       ['T045', '/Volumes/data_shared/marseille/T045/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', '/Volumes/data_shared/marseille/T045/01_0101_t2-3d-composing'],['TM', '/Volumes/data_shared/marseille/TM_T057c/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', '/Volumes/data_shared/marseille/TM_T057c/01_0105_t2-composing'],\
                       ['TR', '/Volumes/data_shared/marseille/TR_T076/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', '/Volumes/data_shared/marseille/TR_T076/01_0016_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-19'],['TT', '/Volumes/data_shared/marseille/TT/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-23', '/Volumes/data_shared/marseille/TT/01_0100_compo-space'],\
                       ]




SUBJECTS_LIST_montreal=[['errsm_02', '/Volumes/data_shared/montreal_criugm/errsm_02/28-SPINE_T1', '/Volumes/data_shared/montreal_criugm/errsm_02/28-SPINE_T2'], ['errsm_03', '/Volumes/data_shared/montreal_criugm/errsm_03/32-SPINE_all/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_03/38-SPINE_all_space'],['errsm_04', '/Volumes/data_shared/montreal_criugm/errsm_04/16-SPINE_memprage/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_04/18-SPINE_space'],['errsm_05', '/Volumes/data_shared/montreal_criugm/errsm_05/23-SPINE_MEMPRAGE/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_05/24-SPINE_SPACE'],['errsm_09', '/Volumes/data_shared/montreal_criugm/errsm_09/34-SPINE_MEMPRAGE2', '/Volumes/data_shared/montreal_criugm/errsm_09/33-SPINE_SPACE'],['errsm_10', '/Volumes/data_shared/montreal_criugm/errsm_10/13-SPINE_MEMPRAGE/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_10/20-SPINE_SPACE'], ['errsm_11', '/Volumes/data_shared/montreal_criugm/errsm_11/24-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_11/09-SPINE_T2'],['errsm_12', '/Volumes/data_shared/montreal_criugm/errsm_12/19-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_12/18-SPINE_T2'],['errsm_13', '/Volumes/data_shared/montreal_criugm/errsm_13/33-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_13/34-SPINE_T2'],['errsm_14', '/Volumes/data_shared/montreal_criugm/errsm_14/5002-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_14/5003-SPINE_T2'], ['errsm_16', '/Volumes/data_shared/montreal_criugm/errsm_16/23-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_16/39-SPINE_T2'], ['errsm_17', '/Volumes/data_shared/montreal_criugm/errsm_17/41-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_17/42-SPINE_T2'], ['errsm_18', '/Volumes/data_shared/montreal_criugm/errsm_18/36-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_18/33-SPINE_T2'],['errsm_21', '/Volumes/data_shared/montreal_criugm/errsm_21/27-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_21/30-SPINE_T2'],['errsm_22', '/Volumes/data_shared/montreal_criugm/errsm_22/29-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_22/25-SPINE_T2'],['errsm_23', '/Volumes/data_shared/montreal_criugm/errsm_23/29-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_23/28-SPINE_T2'],['errsm_24', '/Volumes/data_shared/montreal_criugm/errsm_24/20-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_24/24-SPINE_T2'],['errsm_25', '/Volumes/data_shared/montreal_criugm/errsm_25/25-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_25/26-SPINE_T2'],['errsm_26', None, '/Volumes/data_shared/montreal_criugm/errsm_26/31-SPINE_T2'],['errsm_30', '/Volumes/data_shared/montreal_criugm/errsm_30/51-SPINE_T1', '/Volumes/data_shared/montreal_criugm/errsm_30/50-SPINE_T2'],['errsm_31', '/Volumes/data_shared/montreal_criugm/errsm_31/31-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_31/32-SPINE_T2'],['errsm_32', '/Volumes/data_shared/montreal_criugm/errsm_32/16-SPINE_T1/echo_2.09 ', '/Volumes/data_shared/montreal_criugm/errsm_32/19-SPINE_T2'],['errsm_33', '/Volumes/data_shared/montreal_criugm/errsm_33/30-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_33/31-SPINE_T2'],['pain_pilot_1','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot1/25-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot1/25-SPINE'],['pain_pilot_2','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot2/13-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot2/30-SPINE_T2'],['pain_pilot_3','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot3/16-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot3/31-SPINE_T2'],['pain_pilot_4','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot4/33-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot4/32-SPINE_T2']]
SUBJECTS_LIST_marseille = [['1','/Volumes/data_shared/marseille/ED/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-101','/Volumes/data_shared/marseille/ED/01_0008_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-65'],['ALT','/Volumes/data_shared/marseille/ALT/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-15','/Volumes/data_shared/marseille/ALT/01_0100_space-composing'],['JD','/Volumes/data_shared/marseille/JD/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-23','/Volumes/data_shared/marseille/JD/01_0100_compo-space'],['JW','/Volumes/data_shared/marseille/JW/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5','/Volumes/data_shared/marseille/JW/01_0100_compo-space'],['MD','/Volumes/data_shared/marseille/MD_T075/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-7','/Volumes/data_shared/marseille/MD_T075/01_0100_t2-compo'],['MLL','/Volumes/data_shared/marseille/MLL_1016/01_0008_sc-mprage-1mm-2palliers-fov384-comp-sp-7','/Volumes/data_shared/marseille/MLL_1016/01_0100_t2-compo'],['MT','/Volumes/data_shared/marseille/MT/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5','/Volumes/data_shared/marseille/MT/01_0100_t2composing'],['T047','/Volumes/data_shared/marseille/T047/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5','/Volumes/data_shared/marseille/T047/01_0100_t2-3d-composing'],['VC','/Volumes/data_shared/marseille/VC/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-23','/Volumes/data_shared/marseille/VC/01_0008_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-113'],['VG','/Volumes/data_shared/marseille/VG/T1/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-15','/Volumes/data_shared/marseille/VG/T2/01_0024_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-11'],['VP','/Volumes/data_shared/marseille/VP/01_0011_sc-mprage-1mm-2palliers-fov384-comp-sp-25','/Volumes/data_shared/marseille/VP/01_0100_space-compo']]

SUBJECTS_LIST_TO_ADD = [['errsm_20', '/Volumes/data_shared/montreal_criugm/errsm_20/12-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_20/34-SPINE_T2'],['pain_pilot_3','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot3/16-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot3/31-SPINE_T2'],\
                        ['errsm_34','/Volumes/data_shared/montreal_criugm/errsm_34/41-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/errsm_34/40-SPINE_T2'],['errsm_35','/Volumes/data_shared/montreal_criugm/errsm_35/37-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/errsm_35/38-SPINE_T2'],\
                        ['pain_pilot_7', '/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot7/32-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot7/33-SPINE_T2'], ['errsm_03', '/Volumes/data_shared/montreal_criugm/errsm_03/32-SPINE_all/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_03/38-SPINE_all_space'],\
                        ['AP', '/Volumes/data_shared/marseille/AP_T077/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-7', '/Volumes/data_shared/marseille/AP_T077/01_0102_t2comp'],\
                        ['FR', '/Volumes/data_shared/marseille/FR_T080/01_0039_sc-mprage-1mm-3palliers-fov384-comp-sp-13', '/Volumes/data_shared/marseille/FR_T080/01_0104_spine2'],['GB', '/Volumes/data_shared/marseille/GB_T083/01_0029_sc-mprage-1mm-2palliers-fov384-comp-sp-5', '/Volumes/data_shared/marseille/GB_T083/01_0033_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-7'],\
                        ['T045', '/Volumes/data_shared/marseille/T045/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', '/Volumes/data_shared/marseille/T045/01_0101_t2-3d-composing'],['TM', '/Volumes/data_shared/marseille/TM_T057c/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', '/Volumes/data_shared/marseille/TM_T057c/01_0105_t2-composing'],\
                        ['TR', '/Volumes/data_shared/marseille/TR_T076/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', '/Volumes/data_shared/marseille/TR_T076/01_0016_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-19'],['TT', '/Volumes/data_shared/marseille/TT/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-23', '/Volumes/data_shared/marseille/TT/01_0100_compo-space'],\
                        ['T020b', '/Volumes/data_shared/marseille/T020b/01_0010_sc-mprage-1mm-2palliers-fov384-comp-sp-15', None]]

SUBJECTS_LIST_BUG = [['errsm_20', '/Volumes/data_shared/montreal_criugm/errsm_20/12-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_20/34-SPINE_T2'],['pain_pilot_3','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot3/16-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot3/31-SPINE_T2']]

SUBJECTS_LIST_STR = [['errsm_26', None, '/Volumes/data_shared/montreal_criugm/errsm_26/31-SPINE_T2'],['FL', '/Volumes/data_shared/marseille/FL_T056b/01_0044_sc-mprage-1mm-2palliers-fov384-comp-sp-5', '/Volumes/data_shared/marseille/FL_T056b/01_0049_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-7'],['MD','/Volumes/data_shared/marseille/MD_T075/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-7','/Volumes/data_shared/marseille/MD_T075/01_0100_t2-compo']]

SUBJECTS_LIST = SUBJECTS_LIST_total
# add path to scripts
#export PATH=${PATH}:$SCT_DIR/dev/template_creation
#export PATH_OUTPUT=/Users/tamag/data/template/
#export PATH_DICOM='/Volumes/data_shared/'
do_preprocessing_T1 = 0
create_cross = 0
normalize_levels_T1 = 0
average_level = 0
align_vertebrae_T1 = 1

number_labels_for_template = 20

if do_preprocessing_T1:
   # Create folder to gather all labels_vertebral.nii.gz files
    if not os.path.isdir(PATH_OUTPUT + '/'+'labels_vertebral_T1'):
        os.makedirs(PATH_OUTPUT + '/'+'labels_vertebral_T1')

   # loop across subjects
    for i in range(0,len(SUBJECTS_LIST)):
        subject = SUBJECTS_LIST[i][0]

        # create and go to output folder
        print '\nCreate -if not existing- and go to output folder '+ PATH_OUTPUT + '/subjects/'+subject+'/'+'T1'
        if not os.path.isdir(PATH_OUTPUT + '/subjects/'+subject):
            os.makedirs(PATH_OUTPUT + '/subjects/'+subject)
        if not os.path.isdir(PATH_OUTPUT + '/subjects/'+subject+'/'+'T1'):
            os.makedirs(PATH_OUTPUT + '/subjects/'+subject+'/'+'T1')
        os.chdir(PATH_OUTPUT + '/subjects/'+subject+'/'+'T1')

        # # convert to nii
        # print '\nConvert to nii'
        # sct.run('dcm2nii -o . -r N ' + SUBJECTS_LIST[i][1] + '/*.dcm')
        #
        # # change file name
        # print '\nChange file name to data.nii.gz...'
        # sct.run('mv *.nii.gz data.nii.gz')

        # Convert to RPI
        # Input:
        # - data.nii.gz
        # - data_RPI.nii.gz
        print '\nConvert to RPI'
        orientation = get_orientation('data.nii.gz')
        sct.run('sct_orientation -i data.nii.gz -s RPI')

        # Get info from txt file
        print '\nRecover infos from text file' + PATH_INFO + '/' + subject+ '/' + 'crop.txt\n'
        file_name = 'crop.txt'
        os.chdir(PATH_INFO + '/' + subject)
        file_results = open(PATH_INFO+ '/' +subject+ '/' +file_name, 'r')
        ymin_anatomic = None
        ymax_anatomic = None
        for line in file_results:
            line_list = line.split(',')
            zmin_anatomic = line.split(',')[0]
            zmax_anatomic = line.split(',')[1]
            zmin_seg = line.split(',')[2]
            zmax_seg = line.split(',')[3]
            if len(line_list)==6:
                ymin_anatomic = line.split(',')[4]
                ymax_anatomic = line.split(',')[5]
        file_results.close()

        os.chdir(PATH_OUTPUT + '/subjects/'+subject+'/'+'T1')

        # Crop image
        if ymin_anatomic == None and ymax_anatomic == None:
            sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start ' + zmin_anatomic + ' -end ' + zmax_anatomic )
        else: sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 1,2 -start ' + ymin_anatomic +','+zmin_anatomic+ ' -end ' + ymax_anatomic+','+zmax_anatomic )

        # propseg
        # input:
        # - data_RPI_crop.nii.gz
        # - centerline_propseg_RPI.nii.gz
        # output:
        # - data_RPI_crop_seg.nii.gz
        print '\nExtracting segmentation...'
        list_dir = os.listdir(PATH_INFO + '/'+subject)
        centerline_proseg = False
        for k in range(len(list_dir)):
            if list_dir[k] == 'centerline_propseg_RPI.nii.gz':
                centerline_proseg = True
        if centerline_proseg == True:
            sct.printv('sct_propseg -i data_RPI_crop.nii.gz -t t1 -init-centerline ' + PATH_INFO + '/' + subject + '/centerline_propseg_RPI.nii.gz')
            sct.run('sct_propseg -i data_RPI_crop.nii.gz -t t1 -init-centerline ' + PATH_INFO + '/' + subject + '/centerline_propseg_RPI.nii.gz')
        else:
            sct.printv('sct_propseg -i data_RPI_crop.nii.gz -t t1')
            sct.run('sct_propseg -i data_RPI_crop.nii.gz -t t1')

        # Erase 3 top and 3 bottom slices of the segmentation to avoid edge effects  (Done because propseg tends to diverge on edges)
        print '\nErasing 3 top and 3 bottom slices of the segmentation to avoid edge effects...'
        path_seg, file_seg, ext_seg = sct.extract_fname('data_RPI_crop_seg.nii.gz')
        image_seg = nibabel.load('data_RPI_crop_seg.nii.gz')
        nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension('data_RPI_crop_seg.nii.gz')
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
        # - data_crop_denoised_seg.nii.gz
        # - crop.txt
        # output:
        # - data_crop_denoised_seg_crop.nii.gz
        print '\nCrop segmentation...'
        if zmax_seg == 'max':
            nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension('data_RPI_crop_seg.nii.gz')
            sct.printv('sct_crop_image -i data_RPI_crop_seg_mod.nii.gz -o data_RPI_crop_seg_mod_crop.nii.gz -start ' + zmin_seg + ' -end ' + str(nz) + ' -dim 2 -b 0')
            os.system('sct_crop_image -i data_RPI_crop_seg_mod.nii.gz -o data_RPI_crop_seg_mod_crop.nii.gz -start ' + zmin_seg + ' -end ' + str(nz) + ' -dim 2 -b 0')
        else: sct.run('sct_crop_image -i data_RPI_crop_seg_mod.nii.gz -o data_RPI_crop_seg_mod_crop.nii.gz -start ' + zmin_seg + ' -end ' + zmax_seg + ' -dim 2 -b 0')

        print '\n Concatenating segmentation and label files...'
        labels_updown = False
        list_file_info = os.listdir(PATH_INFO+'/'+subject)
        for k in range(0,len(list_file_info)):
            if list_file_info[k] == 'labels_updown.nii.gz':
                labels_updown = True
        if labels_updown:
            sct.run('fslmaths data_RPI_crop_seg_mod_crop.nii.gz -add '+ PATH_INFO + '/' + subject + '/labels_updown.nii.gz seg_and_labels.nii.gz')
        else: sct.run('fslmaths data_RPI_crop_seg_mod_crop.nii.gz -add '+ PATH_INFO + '/' + subject + '/centerline_propseg_RPI.nii.gz seg_and_labels.nii.gz')


        # straighten image using centerline
        # function: sct_straighten_spinalcord (option: hanning)
        # input:
        # - data_crop_denoised.nii.gz
        # - centerline.nii.gz
        # output:
        # - warp_curve2straight.nii.gz
        # - data_crop_denoised_straight.nii.gz
        print '\nStraighten image using centerline'

        sct.printv('sct_straighten_spinalcord -i data_RPI_crop.nii.gz -c ' + PATH_OUTPUT + '/subjects/' + subject + '/T1/seg_and_labels.nii.gz -a nurbs')
        os.system('sct_straighten_spinalcord -i data_RPI_crop.nii.gz -c ' + PATH_OUTPUT + '/subjects/' + subject + '/T1/seg_and_labels.nii.gz -a nurbs')
        #
        # # normalize intensity
        print '\nNormalizing intensity of the straightened image...'
        sct.printv('sct_normalize.py -i data_RPI_crop_straight.nii.gz')
        os.system('sct_normalize.py -i data_RPI_crop_straight.nii.gz')

        # Crop labels_vertebral file
        print '\nCroping labels_vertebral file...'
        if ymin_anatomic == None and ymax_anatomic == None:
            sct.run('sct_crop_image -i '+PATH_INFO + '/' + subject+ '/labels_vertebral.nii.gz -o labels_vertebral_crop.nii.gz -start ' + zmin_anatomic + ' -end ' + zmax_anatomic + ' -dim 2')
        else: sct.run('sct_crop_image -i '+PATH_INFO + '/' + subject+ '/labels_vertebral.nii.gz -o labels_vertebral_crop.nii.gz -start ' + ymin_anatomic+','+zmin_anatomic + ' -end ' + ymax_anatomic+','+ zmax_anatomic + ' -dim 1,2')
        #Dilate labels from labels_vertebral file
        print '\nDilating labels from labels_vertebral file...'
        sct.run('fslmaths '+ PATH_OUTPUT + '/subjects/' + subject+ '/T1/labels_vertebral_crop.nii.gz -dilF labels_vertebral_dilated.nii.gz')

        # apply straightening to centerline and to labels_vertebral.nii.gz
        # function: sct_apply_transfo
        # input:
        # - centerline.nii.gz + labels_vertebral.nii.gz
        # - warp_curve2straight.nii.gz
        # output:
        # - centerline_straight.nii.gz
        print '\nApply straightening to labels_vertebral_dilated.nii.gz'
        sct.printv('sct_apply_transfo -i labels_vertebral_dilated.nii.gz -d data_RPI_crop_straight_normalized.nii.gz -w warp_curve2straight.nii.gz -x nn')
        os.system('sct_apply_transfo -i labels_vertebral_dilated.nii.gz -d data_RPI_crop_straight_normalized.nii.gz -w warp_curve2straight.nii.gz -x nn')

        # Select center of mass of labels volume due to past dilatation
        sct.run('sct_label_utils -i labels_vertebral_dilated_reg.nii.gz -o labels_vertebral_dilated_reg_2point.nii.gz -t cubic-to-point')

        # Apply transfo to seg_and_labels.nii.gz which replace the centerline file
        print'\nApplying transfo to seg_and_labels.nii.gz ...'
        sct.run('sct_apply_transfo -i seg_and_labels.nii.gz -d data_RPI_crop_straight_normalized.nii.gz -w warp_curve2straight.nii.gz -x nn')

        #Calculate the extrem non zero points of the straightened centerline file

        file = nibabel.load('seg_and_labels_reg.nii.gz')
        data_c = file.get_data()
        hdr = file.get_header()

        # Get center of mass of the centerline
        nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension('seg_and_labels_reg.nii.gz')
        z_centerline = [iz for iz in range(0, nz, 1) if data_c[:,:,iz].any() ]
        nz_nonz = len(z_centerline)
        x_centerline = [0 for iz in range(0, nz_nonz, 1)]
        y_centerline = [0 for iz in range(0, nz_nonz, 1)]
        for iz in xrange(len(z_centerline)):
           x_centerline[iz], y_centerline[iz] = ndimage.measurements.center_of_mass(array(data_c[:,:,z_centerline[iz]]))

        X,Y,Z = (data_c>0).nonzero()

        x_max,y_max = (data_c[:,:,max(Z)]).nonzero()
        x_max = x_max[0]
        y_max = y_max[0]
        z_max = max(Z)

        x_min,y_min = (data_c[:,:,min(Z)]).nonzero()
        x_min = x_min[0]
        y_min = y_min[0]
        z_min = min(Z)



        # Crop image one last time and create cross to push into template space
        print'\nCrop image one last time and create cross to push into template space...'
        os.system('sct_crop_image -i data_RPI_crop_straight_normalized.nii.gz -o data_RPI_crop_straight_normalized_crop.nii.gz -dim 2 -start '+ str(z_min)+' -end '+ str(z_max))

        # Crop labels_vertebral_reg.nii.gz and create cross to push into template space
        print'\nCrop labels_vertebral_reg.nii.gz and use cross to push into template space...'
        os.system('sct_crop_image -i labels_vertebral_dilated_reg_2point.nii.gz -o labels_vertebral_dilated_reg_2point_crop.nii.gz -dim 2 -start '+ str(z_min)+' -end '+ str(z_max))


# Define list to gather all distances
list_distances_1 = []
list_distances_2 = []


if create_cross:
    for i in range(0,len(SUBJECTS_LIST)):
        subject = SUBJECTS_LIST[i][0]

        # go to output folder
        print '\nGo to output folder '+ PATH_OUTPUT + '/subjects/'+subject+'/'+'T1'
        os.chdir(PATH_OUTPUT + '/subjects/'+subject+'/'+'T1')

        #Calculate distances between : (last_label and bottom)  and (first label and top)
        img_label = nibabel.load('labels_vertebral_dilated_reg_2point_crop.nii.gz')
        data_labels = img_label.get_data()
        nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension('labels_vertebral_dilated_reg_2point_crop.nii.gz')
        X, Y, Z = (data_labels > 0).nonzero()
        list_coordinates = [([X[i], Y[i], Z[i], data_labels[X[i], Y[i], Z[i]]]) for i in range(0, len(X))]
        for i in range(len(list_coordinates)):
            if list_coordinates[i][3] == 1:
                coordinates_first_label = list_coordinates[i] #top label
            if list_coordinates[i][3] == 20:
                coordinates_last_label = list_coordinates[i] #bottom label
        # Distance 1st label top
        distance_1 = nz - 1 - coordinates_first_label[2] # distance from top image to top label
        distance_2 = nz - 1 - coordinates_last_label[2] # distance from top image to bottom label

        # Complete list to gather all distances
        list_distances_1.append(distance_1)
        list_distances_2.append(distance_2)


        #Create a cross on each subject at first and last labels (modified create cross to do so)
        #os.system('sct_create_cross.py -i data_RPI_crop_straight_normalized_crop.nii.gz -x ' +str(x_min)+' -y '+str(y_min))
        os.system('sct_create_cross.py -i data_RPI_crop_straight_normalized_crop.nii.gz -x ' +str(int(round(nx/2.0)))+' -y '+str(int(round(ny/2.0)))+ ' -s '+str(coordinates_last_label[2])+ ' -e '+ str(coordinates_first_label[2]))

        # Write into a txt file the list of distances
        # os.chdir('../')
        # f_distance = open('list_distances.txt', 'w')
        # f_distance.write(str(distance_1))
        # f_distance.write(' ')
        # f_distance.write(str(distance_2))
        # f_distance.write('\n')

# # Calculate mean cross height for template and create file of reference
# mean_distance_1 = int(round(sum(list_distances_1)/len(list_distances_1))) # mean distance from top image to top label
# mean_distance_2 = int(round(sum(list_distances_2)/len(list_distances_2))) # mean distance from top image to bottom label
# L = 1100 - 2*mean_distance_2 # mean position of top label
# H = 1100 - 2*mean_distance_1 # mean position of bottom label
# os.chdir('/Users/tamag/code/spinalcordtoolbox/dev/template_creation')
# # Create a cross for the template at first and last labels
# os.system('sct_create_cross.py -i template_landmarks-mm.nii.gz -x ' +str(100)+' -y '+str(100)+ ' -s '+str(L)+ ' -e '+ str(H))



if normalize_levels_T1:
    for i in range(0,len(SUBJECTS_LIST)):
        subject = SUBJECTS_LIST[i][0]

        # go to output folder
        print '\nGo to output folder '+ PATH_OUTPUT + '/subjects/'+subject+'/'+'T1'
        os.chdir(PATH_OUTPUT + '/subjects/'+subject+'/'+'T1')

        # Push into template
        print'\nPush into template space...'
        sct.run('sct_push_into_template_space.py -i data_RPI_crop_straight_normalized_crop.nii.gz -n landmark_native.nii.gz')
        sct.run('sct_push_into_template_space.py -i labels_vertebral_dilated_reg_2point_crop.nii.gz -n landmark_native.nii.gz -a nn')

        # Apply cubic to point to the label file as it now presents cubic group of labels instead of discrete labels
        sct.run('sct_label_utils -i labels_vertebral_dilated_reg_2point_crop_2temp.nii.gz -o labels_vertebral_dilated_reg_2point_crop_2temp.nii.gz -t cubic-to-point')
        #os.rename('labels.nii.gz', 'labels_vertebral_straight_in_template_space.nii.gz')

        # Copy labels_vertebral_straight_in_template_space.nii.gz into a folder that will contain each subject labels_vertebral_straight_in_template_space.nii.gz file and rename them
        # check if forlder exists and if not create it
        print'\nCheck if forlder '+PATH_OUTPUT +'/labels_vertebral_T1 exists and if not creates it ...'
        if not os.path.isdir(PATH_OUTPUT +'/labels_vertebral_T1'):
            os.makedirs(PATH_OUTPUT + '/labels_vertebral_T1')

        sct.run('cp labels_vertebral_dilated_reg_2point_crop_2temp.nii.gz '+PATH_OUTPUT +'/labels_vertebral_T1')
        os.chdir(PATH_OUTPUT +'/labels_vertebral_T1')
        os.rename('labels_vertebral_dilated_reg_2point_crop_2temp.nii.gz', subject+'.nii.gz')

# Check position of labels crop_2temp with image crop_2temp
# if no good: check position of labels reg with image straight_normalized
# if no good: check position of labels dilated with image crop


# Calculate mean labels and save it into
if average_level:
    print '\nGo to output folder '+ PATH_OUTPUT + '/labels_vertebral_T1\n'
    os.chdir(PATH_OUTPUT +'/labels_vertebral_T1')
    print'\nCalculate mean of files labels_vertebral and save it into '+PATH_OUTPUT +'/labels_vertebral_T1'
    template_shape = path_sct + '/dev/template_creation/template_shape.nii.gz'
    sct.run('sct_average_levels.py -i ' +PATH_OUTPUT +'/labels_vertebral_T1 -t '+ template_shape +' -n '+ str(number_labels_for_template))


# Aligning vertebrae for all subject

if align_vertebrae_T1:
    for i in range(0,len(SUBJECTS_LIST)):
        subject = SUBJECTS_LIST[i][0]

        # go to output folder
        print '\nGo to output folder '+ PATH_OUTPUT + '/subjects/'+subject+'/'+'T1\n'
        os.chdir(PATH_OUTPUT + '/subjects/'+subject+'/'+'T1')

        # print '\nAligning vertebrae for subject '+subject+'...'
        # sct.printv('\nsct_align_vertebrae.py -i data_RPI_crop_straight_normalized_crop_2temp.nii.gz -l ' + PATH_OUTPUT + '/subjects/' + subject + '/T1/labels_vertebral_dilated_reg_2point_crop_2temp.nii.gz -R ' +PATH_OUTPUT +'/labels_vertebral_T1/template_landmarks.nii.gz -o '+ subject+'_aligned.nii.gz -t SyN -w spline')
        # os.system('sct_align_vertebrae.py -i data_RPI_crop_straight_normalized_crop_2temp.nii.gz -l ' + PATH_OUTPUT + '/subjects/' + subject + '/T1/labels_vertebral_dilated_reg_2point_crop_2temp.nii.gz -R ' +PATH_OUTPUT +'/labels_vertebral_T1/template_landmarks.nii.gz -o '+ subject+'_aligned.nii.gz -t SyN -w spline')
        #
        # #Normalize intensity of result
        # print'\nNormalizing intensity of results...'
        # sct.run('sct_normalize.py -i '+subject+'_aligned.nii.gz')
        #
        # #Warning that results for the subject is ready
        # print'\nThe results for subject '+subject+' are ready. You can visualize them by tapping: fslview '+subject+'_aligned_normalized.nii.gz'
        #
        # #Copy final results into final results
        # if not os.path.isdir(PATH_OUTPUT +'/Final_results'):
        #     os.makedirs(PATH_OUTPUT +'/Final_results')
        # sct.run('cp '+subject+'_aligned.nii.gz ' +PATH_OUTPUT +'/Final_results/'+subject+'_aligned_T1.nii.gz')
        # sct.run('cp '+subject+'_aligned_normalized.nii.gz ' +PATH_OUTPUT +'/Final_results/'+subject+'_aligned_normalized_T1.nii.gz')

        #Save png images into a different folder
        if not os.path.isdir(PATH_OUTPUT +'/Image_results'):
            os.makedirs(PATH_OUTPUT +'/Image_results')
        f = nibabel.load(PATH_OUTPUT +'/Final_results/'+subject+'_aligned_normalized.nii.gz')
        data = f.get_data()
        nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(PATH_OUTPUT +'/Final_results/'+subject+'_aligned_normalized.nii.gz')
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
        #plt.show()
        fig1.savefig(PATH_OUTPUT +'/Image_results'+'/'+subject+'_aligned_normalized.png', format='png')
