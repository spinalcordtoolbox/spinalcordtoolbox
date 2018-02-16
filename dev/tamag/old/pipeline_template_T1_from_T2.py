#!/usr/bin/env python
#
# WHAT DOES IT DO:
# pre-process data for template T1.
#
# LOCATION OF pipeline_template_T1.sh
# $SCT_DIR/dev/template_preprocessing
#
# HOW TO USE:
# run: pipeline_template_T1.sh
#
# REQUIRED DATA:
# ~/subject/t2/centerline_propseg_RPI.nii.gz --> a series of binary labels along the cord to help propseg. To be done on the image cropped and in RPI orientation ! (2 solutions: save manually a serie of labels along the spinalcord or use command: "matlab_batcher.sh sct_get_centerline "'image_RPI_crop.nii.gz'" if image_RPI_crop.nii.gz is your anatomic image, cropped and in RPI orientation)
# ~/subject/t2/crop.txt --> ASCII txt file that indicates zmin and zmax for cropping the anatomic image and the segmentation . Format: zmin_anatomic,zmax_anatomic,zmin_seg,zmax_seg  If there is a need to crop along y axis the RPI image, please specify as follow: zmin_anatomic,zmax_anatomic,zmin_seg,zmax_seg,ymin_anatomic,ymax_anatomic
# ~/subject/t2/labels_updown.nii.gz --> a series of binary labels to complete the segmentation.
# ~/subject/t2/labels_vertebral.nii.gz --> a series of labels to identify vertebral level. These are placed on the left side of the vertebral body, at the edge of the cartilage separating two vertebra. The value of the label corresponds to the level. There are 19 labels from PMJ to the frontier T12/L1 I.e., Brainstem (PMJ)=1, C2/C3=2, C3/C4=3, C4/C5=4, C5/C6=5, T1/T2=6, T2/T3=7, T3/T4=8 ... T11/T12=18, T12/L1=19.
# cf snapshot in $SCT_DIR/dev/template_preprocessing/snap1, 2, etc.

import os, sys

# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, "scripts"))
sys.path.append(path_sct + '/dev/tamag')

import sct_utils as sct
import nibabel
from sct_orientation import get_orientation, set_orientation
from scipy import ndimage
from numpy import array
from msct_image import Image
from msct_register_reg import register_images,register_seg,generate_warping_field

# add path to scripts
PATH_DICOM= '/Volumes/data_shared/' #sert a rien
PATH_OUTPUT= '/Users/tamag/data/data_template/Results_template' #folder where you want the results to be stored
PATH_INFO = '/Users/tamag/data/data_template/info/template_subjects/T1'  # to be replaced by URL from github

# define subject
# SUBJECTS_LIST=[['errsm_14', ,'pathtodicomt1', 'pathtodicomt2']
SUBJECTS_LIST_test=[['errsm_14', '/Volumes/data_shared/montreal_criugm/errsm_14/5002-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_14/5003-SPINE_T2'], ['errsm_16', '/Volumes/data_shared/montreal_criugm/errsm_16/23-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_16/39-SPINE_T2'], ['errsm_17', '/Volumes/data_shared/montreal_criugm/errsm_17/41-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_17/42-SPINE_T2'], ['errsm_18', '/Volumes/data_shared/montreal_criugm/errsm_18/36-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_18/33-SPINE_T2']]
SUBJECTS_LIST_total = [['errsm_02', '/Volumes/data_shared/montreal_criugm/errsm_02/22-SPINE_T1', '/Volumes/data_shared/montreal_criugm/errsm_02/28-SPINE_T2'],['errsm_04', '/Volumes/data_shared/montreal_criugm/errsm_04/16-SPINE_memprage/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_04/18-SPINE_space'],['errsm_05', '/Volumes/data_shared/montreal_criugm/errsm_05/23-SPINE_MEMPRAGE/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_05/24-SPINE_SPACE'],['errsm_09', '/Volumes/data_shared/montreal_criugm/errsm_09/34-SPINE_MEMPRAGE2/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_09/33-SPINE_SPACE'],['errsm_10', '/Volumes/data_shared/montreal_criugm/errsm_10/13-SPINE_MEMPRAGE/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_10/20-SPINE_SPACE'], ['errsm_11', '/Volumes/data_shared/montreal_criugm/errsm_11/24-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_11/09-SPINE_T2'],['errsm_12', '/Volumes/data_shared/montreal_criugm/errsm_12/19-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_12/18-SPINE_T2'],['errsm_13', '/Volumes/data_shared/montreal_criugm/errsm_13/33-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_13/34-SPINE_T2'],['errsm_14', '/Volumes/data_shared/montreal_criugm/errsm_14/5002-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_14/5003-SPINE_T2'], ['errsm_16', '/Volumes/data_shared/montreal_criugm/errsm_16/23-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_16/39-SPINE_T2'], ['errsm_17', '/Volumes/data_shared/montreal_criugm/errsm_17/41-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_17/42-SPINE_T2'], ['errsm_18', '/Volumes/data_shared/montreal_criugm/errsm_18/36-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_18/33-SPINE_T2'],['errsm_21', '/Volumes/data_shared/montreal_criugm/errsm_21/27-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_21/30-SPINE_T2'],['errsm_22', '/Volumes/data_shared/montreal_criugm/errsm_22/29-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_22/25-SPINE_T2'],['errsm_23', '/Volumes/data_shared/montreal_criugm/errsm_23/29-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_23/28-SPINE_T2'],['errsm_24', '/Volumes/data_shared/montreal_criugm/errsm_24/20-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_24/24-SPINE_T2'],['errsm_25', '/Volumes/data_shared/montreal_criugm/errsm_25/25-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_25/26-SPINE_T2'],['errsm_30', '/Volumes/data_shared/montreal_criugm/errsm_30/51-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_30/50-SPINE_T2'],['errsm_31', '/Volumes/data_shared/montreal_criugm/errsm_31/31-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_31/32-SPINE_T2'],['errsm_32', '/Volumes/data_shared/montreal_criugm/errsm_32/16-SPINE_T1/echo_2.09 ', '/Volumes/data_shared/montreal_criugm/errsm_32/19-SPINE_T2'],['errsm_33', '/Volumes/data_shared/montreal_criugm/errsm_33/30-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_33/31-SPINE_T2'],['1','/Volumes/data_shared/marseille/ED/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-101','/Volumes/data_shared/marseille/ED/01_0008_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-65'],['ALT','/Volumes/data_shared/marseille/ALT/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-15','/Volumes/data_shared/marseille/ALT/01_0100_space-composing'],['JD','/Volumes/data_shared/marseille/JD/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-23','/Volumes/data_shared/marseille/JD/01_0100_compo-space'],['JW','/Volumes/data_shared/marseille/JW/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5','/Volumes/data_shared/marseille/JW/01_0100_compo-space'],['MD','/Volumes/data_shared/marseille/MD_T075/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-7','/Volumes/data_shared/marseille/MD_T075/01_0100_t2-compo'],['MLL','/Volumes/data_shared/marseille/MLL_1016/01_0008_sc-mprage-1mm-2palliers-fov384-comp-sp-7','/Volumes/data_shared/marseille/MLL_1016/01_0100_t2-compo'],['MT','/Volumes/data_shared/marseille/MT/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5','/Volumes/data_shared/marseille/MT/01_0100_t2composing'],['T047','/Volumes/data_shared/marseille/T047/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5','/Volumes/data_shared/marseille/T047/01_0100_t2-3d-composing'],['VC','/Volumes/data_shared/marseille/VC/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-23','/Volumes/data_shared/marseille/VC/01_0008_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-113'],['VG','/Volumes/data_shared/marseille/VG/T1/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-15','/Volumes/data_shared/marseille/VG/T2/01_0024_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-11'],['VP','/Volumes/data_shared/marseille/VP/01_0011_sc-mprage-1mm-2palliers-fov384-comp-sp-25','/Volumes/data_shared/marseille/VP/01_0100_space-compo'],['pain_pilot_1','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot1/24-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot1/25-SPINE'],['pain_pilot_2','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot2/13-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot2/30-SPINE_T2'],['pain_pilot_4','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot4/33-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot4/32-SPINE_T2']]

SUBJECTS_LIST_montreal=[['errsm_02', '/Volumes/data_shared/montreal_criugm/errsm_02/28-SPINE_T1', '/Volumes/data_shared/montreal_criugm/errsm_02/28-SPINE_T2'], ['errsm_03', '/Volumes/data_shared/montreal_criugm/errsm_03/32-SPINE_all/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_03/38-SPINE_all_space'],['errsm_04', '/Volumes/data_shared/montreal_criugm/errsm_04/16-SPINE_memprage/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_04/18-SPINE_space'],['errsm_05', '/Volumes/data_shared/montreal_criugm/errsm_05/23-SPINE_MEMPRAGE/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_05/24-SPINE_SPACE'],['errsm_09', '/Volumes/data_shared/montreal_criugm/errsm_09/34-SPINE_MEMPRAGE2/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_09/33-SPINE_SPACE'],['errsm_10', '/Volumes/data_shared/montreal_criugm/errsm_10/13-SPINE_MEMPRAGE/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_10/20-SPINE_SPACE'], ['errsm_11', '/Volumes/data_shared/montreal_criugm/errsm_11/24-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_11/09-SPINE_T2'],['errsm_12', '/Volumes/data_shared/montreal_criugm/errsm_12/19-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_12/18-SPINE_T2'],['errsm_13', '/Volumes/data_shared/montreal_criugm/errsm_13/33-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_13/34-SPINE_T2'],['errsm_14', '/Volumes/data_shared/montreal_criugm/errsm_14/5002-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_14/5003-SPINE_T2'], ['errsm_16', '/Volumes/data_shared/montreal_criugm/errsm_16/23-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_16/39-SPINE_T2'], ['errsm_17', '/Volumes/data_shared/montreal_criugm/errsm_17/41-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_17/42-SPINE_T2'], ['errsm_18', '/Volumes/data_shared/montreal_criugm/errsm_18/36-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_18/33-SPINE_T2'],['errsm_21', '/Volumes/data_shared/montreal_criugm/errsm_21/27-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_21/30-SPINE_T2'],['errsm_22', '/Volumes/data_shared/montreal_criugm/errsm_22/29-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_22/25-SPINE_T2'],['errsm_23', '/Volumes/data_shared/montreal_criugm/errsm_23/29-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_23/28-SPINE_T2'],['errsm_24', '/Volumes/data_shared/montreal_criugm/errsm_24/20-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_24/24-SPINE_T2'],['errsm_25', '/Volumes/data_shared/montreal_criugm/errsm_25/25-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_25/26-SPINE_T2'],['errsm_26', None, '/Volumes/data_shared/montreal_criugm/errsm_26/31-SPINE_T2'],['errsm_30', '/Volumes/data_shared/montreal_criugm/errsm_30/51-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_30/50-SPINE_T2'],['errsm_31', '/Volumes/data_shared/montreal_criugm/errsm_31/31-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_31/32-SPINE_T2'],['errsm_32', '/Volumes/data_shared/montreal_criugm/errsm_32/16-SPINE_T1/echo_2.09 ', '/Volumes/data_shared/montreal_criugm/errsm_32/19-SPINE_T2'],['errsm_33', '/Volumes/data_shared/montreal_criugm/errsm_33/30-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_33/31-SPINE_T2'],['pain_pilot_1','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot1/25-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot1/25-SPINE'],['pain_pilot_2','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot2/13-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot2/30-SPINE_T2'],['pain_pilot_3','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot3/16-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot3/31-SPINE_T2'],['pain_pilot_4','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot4/33-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot4/32-SPINE_T2']]
SUBJECTS_LIST_marseille = [['1','/Volumes/data_shared/marseille/ED/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-101','/Volumes/data_shared/marseille/ED/01_0008_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-65'],['ALT','/Volumes/data_shared/marseille/ALT/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-15','/Volumes/data_shared/marseille/ALT/01_0100_space-composing'],['JD','/Volumes/data_shared/marseille/JD/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-23','/Volumes/data_shared/marseille/JD/01_0100_compo-space'],['JW','/Volumes/data_shared/marseille/JW/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5','/Volumes/data_shared/marseille/JW/01_0100_compo-space'],['MD','/Volumes/data_shared/marseille/MD_T075/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-7','/Volumes/data_shared/marseille/MD_T075/01_0100_t2-compo'],['MLL','/Volumes/data_shared/marseille/MLL_1016/01_0008_sc-mprage-1mm-2palliers-fov384-comp-sp-7','/Volumes/data_shared/marseille/MLL_1016/01_0100_t2-compo'],['MT','/Volumes/data_shared/marseille/MT/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5','/Volumes/data_shared/marseille/MT/01_0100_t2composing'],['T047','/Volumes/data_shared/marseille/T047/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5','/Volumes/data_shared/marseille/T047/01_0100_t2-3d-composing'],['VC','/Volumes/data_shared/marseille/VC/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-23','/Volumes/data_shared/marseille/VC/01_0008_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-113'],['VG','/Volumes/data_shared/marseille/VG/T1/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-15','/Volumes/data_shared/marseille/VG/T2/01_0024_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-11'],['VP','/Volumes/data_shared/marseille/VP/01_0011_sc-mprage-1mm-2palliers-fov384-comp-sp-25','/Volumes/data_shared/marseille/VP/01_0100_space-compo']]
SUBJECTS_LIST_TO_ADD = [['errsm_34','/Volumes/data_shared/montreal_criugm/errsm_34/41-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/errsm_34/40-SPINE_T2'],['errsm_35','/Volumes/data_shared/montreal_criugm/errsm_35/37-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/errsm_35/38-SPINE_T2']]

SUBJECTS_LIST_BUG = [['errsm_20', '/Volumes/data_shared/montreal_criugm/errsm_20/12-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_20/34-SPINE_T2'],['pain_pilot_3','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot3/16-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot3/31-SPINE_T2']]

SUBJECTS_LIST_STR1 = [['VC','/Volumes/data_shared/marseille/VC/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-23','/Volumes/data_shared/marseille/VC/01_0008_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-113'],['1','/Volumes/data_shared/marseille/ED/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-101','/Volumes/data_shared/marseille/ED/01_0008_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-65'],['errsm_10', '/Volumes/data_shared/montreal_criugm/errsm_10/13-SPINE_MEMPRAGE/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_10/20-SPINE_SPACE'],['pain_pilot_1','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot1/25-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot1/25-SPINE']]
SUBJECTS_LIST_STR = [['pain_pilot_4','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot4/33-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot4/32-SPINE_T2'],['errsm_11', '/Volumes/data_shared/montreal_criugm/errsm_11/24-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_11/09-SPINE_T2']]

SUBJECTS_LIST = SUBJECTS_LIST_STR
# add path to scripts
#export PATH=${PATH}:$SCT_DIR/dev/template_creation
#export PATH_OUTPUT=/Users/tamag/data/template/
#export PATH_DICOM='/Volumes/data_shared/'
do_preprocessing_T1 = 0
register = 0
apply_warp = 1

if do_preprocessing_T1:
   # Create folder to gather all labels_vertebral.nii.gz files
    if not os.path.isdir(PATH_OUTPUT + '/'+'labels_vertebral'):
        os.makedirs(PATH_OUTPUT + '/'+'labels_vertebral')

   # loop across subjects
    for i in range(0, len(SUBJECTS_LIST)):
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

       #  # Convert to RPI
       #  # Input:
       #  # - data.nii.gz
       #  # - data_RPI.nii.gz
       #  print '\nConvert to RPI'
       #  orientation = get_orientation('data.nii.gz')
       #  sct.run('sct_orientation -i data.nii.gz -s RPI')

        # Get info from txt file
        print '\nRecover infos from text file' + PATH_INFO + '/' + subject+ '/' + 'crop.txt\n'
        file_name = 'crop.txt'
        os.chdir(PATH_INFO + '/' + subject)
        file_results = open(PATH_INFO+ '/' +subject+ '/' +file_name, 'r')
        ymin_anatomic = None
        ymax_anatomic = None
        for line in file_results:
            line_list = line.split(',')
            zmin_seg = line.split(',')[0]
            zmax_seg = line.split(',')[1]
            if len(line_list)==4:
                ymin_anatomic = line.split(',')[2]
                ymax_anatomic = line.split(',')[3]
        file_results.close()

        os.chdir(PATH_OUTPUT + '/subjects/'+subject+'/'+'T1')


        # Copy T2 image in the folder
        sct.run('cp ../T2/data_RPI_crop.nii.gz data_T2_RPI.nii.gz')

        # Set T1 image in the space of T2
        # -input: data_RPI.nii.gz
        # -output: data_RPI_registered.nii.gz
        #sct.run('isct_c3d data_T2_RPI.nii.gz data_RPI.nii.gz -reslice-identity data_RPI_registered.nii.gz')

        sct.run('isct_antsRegistration -d 3 --transform Translation[0.5] --metric MI[data_T2_RPI.nii.gz,data_RPI.nii.gz,1,32] --convergence 10 -s 0 -f 1 -o [warp_,data_T1_reg.nii.gz,data_T2_reg.nii.gz]')
        # #name output: warp_0GenericAffine.mat

        sct.run('isct_antsRegistration --dimensionality 3 --transform syn[0.5,3,0] --metric MI[data_T2_RPI.nii.gz,data_T1_reg.nii.gz,1,32] --convergence 0 --shrink-factors 1 --smoothing-sigmas 0mm --restrict-deformation 1x1x0 --output [step0,data_RPI_registered.nii.gz] --interpolation BSpline[3]')
        sct.run('sct_apply_transfo -i data_RPI.nii.gz -d data_T2_RPI.nii.gz -w warp_0GenericAffine.mat,step00Warp.nii.gz -x linear -o data_RPI_registered.nii.gz')
        #sct.run('sct_apply_transfo -i data_RPI.nii.gz -d data_T2_RPI.nii.gz -w step00Warp.nii.gz -x linear -o data_RPI_registered.nii.gz')


        # Crop anatomic image along y if needed (due to artifacts of marseille data)
        # input:  data_RPI_registered.nii.gz
        # output: data_RPI_registered.nii.gz
        if ymin_anatomic !=None and ymax_anatomic != None:
            sct.run('sct_crop_image -i data_RPI_registered.nii.gz -o data_RPI_registered.nii.gz -start ' + ymin_anatomic + ' -end ' + ymax_anatomic + ' -dim 1 -b 0')


        # propseg
        # input:
        # - data_RPI_registered.nii.gz
        # - centerline_propseg_RPI.nii.gz
        # output:
        # - data_RPI_registered_seg.nii.gz

        print '\nExtracting segmentation...'
        list_dir = os.listdir(PATH_INFO + '/'+subject)
        centerline_proseg = False
        for k in range(len(list_dir)):
            if list_dir[k] == 'centerline_propseg_RPI.nii.gz':
                centerline_proseg = True
        if centerline_proseg == True:
            sct.printv('sct_propseg -i data_RPI_registered.nii.gz -t t1 -init-centerline ' + PATH_INFO + '/' + subject + '/centerline_propseg_RPI.nii.gz')
            sct.run('sct_propseg -i data_RPI_registered.nii.gz -t t1 -init-centerline ' + PATH_INFO + '/' + subject + '/centerline_propseg_RPI.nii.gz')
        else:
            sct.printv('sct_propseg -i data_RPI_registered.nii.gz -t t1')
            sct.run('sct_propseg -i data_RPI_registered.nii.gz -t t1')

        # Erase 3 top and 3 bottom slices of the segmentation to avoid edge effects  (Done because propseg tends to diverge on edges)
        print '\nErasing 3 top and 3 bottom slices of the segmentation to avoid edge effects...'
        path_seg, file_seg, ext_seg = sct.extract_fname('data_RPI_registered_seg.nii.gz')
        image_seg = nibabel.load('data_RPI_registered_seg.nii.gz')
        nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension('data_RPI_registered_seg.nii.gz')
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

        # crop segmentation along z(but keep same dimension)
        # input:
        # - data_RPI_registered_seg_mod.nii.gz
        # - crop.txt
        # output:
        # - data_RPI_registered_seg_mod_crop.nii.gz
        print '\nCrop segmentation...'
        if zmax_seg == 'max':
            nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension('data_RPI_registered_seg.nii.gz')
            sct.printv('sct_crop_image -i data_RPI_registered_seg_mod.nii.gz -o data_RPI_registered_seg_mod_crop.nii.gz -start ' + zmin_seg + ' -end ' + str(nz) + ' -dim 2 -b 0')
            os.system('sct_crop_image -i data_RPI_registered_seg_mod.nii.gz -o data_RPI_registered_seg_mod_crop.nii.gz -start ' + zmin_seg + ' -end ' + str(nz) + ' -dim 2 -b 0')
        else: sct.run('sct_crop_image -i data_RPI_registered_seg_mod.nii.gz -o data_RPI_registered_seg_mod_crop.nii.gz -start ' + zmin_seg + ' -end ' + zmax_seg + ' -dim 2 -b 0')

        os.system('pwd')
        # Extract centerline from T1 and T2
        # use segmentation + initiation centerline (+ labels_updown to complete)
        print '\nExtracting centerline from T1...'
        labels_updown = False
        list_file_info = os.listdir(PATH_INFO+'/'+subject)
        for k in range(0,len(list_file_info)):
            if list_file_info[k] == 'labels_updown.nii.gz':
                labels_updown = True
        if labels_updown:
            sct.run('sct_get_centerline_from_labels -i  data_RPI_registered_seg_mod_crop.nii.gz,'+PATH_INFO+'/'+subject+'/centerline_propseg_RPI.nii.gz,'+PATH_INFO+'/'+subject+'/labels_updown.nii.gz -o centerline_T1.nii.gz')
        else: sct.run('sct_get_centerline_from_labels -i  data_RPI_registered_seg_mod_crop.nii.gz,'+PATH_INFO+'/'+subject+'/centerline_propseg_RPI.nii.gz -o centerline_T1.nii.gz')
        print '\nExtracting centerline from T2...'
        os.system('pwd')
        sct.run('sct_get_centerline_from_labels -i ../T2/seg_and_labels.nii.gz -o centerline_T2.nii.gz')



if register:
    for i in range(0,len(SUBJECTS_LIST)):
        subject = SUBJECTS_LIST[i][0]

        from numpy import asarray
        from msct_smooth import smoothing_window

        # go to output folder
        print '\nGo to output folder '+ PATH_OUTPUT + '/subjects/'+subject+'/T1'
        os.chdir(PATH_OUTPUT + '/subjects/'+subject+'/T1')

        # Register seg to create first warping field and apply it
        print '\nUsing register_seg with centerlines to create first warping field and applying it...'
        x_disp, y_disp = register_seg('centerline_T1.nii.gz', 'centerline_T2.nii.gz')
        x_disp_a = asarray(x_disp)
        y_disp_a = asarray(y_disp)
        x_disp_smooth = smoothing_window(x_disp_a, window_len=31, window='hanning', verbose=2)
        y_disp_smooth = smoothing_window(y_disp_a, window_len=31, window='hanning', verbose=2)

        generate_warping_field('centerline_T2.nii.gz', x_disp_smooth, y_disp_smooth, fname='warping_field_seg.nii.gz')
        sct.run('sct_apply_transfo -i data_RPI_registered.nii.gz -d data_T2_RPI.nii.gz -w warping_field_seg.nii.gz -o data_RPI_registered_reg1.nii.gz -x spline')


        # Register_image to create second warping field and apply it
        print'\nUsing register_image with images to create second warping field and applying it...'
        x_disp_2, y_disp_2 = register_images('data_RPI_registered_reg1.nii.gz', 'data_T2_RPI.nii.gz')
        x_disp_2_a = asarray(x_disp_2)
        y_disp_2_a = asarray(y_disp_2)
        x_disp_2_smooth = smoothing_window(x_disp_2_a, window_len=31, window='hanning', verbose=2)
        y_disp_2_smooth = smoothing_window(y_disp_2_a, window_len=31, window='hanning', verbose=2)

        generate_warping_field('data_T2_RPI.nii.gz', x_disp_2_smooth, y_disp_2_smooth, fname='warping_field_im_trans.nii.gz')
        sct.run('sct_apply_transfo -i data_RPI_registered_reg1.nii.gz -d data_T2_RPI.nii.gz -w warping_field_im_trans.nii.gz -o data_RPI_registered_reg2.nii.gz -x spline')


if apply_warp:
    for i in range(0, 1):#len(SUBJECTS_LIST)):
        subject = SUBJECTS_LIST[i][0]

        # go to output folder
        print '\nGo to output folder '+ PATH_OUTPUT + '/subjects/'+subject+'/T1'
        os.chdir(PATH_OUTPUT + '/subjects/'+subject+'/T1')

        # Straightened image
        print '\nStraightening image...'
        sct.run('sct_apply_transfo -i data_RPI_registered_reg2.nii.gz -d ../T2/data_RPI_crop_straight_normalized.nii.gz -o data_RPI_registered_reg2_straight.nii.gz -w ../T2/warp_curve2straight.nii.gz -x spline')

        # Push into template space
        print '\nPushing into template space...'
        sct.run('sct_apply_transfo -i data_RPI_registered_reg2_straight.nii.gz -d ../T2/data_RPI_crop_straight_normalized_crop_2temp.nii.gz -o data_RPI_registered_reg2_straight_2temp.nii.gz -w ../T2/native2temp.txt -x spline')

        # # Straightening and pushing into template space
        # sct.run('sct_apply_transfo -i data_RPI_registered_reg2.nii.gz -d ../T2/data_RPI_crop_straight_normalized_crop_2temp.nii.gz -o data_RPI_registered_reg2_straight_2temp.nii.gz -w ../T2/warp_curve2straight.nii.gz,../T2/native2temp.txt -x spline')

        # Align vertebrae
        print '\nAligning vertebrae...'
        list_dir_3 = os.listdir(PATH_OUTPUT + '/subjects/' + subject + '/T2')
        list_tmp_folder = [file for file in list_dir_3 if file.startswith('tmp')]
        last_tmp_folder_name = list_tmp_folder[-1]
        sct.run('sct_apply_transfo -i data_RPI_registered_reg2_straight_2temp.nii.gz -o '+subject+'_T1_aligned.nii.gz -d '+PATH_OUTPUT +'/labels_vertebral/template_landmarks.nii.gz -w ../T2/'+last_tmp_os.path.join(folder_name, '/warp_subject2template.nii.gz) -x spline')

        #Normalize intensity of result
        print'\nNormalizing intensity of results...'
        sct.run('sct_normalize.py -i '+subject+'_T1_aligned.nii.gz')

        #Warning that results for the subject is ready
        print'\nThe results for subject '+subject+' are ready. You can visualize them by tapping: fslview '+subject+'_T1_aligned_normalized.nii.gz'

        #Copy final results into final results
        if not os.path.isdir(PATH_OUTPUT +'/Final_results'):
            os.makedirs(PATH_OUTPUT +'/Final_results')
        sct.run('cp '+subject+'_T1_aligned.nii.gz ' +PATH_OUTPUT +'/Final_results/'+subject+'_T1_aligned.nii.gz')
        sct.run('cp '+subject+'_T1_aligned_normalized.nii.gz ' +PATH_OUTPUT +'/Final_results/'+subject+'_T1_aligned_normalized.nii.gz')


