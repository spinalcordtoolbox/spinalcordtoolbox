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
# ~/subject/t2/labels_propseg.nii.gz --> a series of binary labels along the cord to help propseg
# ~/subject/t2/crop.txt --> ASCII txt file that indicates zmin and zmax for cropping the anatomic image and the segmentation . Format: zmin_anatomic zmax_anatomic zmin_seg zmax_seg
# ~/subject/t2/labels_updown.nii.gz --> a series of binary labels to complete the centerline from brainstem to L2/L3.
# ~/subject/t2/labels_vertebral.nii.gz --> a series of labels to identify vertebral level. These are placed in the middle of the cord and vertebral body. The value of the label corresponds to the level. I.e., Brainstem (PMJ)=1, C1=2, C2=3, etc.
# cf snapshot in $SCT_DIR/dev/template_preprocessing/snap1, 2, etc.

import os, sys, commands

# Get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# Append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')

import sct_utils as sct
import nibabel
from sct_orientation import get_orientation, set_orientation
from scipy import ndimage
from numpy import array

# add path to scripts
PATH_DICOM= '/Volumes/data_shared/'
PATH_OUTPUT= '/Users/tamag/data/data_template/subject_test'
PATH_INFO = '/Users/tamag/code/spinalcordtoolbox/data'  # to be replaced by URL from github

# define subject
# SUBJECTS_LIST=[['errsm_14', ,'pathtodicomt1', 'pathtodicomt2']
SUBJECTS_LIST=[['errsm_14', '/Volumes/data_shared/montreal_criugm/errsm_14/5002-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_14/5003-SPINE_T2'], ['errsm_16', '/Volumes/data_shared/montreal_criugm/errsm_16/23-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_16/39-SPINE_T2'], ['errsm_17', '/Volumes/data_shared/montreal_criugm/errsm_17/41-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_17/42-SPINE_T2'], ['errsm_18', '/Volumes/data_shared/montreal_criugm/errsm_18/36-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_18/33-SPINE_T2']]

# add path to scripts
#export PATH=${PATH}:$SCT_DIR/dev/template_creation
#export PATH_OUTPUT=/Users/tamag/data/template/
#export PATH_DICOM='/Volumes/data_shared/'
do_preprocessing_T2 = 0
normalize_levels_T2 = 1


if normalize_levels_T2:
    for i in range(2,len(SUBJECTS_LIST)):
        subject = SUBJECTS_LIST[i][0]

        # go to output folder
        print '\nGo to output folder '+ PATH_OUTPUT + '/'+subject+'/'+'T2'
        os.chdir(PATH_OUTPUT + '/'+subject+'/'+'T2')

        ## Create a cross after recropping (cross in landmark_native.nii.gz)
        # Detect extrema: (same code as sct_detect_extrema except for the detection of the center of mass)
        file = nibabel.load('centerline_RPI_reg.nii.gz')
        data_c = file.get_data()
        hdr = file.get_header()

        #Get center of mass of the centerline
        nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension('centerline_RPI_reg.nii.gz')
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
        os.system('sct_crop_image -i data_RPI_crop_denoised_straight.nii.gz -o data_RPI_crop_denoised_straight_crop.nii.gz -dim 2 -start '+ str(z_min)+' -end '+ str(z_max))
        os.system('sct_create_cross.py -i data_RPI_crop_denoised_straight_crop.nii.gz -x ' +str(x_min)+' -y '+str(y_min))
        # Listing the different levels into (labels_vertebral.nii.gz)
        # fslmaths *_t2_crop.nii.gz -mul 0 level_marker.nii.gz
        # sct_label_utils -i level_marker.nii.gz -t create -x 19,109,27,1.0:19,103,59,1.0:19,98,92,1.0:19,94,120,1.0:19,90,148,1.0:19,88,172,1.0:19,85,197,1.0:19,85,220,1.0:19,87,242,1.0:19,92,265,1.0:19,98,286,1.0:19,105,305,1.0:19,113,322,1.0:19,123,340,1.0:19,128,358,1.0:19,131,377,1.0:19,131,395,1.0:19,131,411,1.0:19,133,430,1.0:19,132,449,1.0:19,130,458,1.0:19,117,498,1.0
        # sct_label_utils -i level_marker.nii.gz  -o level_marker.nii.gz -t increment

        # Push into template
        os.system('sct_push_into_template_space.py -i data_RPI_crop_denoised_straight_crop.nii.gz -n landmark_native.nii.gz')
