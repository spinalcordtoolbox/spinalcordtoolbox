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
do_preprocessing_T2 = 1
normalize_levels_T2 = 0

if do_preprocessing_T2:

    # loop across subjects
    for i in range(len(SUBJECTS_LIST)):
        subject = SUBJECTS_LIST[i][0]

        # create and go to output folder
        #os.makedirs(PATH_OUTPUT + '/'+subject)
        os.chdir(PATH_OUTPUT + '/'+subject+'/'+'T2')

        # convert to nii
        sct.run('dcm2nii -o . -r N ' + SUBJECTS_LIST[i][2] + '/*.dcm')

        # change file name
        sct.run('mv *.nii.gz data.nii.gz')

        # Get info from txt file
        file_name = 'crop.txt'
        os.chdir(PATH_INFO + '/' + subject)
        file_results = open(file_name, 'r')
        for line in file_results:
            zmin_anatomic = line.split(',')[0]
            zmax_anatomic = line.split(',')[1]
            zmin_seg = line.split(',')[2]
            zmax_seg = line.split(',')[3]
        file_results.close()
        # Crop image
        sct.run('sct_crop_image -i data.nii.gz -o data_crop.nii.gz -dim 2 -start ' + zmin_anatomic + ' -end ' + zmax_anatomic )

        # denoising
        # input:
        # - data_crop.nii.gz
        # output:
        # - data_crop_denoised.nii.gz
        sct.run('sct_denoising_nlm.py -i data_crop.nii.gz')

        # propseg
        # input:
        # - data__crop_denoised.nii.gz
        # - labels_propseg.nii.gz
        # output:
        # - data_crop_denoised_seg.nii.gz
        sct.printv('sct_propseg -i data_crop_denoised.nii.gz -t t2 -init-centerline centerline_propseg.nii.gz')
        os.system('sct_propseg -i data_crop_denoised.nii.gz -t t2 -init-centerline centerline_propseg.nii.gz')

        # Erase 3 top and 3 bottom slices of the segmentation to avoid edge effects
        path_seg, file_seg, ext_seg = sct.extract_fname('data_crop_denoised_seg.nii.gz')
        image_seg = nibabel.load('data_crop_denoised_seg.nii.gz')
        nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension('data_crop_denoised_seg.nii.gz')
        data_seg = image_seg.get_data()
        hdr_seg = image_seg.get_header()
           # List slices that contain non zero values
        z_centerline = [iz for iz in range(0, nz, 1) if data_seg[:,:,iz].any() ]

        for k in range(0,3):
            data_seg[:,:,z_centerline[-1]-k] = 0
            if z_centerline[0]+k < nz:
                data_seg[:,:,z_centerline[0]+k] = 0
        img_seg = nibabel.Nifti1Image(data_seg, None, hdr_seg)
        #nibabel.save(img_seg, file_seg + '_mod' + ext_seg)
        nibabel.save(img_seg, file_seg + ext_seg)

        # crop segmentation (but keep same dimension)
        # input:
        # - data_crop_denoised_seg.nii.gz
        # - crop.txt
        # output:
        # - data_crop_denoised_seg_crop.nii.gz
        if zmax_seg == 'max':
            nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension('data_crop_denoised_seg.nii.gz')
            sct.run('sct_crop_image -i data_crop_denoised_seg.nii.gz -o data_crop_denoised_seg_crop.nii.gz -start ' + zmin_seg + ' -end ' + nz + ' -dim 2 -b 0')
        else: sct.run('sct_crop_image -i data_crop_denoised_seg.nii.gz -o data_crop_denoised_seg_crop.nii.gz -start ' + zmin_seg + ' -end ' + zmax_seg + ' -dim 2 -b 0')

        # extract centerline
        # function: sct_get_centerline_from_labels
        # input:
        # - data_crop_denoised_seg_crop.nii.gz
        # - labels_updown.nii.gz
        # output:
        # - centeline.nii.gz
        sct.run('sct_get_centerline_from_labels -i data_crop_denoised_seg_crop.nii.gz,up.nii.gz,down.nii.gz -o centerline.nii.gz')

        # straighten image using centerline
        # function: sct_straighten_spinalcord (option: hanning)
        # input:
        # - data_crop_denoised.nii.gz
        # - centerline.nii.gz
        # output:
        # - warp_curve2straight.nii.gz
        # - data_crop_denoised_straight.nii.gz
        sct.run('sct_straighten_spinalcord -i data_crop_denoised.nii.gz -c centerline.nii.gz')

        # apply straightening to centerline and to labels_vertebral.nii.gz
        # function: sct_apply_transfo
        # input:
        # - centerline.nii.gz + labels_vertebral.nii.gz
        # - warp_curve2straight.nii.gz
        # output:
        # - centerline_straight.nii.gz
        sct.run('sct_straighten_spinalcord -i centerline.nii.gz -c centerline.nii.gz')

        # normalize intensity
        print '\nNormalizing intensity of the straightened image...'
        sct.run('sct_normalize.py -i data_crop_denoised_straight.nii.gz -c centerline_straight.nii.gz')




#if normalize_levels: