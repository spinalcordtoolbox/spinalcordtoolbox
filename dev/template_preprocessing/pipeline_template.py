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
do_preprocessing_T2 = 1
normalize_levels_T2 = 0

if do_preprocessing_T2:

    # loop across subjects
    for i in range(2,len(SUBJECTS_LIST)):
        subject = SUBJECTS_LIST[i][0]

        # create and go to output folder
        print '\nCreate and go to output folder '+ PATH_OUTPUT + '/'+subject+'/'+'T2'
        os.makedirs(PATH_OUTPUT + '/'+subject)
        os.makedirs(PATH_OUTPUT + '/'+subject+'/'+'T2')
        os.chdir(PATH_OUTPUT + '/'+subject+'/'+'T2')

        # convert to nii
        print '\nConvert to nii'
        sct.run('dcm2nii -o . -r N ' + SUBJECTS_LIST[i][2] + '/*.dcm')

        # change file name
        print '\nChange file name to data.nii.gz...'
        sct.run('mv *.nii.gz data.nii.gz')

        # Get info from txt file
        print '\nRecover infos from text file' + PATH_INFO + '/' + subject+ '/' + 'crop.txt'
        file_name = 'crop.txt'
        os.chdir(PATH_INFO + '/' + subject)
        file_results = open(PATH_INFO+ '/' +subject+ '/' +file_name, 'r')
        for line in file_results:
            zmin_anatomic = line.split(',')[0]
            zmax_anatomic = line.split(',')[1]
            zmin_seg = line.split(',')[2]
            zmax_seg = line.split(',')[3]
        file_results.close()

        os.chdir(PATH_OUTPUT + '/'+subject+'/'+'T2')

        # Convert to RPI
        # Input:
        # - data.nii.gz
        # - data_RPI.nii.gz
        print '\nConvert to RPI'
        orientation = get_orientation('data.nii.gz')
        sct.run('sct_orientation -i data.nii.gz -s RPI')
        # Crop image
        sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start ' + zmin_anatomic + ' -end ' + zmax_anatomic )
        # sct.run('sct_orientation -i data_RPI_crop.nii.gz -o data_crop.nii.gz -s '+ orientation)

        # denoising
        # input:
        # - data_crop.nii.gz
        # output:
        # - data_crop_denoised.nii.gz
        print '\nDenoising image data_RPI_crop.nii.gz...'
        sct.printv('sct_denoising_nlm.py -i data_RPI_crop.nii.gz')
        os.system('sct_denoising_nlm.py -i data_RPI_crop.nii.gz')

        # propseg
        # input:
        # - data__crop_denoised.nii.gz
        # - labels_propseg.nii.gz
        # output:
        # - data_crop_denoised_seg.nii.gz
        print '\nExtracting segmentation...'
        sct.printv('sct_propseg -i data_RPI_crop_denoised.nii.gz -t t2 -init-centerline ' + PATH_INFO + '/' + subject + '/centerline_propseg_RPI.nii.gz')
        os.system('sct_propseg -i data_RPI_crop_denoised.nii.gz -t t2 -init-centerline ' + PATH_INFO + '/' + subject + '/centerline_propseg_RPI.nii.gz')

        # Erase 3 top and 3 bottom slices of the segmentation to avoid edge effects
        print '\nErasing 3 top and 3 bottom slices of the segmentation to avoid edge effects...'
        path_seg, file_seg, ext_seg = sct.extract_fname('data_RPI_crop_denoised_seg.nii.gz')
        image_seg = nibabel.load('data_RPI_crop_denoised_seg.nii.gz')
        nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension('data_RPI_crop_denoised_seg.nii.gz')
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
        #nibabel.save(img_seg, file_seg + ext_seg)

        # crop segmentation (but keep same dimension)
        # input:
        # - data_crop_denoised_seg.nii.gz
        # - crop.txt
        # output:
        # - data_crop_denoised_seg_crop.nii.gz
        print '\nCrop segmentation...'
        if zmax_seg == 'max':
            nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension('data_RPI_crop_denoised_seg.nii.gz')
            sct.printv('sct_crop_image -i data_RPI_crop_denoised_seg_mod.nii.gz -o data_RPI_crop_denoised_seg_mod_crop.nii.gz -start ' + zmin_seg + ' -end ' + str(nz) + ' -dim 2 -b 0')
            os.system('sct_crop_image -i data_RPI_crop_denoised_seg_mod.nii.gz -o data_RPI_crop_denoised_seg_mod_crop.nii.gz -start ' + zmin_seg + ' -end ' + str(nz) + ' -dim 2 -b 0')
        else: sct.run('sct_crop_image -i data_RPI_crop_denoised_seg_mod.nii.gz -o data_RPI_crop_denoised_seg_mod_crop.nii.gz -start ' + zmin_seg + ' -end ' + zmax_seg + ' -dim 2 -b 0')

        # extract centerline
        # function: sct_get_centerline_from_labels
        # input:
        # - data_crop_denoised_seg_crop.nii.gz
        # - labels_updown.nii.gz
        # output:
        # - centeline.nii.gz
        print '\nExtracting centerline...'
        sct.printv('sct_get_centerline_from_labels -i data_RPI_crop_denoised_seg_mod_crop.nii.gz,'+ PATH_INFO + '/' + subject + '/up.nii.gz,'+ PATH_INFO + '/' + subject + '/down.nii.gz -o centerline_RPI.nii.gz')
        os.system('sct_get_centerline_from_labels -i data_RPI_crop_denoised_seg_mod_crop.nii.gz,'+ PATH_INFO + '/' + subject + '/up.nii.gz,'+ PATH_INFO + '/' + subject + '/down.nii.gz -o centerline_RPI.nii.gz')

        # straighten image using centerline
        # function: sct_straighten_spinalcord (option: hanning)
        # input:
        # - data_crop_denoised.nii.gz
        # - centerline.nii.gz
        # output:
        # - warp_curve2straight.nii.gz
        # - data_crop_denoised_straight.nii.gz
        print '\nStraighten image using centerline'
        sct.printv('sct_straighten_spinalcord -i data_RPI_crop_denoised.nii.gz -c centerline_RPI.nii.gz')
        os.system('sct_straighten_spinalcord -i data_RPI_crop_denoised.nii.gz -c centerline_RPI.nii.gz')

        # apply straightening to centerline and to labels_vertebral.nii.gz
        # function: sct_apply_transfo
        # input:
        # - centerline.nii.gz + labels_vertebral.nii.gz
        # - warp_curve2straight.nii.gz
        # output:
        # - centerline_straight.nii.gz
        print '\nApply straightening to centerline and to labels_vertebral.nii.gz'
        # sct.printv('sct_straighten_spinalcord -i '+ PATH_INFO + '/' + subject+ '/labels_vertebral.nii.gz -c centerline_RPI.nii.gz')
        # os.system('sct_straighten_spinalcord -i '+ PATH_INFO + '/' + subject+ '/labels_vertebral.nii.gz -c centerline_RPI.nii.gz')
        sct.printv('sct_apply_transfo -i ' + PATH_INFO + '/' + subject+ '/labels_vertebral.nii.gz -d data_RPI_crop_denoised_straight.nii.gz -w warp_curve2straight.nii.gz -x linear')
        os.system('sct_apply_transfo -i ' + PATH_INFO + '/' + subject+ '/labels_vertebral.nii.gz -d data_RPI_crop_denoised_straight.nii.gz -w warp_curve2straight.nii.gz -x linear')

        # sct.printv('sct_straighten_spinalcord -i centerline_RPI.nii.gz -c centerline_RPI.nii.gz')
        # os.system('sct_straighten_spinalcord -i centerline_RPI.nii.gz -c centerline_RPI.nii.gz')
        sct.printv('sct_apply_transfo -i centerline_RPI.nii.gz -d data_RPI_crop_denoised_straight.nii.gz -w warp_curve2straight.nii.gz -x linear')
        os.system('sct_apply_transfo -i centerline_RPI.nii.gz -d data_RPI_crop_denoised_straight.nii.gz -w warp_curve2straight.nii.gz -x linear')

        # normalize intensity
        print '\nNormalizing intensity of the straightened image...'
        sct.printv('sct_normalize.py -i data_RPI_crop_denoised_straight.nii.gz -c centerline_RPI_reg.nii.gz')
        os.system('sct_normalize.py -i data_RPI_crop_denoised_straight.nii.gz -c centerline_RPI_reg.nii.gz')


# Calculate mean labels
#os.system('sct_average_levels.py   ')

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

        # Crop image one last time
        os.system('sct_crop_image -i data_RPI_crop_denoised_straight.nii.gz -o data_RPI_crop_denoised_straight_crop.nii.gz -dim 2 -start '+ str(z_min)+' -end '+ str(z_max))
        os.system('sct_create_cross.py -i data_RPI_crop_denoised_straight_crop.nii.gz -x ' +str(x_min)+' -y '+str(y_min))


        # Push into template
        os.system('sct_push_into_template_space.py -i data_RPI_crop_denoised_straight_crop.nii.gz -n landmark_native.nii.gz')
