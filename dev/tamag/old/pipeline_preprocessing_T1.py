#!/usr/bin/env python


import commands, sys, os


# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, "scripts"))
sys.path.append(path_sct + '/dev/tamag')

import sct_utils as sct
from glob import glob
import time
from shutil import copy


os.chdir('/Users/tamag/data/data_for_template/montreal')
list_dir = os.listdir('/Users/tamag/data/data_for_template/montreal')

for i in range(2, len(list_dir)):
    list_dir_2 = os.listdir('/Users/tamag/data/data_for_template/montreal'+'/'+list_dir[i])
    for j in range(len(list_dir_2)):
        if list_dir_2[j] == 'T1':
            print '\nCompute data from: '+ list_dir[i]+ '/' + list_dir_2[j] + ' ...'
            os.chdir(list_dir[i]+ '/' + list_dir_2[j])

            path_tmp = sct.tmp_create(basename="preprocessing_T1")

            # copy files into tmp folder
            sct.printv('\nCopy files into tmp folder...')
            name_anatomy_file = glob('*t1_crop.nii.gz')[0]
            path_anatomy_file = os.path.abspath(name_anatomy_file)
            path_centerline = os.path.abspath('generated_centerline.nii.gz')
            path_warp1 = os.path.abspath('warp_curve2straight.nii.gz')
            path_warp2 =os.path.abspath('warp_straight2curve.nii.gz')

            sct.run('cp '+path_anatomy_file+' '+path_tmp)
            sct.run('cp '+path_centerline+' '+path_tmp)
            sct.run('cp '+path_warp1+' '+path_tmp)
            sct.run('cp '+path_warp2+' '+path_tmp)

            # Go to temp folder
            os.chdir(path_tmp)

            # Preprocess data for template registration: straightening and intensity normalization of anatomic image (using generated_centerline from T2)
            print '\nPreprocessing data from: '+ list_dir[i]+ '/' + list_dir_2[j] + ' ...'

            # Denoise image


            # Straighten the image using the fitted centerline
            print '\nStraightening the image ' + name_anatomy_file + ' using the fitted centerline ' + 'generated_centerline.nii.gz'+ ' ...'
            sct.run('sct_straighten_spinalcord -i ' + name_anatomy_file + ' -c ' + 'generated_centerline.nii.gz')
            path, file, ext = sct.extract_fname(name_anatomy_file)
            output_straighten_name = file + '_straight' +ext

            # Aplly transfo to the centerline
            print '\nApplying transformation to the centerline...'
            sct.run('sct_apply_transfo -i ' + 'generated_centerline.nii.gz' + ' -d ' + output_straighten_name + ' -w ' + 'warp_curve2straight.nii.gz' + ' -x ' + 'linear')

            # Normalize intensity of the image using the straightened centerline
            print '\nNormalizing intensity of the straightened image...'
            sct.run('sct_normalize.py -i ' + output_straighten_name + ' -c generated_centerline_reg.nii.gz')

            name_output_straight = glob('*t1_crop_straight.nii.gz')[0]
            name_output_straight_normalized = glob('*t1_crop_straight_normalized.nii.gz')[0]

            # Copy resulting files into Results folder
            print '\nCopy output files into:/Users/tamag/data/data_for_template/Results_preprocess/T1'
            copy(name_output_straight, '/Users/tamag/data/data_for_template/Results_preprocess/T1/t1_' + list_dir[i]+'_crop_straight.nii.gz')
            copy(name_output_straight_normalized, '/Users/tamag/data/data_for_template/Results_preprocess/T1/t1_' + list_dir[i]+'_crop_straight_normalized.nii.gz')

            # Remove temporary file
            print('\nRemove temporary files...')
            #sct.run('rm -rf '+path_tmp)

            os.chdir('../../..')
