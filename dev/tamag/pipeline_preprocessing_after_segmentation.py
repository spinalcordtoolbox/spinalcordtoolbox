#!/usr/bin/env python


import commands, sys, os


# Get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# Append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')
sys.path.append(path_sct + '/dev/tamag')

import sct_utils as sct
from glob import glob
from shutil import copy
import fnmatch

os.chdir('/Users/tamag/data/data_for_template/montreal')
list_dir = os.listdir('/Users/tamag/data/data_for_template/montreal')

for i in range(2, len(list_dir)):
    list_dir_2 = os.listdir('/Users/tamag/data/data_for_template/montreal'+'/'+list_dir[i])
    for j in range(len(list_dir_2)):
        if list_dir_2[j] == 'T2':
            # Going into last tmp folder
            list_dir_3 = os.listdir('/Users/tamag/data/data_for_template/montreal'+'/'+list_dir[i]+'/'+list_dir_2[j])
            list_tmp_folder = [file for file in list_dir_3 if file.startswith('tmp')]
            last_tmp_folder_name = list_tmp_folder[-1]
            os.chdir(list_dir[i]+ '/' + list_dir_2[j]+'/'+last_tmp_folder_name)

            # Add label files and preprocess data for template registration
            print '\nPreprocessing data from: '+ list_dir[i]+ '/' + list_dir_2[j] + ' ...'
            name_seg_mod = 't2_crop_seg_mod_crop.nii.gz'
            sct.printv('sct_function_preprocessing.py -i *_t2_crop.nii.gz -l ' + name_seg_mod + ',up.nii.gz,down.nii.gz')
            os.system('sct_function_preprocessing.py -i *_t2_crop.nii.gz -l ' + name_seg_mod + ',up.nii.gz,down.nii.gz')
            name_output_straight = glob('*t2_crop_straight.nii.gz')[0]
            name_output_straight_normalized = glob('*t2_crop_straight_normalized.nii.gz')[0]

            # Copy resulting files into Results folder
            print '\nCopy output files into:/Users/tamag/data/data_for_template/Results_preprocess/T2'
            copy(name_output_straight, '/Users/tamag/data/data_for_template/Results_preprocess/T2/t2_' + list_dir[i]+'_crop_straight.nii.gz')
            copy(name_output_straight_normalized, '/Users/tamag/data/data_for_template/Results_preprocess/T2/t2_' + list_dir[i]+'_crop_straight_normalized.nii.gz')

            # Copy centerline and warping files to T1 folder
            copy('generated_centerline.nii.gz', '../../T1')
            copy('warp_curve2straight.nii.gz', '../../T1')
            copy('warp_straight2curve.nii.gz', '../../T1')

            # Remove temporary file
            print('\nRemove temporary files...')
            #sct.run('rm -rf '+path_tmp)

            os.chdir('../../..')
