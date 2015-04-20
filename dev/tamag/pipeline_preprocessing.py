#!/usr/bin/env python


import commands, sys, os


# Get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# Append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')
sys.path.append(path_sct + '/dev/tamag')

import sct_utils as sct

os.chdir('/Users/tamag/data/original_data/C1-T3')

dirpath, dirnames, filenames = os.walk('/Users/tamag/data/original_data/C1-T3/errsm_03')
list_dir = os.listdir('/Users/tamag/data/original_data/C1-T3')

for i in range(1, len(list_dir)):
    list_dir_2 = os.listdir('/Users/tamag/data/original_data/C1-T3'+'/'+list_dir[i])
    for j in range(len(list_dir_2)):
        if list_dir_2[j] == 't2':
            print '\nPreprocessing data from: '+ list_dir[i]+ '/' + list_dir_2[j] + ' ...'
            os.chdir(list_dir[i]+ '/' + list_dir_2[j])
            sct.run('sct_function_preprocessing.py -i data.nii.gz -l ' + list_dir[i]+'_t2_manual_segmentation.nii.gz')
            os.chdir('../..')


