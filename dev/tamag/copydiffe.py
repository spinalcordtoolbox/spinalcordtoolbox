#!/usr/bin/env python


import commands, sys, os
from glob import glob
import nibabel

# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, "scripts"))
sys.path.append(path_sct + '/dev/tamag')

import sct_utils as sct

path = '/Users/tamag/data/data_template/info/template_subjects'
os.chdir(path)



path_2 = '/Volumes/Usagers/Etudiants/tamag/data/data_template/subject_specific_files/T1'

list_dir_2 = os.listdir(path_2) # subjects


os.chdir(path_2)

for i in range(len(list_dir_2)):
    if os.path.isdir(path_2 +'/'+list_dir_2[i]):
        list_dir_3 = os.listdir(path_2 +'/'+list_dir_2[i])
        if os.path.isdir(path_2 +'/'+list_dir_2[i]):
            os.chdir(path_2 +'/'+list_dir_2[i])
            for file in list_dir_3:
                if file not in ['centerline_propseg_RPI.nii.gz', 'crop.txt', 'labels_vertebral_value_1.nii.gz', 'labels_vertebral.nii.gz', 'labels_updown.nii.gz']:
                    os.remove(file)
            os.chdir('..')