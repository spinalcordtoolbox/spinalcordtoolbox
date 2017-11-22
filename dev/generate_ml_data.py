#!/usr/bin/env python

import sct_utils as sct
import os

#path = '/Users/benjamindeleener/data/data_testing/C2-T2/'
contrast = 't2'
#path_output_seg = '/Users/benjamindeleener/data/spinal_cord_segmentation_data/training/labels/'
#path_output_im = '/Users/benjamindeleener/data/spinal_cord_segmentation_data/training/data/'
path = '/Users/benjamindeleener/data/data_testing/test/'
path_output_seg = '/Users/benjamindeleener/data/spinal_cord_segmentation_data/test/labels/'
path_output_im = '/Users/benjamindeleener/data/spinal_cord_segmentation_data/test/data/'
size = '80'

def generate_data_list(folder_dataset, verbose=1):
    """
    Construction of the data list from the data set
    This function return a list of directory (in folder_dataset) in which the contrast is present.
    :return data:
    """
    data_subjects, subjects_dir = [], []

    # each directory in folder_dataset should be a directory of a subject
    for subject_dir in os.listdir(folder_dataset):
        if subject_dir.startswith("."):
            continue
        if not os.path.isdir(os.path.join(folder_dataset, subject_dir)):
            continue

        data_subjects.append(os.path.join(folder_dataset, subject_dir))
        subjects_dir.append(subject_dir)

    if not data_subjects:
        sct.printv('ERROR: No subject data were found in ' + folder_dataset + '. '
                   'Please organize your data correctly or provide a correct dataset.',
                   verbose=verbose, type='error')

    return data_subjects, subjects_dir

data_subjects, subjects_name = generate_data_list(path)
current_folder = os.getcwd()

for subject_folder in data_subjects:
    print subject_folder
    os.chdir(subject_folder+contrast)
    sct.run('sct_seg_utility.py -i ' + contrast + '.nii.gz'
                                      ' -seg ' + contrast + '_manual_segmentation.nii.gz'
                                      ' -ofolder-im ' + path_output_im +
                                      ' -ofolder-seg ' + path_output_seg +
                                      ' -size ' + size + ' -v 2', verbose=2)
    os.chdir(current_folder)
