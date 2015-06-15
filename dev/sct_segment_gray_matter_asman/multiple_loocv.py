#!/usr/bin/env python
import sct_utils as sct
import numpy as np
import sys
from msct_parser import Parser
from msct_gmseg_utils import leave_one_out_by_subject
import os
import time


def loocv(param):
    use_level, weight, subjects_list = param

    name_dir = './' + time.strftime("%y%m%d%H%M%S") + '_' + str(len(subjects_list)) + 'subjects_levels_' + str(use_level) + '_weight' + str(weight)
    sct.run('mkdir ' + name_dir)
    sct.run('mkdir ' + name_dir + '/dictionary')
    for subject_dir in subjects_list:
        sct.run('cp -r ' + path_dictionary + '/' + subject_dir + ' ' + name_dir + '/dictionary/')
    os.chdir(name_dir)
    leave_one_out_by_subject('dictionary/', use_levels=use_level, weight=weight)
    os.chdir('..')

if __name__ == '__main__':
    parser = Parser(__file__)
    parser.usage.set_description('Gray matter segmentation with several types of registration')
    parser.add_option(name="-dic",
                      type_value="folder",
                      description="Path to the dictionary of images",
                      mandatory=True,
                      example='/home/jdoe/data/dictionary')

    arguments = parser.parse(sys.argv[1:])
    path_dictionary = arguments["-dic"]

    gamma = 1.2
    all_subjects = np.asarray(os.listdir(path_dictionary)[1:])
    n_subjects = [2, 5, 10, 15, 20, 28, 36]
    n_repetition = 8

    for n in n_subjects:
        ir = 0
        while ir < n_repetition:
            subjects_to_use = np.random.choice(all_subjects, n, replace=False)
            loocv((False, 0, subjects_to_use))
            loocv((True, gamma, subjects_to_use))
            ir +=1
