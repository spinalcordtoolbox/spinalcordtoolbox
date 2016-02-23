#!/usr/bin/env python
import sct_utils as sct
import numpy as np
import sys
from msct_parser import Parser
from msct_gmseg_utils import leave_one_out_by_subject
import os
import multiprocessing as mp

import time


def loocv(name_dir):
    words = name_dir.split('_')
    weight = float(words[-1])
    use_level = bool(words[-3])

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
    parser.add_option(name="-list",
                      type_value=[[','], 'int'],
                      description="Path to the dictionary of images",
                      mandatory=False,
                      default_value=[2,5,10,15,20,28,36],
                      example='/home/jdoe/data/dictionary')



    arguments = parser.parse(sys.argv[1:])
    path_dictionary = arguments["-dic"]
    if "-list" in arguments:
        n_subjects = arguments["-list"]
    else:
        n_subjects = [2, 5, 10, 15, 20, 28, 36]


    param = [(True, 1.2), (False, 0)]
    all_subjects = np.asarray(os.listdir(path_dictionary)[1:])
    n_repetition = 8
    list_dir = []

    for n in n_subjects:
        ir = 0
        while ir < n_repetition:
            subjects_to_use = np.random.choice(all_subjects, n, replace=False)
            for p in param:
                name_dir = str(n) + 'subjects_' + str(ir) + '_levels_' + str(p[0]) + '_weight_' + str(p[1])
                sct.run('mkdir ' + name_dir)
                sct.run('mkdir ' + name_dir + '/dictionary')
                for subject_dir in subjects_to_use:
                    sct.run('cp -r ' + path_dictionary + '/' + subject_dir + ' ' + name_dir + '/dictionary/')
                list_dir.append(name_dir)
            ir += 1

    pool = mp.Pool()
    pool.map(loocv, list_dir)
    pool.join()