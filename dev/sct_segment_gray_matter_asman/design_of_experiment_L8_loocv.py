#!/usr/bin/env python
import sct_utils as sct
import numpy as np
import sys
from msct_parser import Parser
from msct_gmseg_utils import leave_one_out_by_subject
import os
import multiprocessing as mp

import time


def prep(to_copy_3d_dic, to_copy_dic_by_slice, to_copy_dic_by_slice_denoised, L=8):
    list_dir = []
    for i in range(L):
        dir_i = 'exp' + str(i)
        sct.run('mkdir ' + dir_i)
        sct.run('cp -rf ' + to_copy_3d_dic + ' ./' + dir_i + '/' + to_copy_3d_dic)
        if i < 4:
            sct.run('cp -rf ' + to_copy_dic_by_slice_denoised + ' ./' + dir_i + '/' + to_copy_dic_by_slice_denoised)
        else:
            sct.run('cp -rf ' + to_copy_dic_by_slice + ' ./' + dir_i + '/' + to_copy_dic_by_slice)
        list_dir.append(dir_i)
    return list_dir


def do_loocv(directory):
    original_path = os.path.abspath('.')
    os.chdir(directory)
    dic_3d = None
    dic_by_slice = None
    for dir_name in os.listdir('.'):
        if os.path.isdir(dir_name):
            if '3d' in dir_name.lower():
                dic_3d = dir_name
            if 'by_slice' in dir_name.lower():
                dic_by_slice = dir_name
    if dic_3d is None or dic_by_slice is None:
        sct.printv('WARNING: dictionaries not in the loocv folder ...', 1, 'warning')
    else:
        denoising = factors_levels['denoising'][exp_plan[directory][factors['denoising']]]
        reg = factors_levels['reg'][exp_plan[directory][factors['reg']]]
        metric = factors_levels['metric'][exp_plan[directory][factors['metric']]]
        gamma = factors_levels['gamma'][exp_plan[directory][factors['gamma']]]
        eq = factors_levels['eq'][exp_plan[directory][factors['eq']]]
        mode_weight = factors_levels['mode_weight'][exp_plan[directory][factors['mode_weight']]]
        w_label_fus = factors_levels['weighted_label_fusion'][exp_plan[directory][factors['weighted_label_fusion']]]

        leave_one_out_by_subject(dic_by_slice, dic_3d, denoising=denoising, reg=reg, metric=metric, use_levels=bool(gamma), weight=gamma, eq=eq, mode_weighted_sim=mode_weight, weighted_label_fusion=w_label_fus)

    os.chdir(original_path)

if __name__ == '__main__':
    parser = Parser(__file__)
    parser.usage.set_description('Gray matter segmentation with several types of registration')
    parser.add_option(name="-3D-dic",
                      type_value="folder",
                      description="Path to the dictionary of 3D images",
                      mandatory=True,
                      example='/home/jdoe/data/dictionary')
    parser.add_option(name="-dic-by-slice",
                      type_value="folder",
                      description="Path to the dictionary slice by slice images to compute the model",
                      mandatory=True,
                      example='/home/jdoe/data/dictionary')
    parser.add_option(name="-dic-by-slice-denoised",
                      type_value="folder",
                      description="Path to the dictionary slice by slice denoised images to compute the model ",
                      mandatory=True,
                      example='/home/jdoe/data/dictionary')



    arguments = parser.parse(sys.argv[1:])
    path_3d_dic = arguments["-3D-dic"]
    path_dic_by_slice = arguments["-dic-by-slice"]
    path_dic_by_slice_denoised = arguments["-dic-by-slice-denoised"]

    # FACTORS DEFAULT LEVEL
    # denoising=True, reg='Affine', metric='MI', use_levels=True, weight=2.5, eq=1, mode_weighted_sim=False, weighted_label_fusion=False
    factors_levels = {'denoising': [True, False], 'reg': ['Affine', 'SyN'], 'metric': ['MI', 'MeanSquares'], 'gamma': [2.5, 0], 'eq': [1, 2], 'mode_weight': [False, True], 'weighted_label_fusion': [False, True]}
    factors = {'denoising': 'A', 'reg': 'B', 'metric': 'C', 'gamma': 'D', 'eq': 'E', 'mode_weight': 'F','weighted_label_fusion':'G'}
    # factors = {'A': 'denoising', 'B': 'reg', 'C': 'metric', 'D': 'gamma', 'E': 'eq', 'F': 'mode_weight', 'G': 'weighted_label_fusion'}


    exp_plan = {'exp0': {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0},
                'exp1': {'A': 0, 'B': 0, 'C': 0, 'D': 1, 'E': 1, 'F': 1, 'G': 1},
                'exp2': {'A': 0, 'B': 1, 'C': 1, 'D': 0, 'E': 0, 'F': 1, 'G': 1},
                'exp3': {'A': 0, 'B': 1, 'C': 1, 'D': 1, 'E': 1, 'F': 0, 'G': 0},
                'exp4': {'A': 1, 'B': 0, 'C': 1, 'D': 0, 'E': 1, 'F': 0, 'G': 1},
                'exp5': {'A': 1, 'B': 0, 'C': 1, 'D': 1, 'E': 0, 'F': 1, 'G': 0},
                'exp6': {'A': 1, 'B': 1, 'C': 0, 'D': 0, 'E': 1, 'F': 1, 'G': 0},
                'exp7': {'A': 1, 'B': 1, 'C': 0, 'D': 1, 'E': 0, 'F': 0, 'G': 1},
                }
    experiments_dir = prep(path_3d_dic, path_dic_by_slice, path_dic_by_slice_denoised)

    pool = mp.Pool()
    pool.map(do_loocv, experiments_dir)
    pool.close()
    pool.join()