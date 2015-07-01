#!/usr/bin/env python
#########################################################################################
#
# Script to optimize the spinal cord straightening
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener
# Modified: 2015-06-17
#
# About the license: see the file LICENSE.TXT
#########################################################################################
from msct_parser import Parser
import sys
import os
import sct_utils as sct
import time
from datetime import datetime, timedelta
import sct_register_pipeline
import numpy as np
from itertools import product
from sys import stdout

class Progress_bar(object):
    def __init__(self, x=0, y=0, mx=1, numeric=False):
        self.x = x
        self.y = y
        self.width = 50
        self.current = 0
        self.max = mx
        self.numeric = numeric
        self.start_time = time.time()
        self.elapsed_time = 0

    def update(self, reading):
        self.elapsed_time = round(time.time() - begin, 2)
        percent = float(reading) * 100.0 / float(self.max)
        cr = '\r'

        if not self.numeric:
            bar = '#' * int(percent)
        else:
            remaining_time = 0
            if percent > 0:
                remaining_time = (self.elapsed_time * 100.0 / percent) - self.elapsed_time
            elapsed = datetime(1, 1, 1) + timedelta(seconds=self.elapsed_time)
            remain = datetime(1, 1, 1) + timedelta(seconds=remaining_time)

            bar = "/".join((str(reading), str(self.max))) + ' - ' + str(percent) + "%\033[K" + \
                  " Elapsed time: " + str(elapsed.day - 1) + " day(s), " + str(elapsed.hour) + ":" + str(elapsed.minute) \
                  + ":" + str(elapsed.second) + \
                  ", Remaining time: " + str(remain.day-1) + " day(s), " + str(remain.hour) + ":" + str(remain.minute) \
                  + ":" + str(remain.second)

        stdout.write(cr)
        stdout.write(bar)
        stdout.flush()
        self.current = percent

        if percent == 100:
            stdout.write(cr)
            if not self.numeric:
                stdout.write(" " * int(percent))
                stdout.write(cr)
                stdout.flush()
            else:
                stdout.write(" " * (len(str(self.max))*2 + 8))
                stdout.write(cr)
                stdout.flush()

#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    begin = time.time()

    # contrast: t1, t2 or both
    input_t = "t2"
    data_folders = ['C1-T3']

    # parameters to optimize
    """parameters = {'algo_fitting': ['hanning', 'nurbs'],
                  'bspline_meshsize': ['5x5x'+str(i) for i in range(5, 16, 2)],
                  'bspline_numberOfLevels': ['2', '3'],
                  'bspline_order': ['2', '3'],
                  'algo_landmark_rigid': ['xy', 'translation', 'translation-xy']}"""
    parameters = {'algo_fitting': ['hanning', 'nurbs'],
                  'algo_landmark_rigid': ['None', 'translation', 'translation-xy'],
                  'all_labels': ['0', '1'],
                  'use_continuous_labels':['0', '1']}


    perm_params = [dict(zip(parameters, v)) for v in product(*parameters.values())]

    results = np.empty([len(data_folders), len(perm_params)])

    pb = Progress_bar(0, 0, len(perm_params), numeric=True)

    for i, folder_name in enumerate(data_folders):
        for index_comb, param in enumerate(perm_params):
            pb.update(index_comb)

            input_straightening_params = ''
            for key in param:
                input_straightening_params += key + '=' + param[key] + ','
            input_straightening_params = input_straightening_params[:-1]

            # copy of the folder
            folder_name_complete = "data_" + folder_name + "_" + input_straightening_params
            sct.run("pwd", verbose=0)
            #sct.run("cp -R original_data/" + folder_name + " " + folder_name_complete)
            if os.path.exists(folder_name_complete):
                sct.run("rm -rf " + folder_name_complete, verbose=0)
            sct.run("cp -R original_data/" + folder_name + " " + folder_name_complete, verbose=0)

            pipeline_test = sct_register_pipeline.Pipeline(folder_name_complete, input_t, seg=False, straightening=True,
                                                           straightening_params=input_straightening_params, verbose=0)
            pipeline_test.cpu_count = 6
            pipeline_test.compute()

            # pipeline_test.straightening_results_dist_max
            results[i, index_comb] = pipeline_test.straightening_results_mse[0]

    pb.update(len(perm_params))
    print '\n'
    print results
    min_value = results.ravel().argmin()
    i_min, j_min = np.unravel_index(min_value, results.shape)
    print "Minimum= "+str(results[i, j_min])
    print "Optimal parameters:"
    for k, v in perm_params[j_min].items():
        print k + ': ' + v

    elapsed_time = round(time.time() - begin, 2)

    import pickle
    pickle.dump(results, open("results_straightening.p", "wb"))
