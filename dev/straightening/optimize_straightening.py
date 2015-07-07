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
    data_folders = ['C2-C5', 'C1-T12', 'C5-T8']
    data_folders = ['C2-C5_copy', 'C1-T12_copy', 'C5-T8_copy']

    # parameters to optimize
    """parameters = {'algo_fitting': ['hanning', 'nurbs'],
                  'bspline_meshsize': ['5x5x'+str(i) for i in range(5, 16, 2)],
                  'bspline_numberOfLevels': ['2', '3'],
                  'bspline_order': ['2', '3'],
                  'algo_landmark_rigid': ['xy', 'translation', 'translation-xy']}"""
    parameters = {'algo_fitting': ['nurbs'],
                  'algo_landmark_rigid': ['translation-xy'],
                  'all_labels': ['0', '1'],
                  'use_continuous_labels': ['1']}


    perm_params = [dict(zip(parameters, v)) for v in product(*parameters.values())]

    #results_mse = np.empty([len(data_folders), len(perm_params)])
    #results_dist_max = np.empty([len(data_folders), len(perm_params)])

    results_mse = [dict() for _ in xrange(len(data_folders))]
    results_dist_max = [dict() for _ in xrange(len(data_folders))]

    pb = Progress_bar(0, 0, len(data_folders)*len(perm_params), numeric=True)

    for i, folder_name in enumerate(data_folders):
        for index_comb, param in enumerate(perm_params):
            pb.update(i*len(perm_params)+index_comb)

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
            #results[i, index_comb] = pipeline_test.straightening_results_mse[0]
            from numpy import array, mean, std

            results_mse[i][input_straightening_params] = [itm[1] for itm in pipeline_test.straightening_results]
            results_dist_max[i][input_straightening_params] = [itm[2] for itm in pipeline_test.straightening_results]

    pb.update(len(data_folders)*len(perm_params))

    print '\n'

    if len(perm_params) == 2:
        results = dict()
        results['folder'] = []
        results['mse'] = []
        results['dist_max'] = []
        results['all_labels'] = []
        for i, folder_name in enumerate(data_folders):
            temp = []
            for param in results_mse[i]:
                temp.extend([folder_name]*len(results_mse[i][param]))
            results['Dataset'].extend(temp)
            temp = []
            for param in results_mse[i]:
                temp.extend(results_mse[i][param])
            results['MSE'].extend(temp)
            temp = []
            for param in results_dist_max[i]:
                temp.extend(results_dist_max[i][param])
            results['Maximal distance'].extend(temp)
            temp = []
            for param in results_mse[i]:
                opt = 'yes'
                if param[-1] == '0':
                    opt = 'no'
                temp.extend([opt]*len(results_mse[i][param]))
            results['Option labels'].extend(temp)

    from pandas import DataFrame
    df = DataFrame(results)
    print df

    import seaborn as sns
    import matplotlib.pyplot as plt

    f, (ax1, ax2) = plt.subplots(1, 2)

    sns.set(style="whitegrid")
    # Set up the matplotlib figure

    # Draw a violinplot with a narrower bandwidth than the default
    sns.violinplot(x='Dataset', y='MSE', hue="Option labels", data=df, split=True, bw=.2, cut=1, linewidth=1, ax=ax1)
    sns.violinplot(x='Dataset', y='Maximal distance', hue="Option labels", data=df, split=True, bw=.2, cut=1, linewidth=1, ax=ax2)

    # change ylim from 0 to max(mse) and max(dist)x

    plt.show()

    # Finalize the figure
    #sns.despine(left=True, bottom=True)

    import pickle
    pickle.dump(df, open("results_straightening.p", "wb"))
