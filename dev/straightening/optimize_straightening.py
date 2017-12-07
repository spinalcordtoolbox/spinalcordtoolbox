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

            bar = "/".join((str(reading), str(self.max))) + ' - ' + str(round(percent, 2)) + "%\033[K" + \
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
    np.seterr(all='ignore')
    begin = time.time()

    # contrast: t1, t2 or both
    input_t = ["t1", "t2"]
    data_folders = ['C1-T12', 'C2-C5', 'C5-T8']
    load_filename = ''
    #load_filename = 'results_straightening.p'

    # parameters to optimize
    """parameters = {'algo_fitting': ['hanning', 'nurbs'],
                  'bspline_meshsize': ['5x5x'+str(i) for i in range(5, 16, 2)],
                  'bspline_numberOfLevels': ['2', '3'],
                  'bspline_order': ['2', '3'],
                  'algo_landmark_rigid': ['xy', 'translation', 'translation-xy'],
                  'all_labels': ['0', '1'],
                  'use_continuous_labels': ['0', '1']}"""
    parameters = {'algo_fitting': ['nurbs'],
                  'bspline_meshsize': ['5x5x5', '5x5x10', '5x5x15'],
                  'bspline_numberOfLevels': ['2', '3'],
                  'bspline_order': ['2', '3'],
                  'algo_landmark_rigid': ['translation-xy'],
                  'all_labels': ['1'],
                  'use_continuous_labels': ['1']}

    perm_params = [dict(zip(parameters, v)) for v in product(*parameters.values())]

    subjects = [dict() for _ in xrange(len(data_folders))]
    contrasts = [dict() for _ in xrange(len(data_folders))]
    results_mse = [dict() for _ in xrange(len(data_folders))]
    results_dist_max = [dict() for _ in xrange(len(data_folders))]

    if load_filename:
        from pandas import DataFrame
        import pickle

        df = pickle.load(open(load_filename, "rb"))

    else:
        pb = Progress_bar(0, 0, len(data_folders)*len(perm_params), numeric=True)

        for i, folder_name in enumerate(data_folders):
            for index_comb, param in enumerate(perm_params):
                pb.update(i*len(perm_params)+index_comb)

                input_straightening_params = ''
                for key in param:
                    input_straightening_params += key + '=' + param[key] + ','
                input_straightening_params = input_straightening_params[:-1]

                # copy of the folder
                folder_name_complete = os.path.join("data_" + folder_name + "_" + input_straightening_params)
                sct.run("pwd", verbose=0)
                #sct.run("cp -R " + os.path.join("original_data", folder_name) + " " + folder_name_complete)
                if os.path.exists(folder_name_complete):
                    sct.run("rm -rf " + folder_name_complete, verbose=0)
                sct.run("cp -R " + os.path.join("original_data", folder_name) + " " + folder_name_complete, verbose=0)

                subjects[i][input_straightening_params] = []
                contrasts[i][input_straightening_params] = []
                results_mse[i][input_straightening_params] = []
                results_dist_max[i][input_straightening_params] = []
                if "t1" in input_t:
                    pipeline_test = sct_register_pipeline.Pipeline(folder_name_complete, "t1", seg=False,
                                                                   straightening=True,
                                                                   straightening_params=input_straightening_params,
                                                                   verbose=0)
                    pipeline_test.cpu_count = 12
                    pipeline_test.compute()

                    subjects[i][input_straightening_params].extend(
                        [itm[0] for itm in pipeline_test.straightening_results])
                    contrasts[i][input_straightening_params].extend(
                        ["t1" for itm in pipeline_test.straightening_results])
                    results_mse[i][input_straightening_params].extend(
                        [itm[1] for itm in pipeline_test.straightening_results])
                    results_dist_max[i][input_straightening_params].extend(
                        [itm[2] for itm in pipeline_test.straightening_results])

                if "t2" in input_t:
                    pipeline_test = sct_register_pipeline.Pipeline(folder_name_complete, "t2", seg=False,
                                                                   straightening=True,
                                                                   straightening_params=input_straightening_params,
                                                                   verbose=0)
                    pipeline_test.cpu_count = 12
                    pipeline_test.compute()

                    subjects[i][input_straightening_params].extend(
                        [itm[0] for itm in pipeline_test.straightening_results])
                    contrasts[i][input_straightening_params].extend(
                        ["t2" for itm in pipeline_test.straightening_results])
                    results_mse[i][input_straightening_params].extend(
                        [itm[1] for itm in pipeline_test.straightening_results])
                    results_dist_max[i][input_straightening_params].extend(
                        [itm[2] for itm in pipeline_test.straightening_results])

        pb.update(len(data_folders)*len(perm_params))

        results = dict()
        results['Subject'] = []
        results['Contrast'] = []
        results['Dataset'] = []
        results['MSE'] = []
        results['Maximal distance'] = []
        results['Parameters'] = []
        for i, folder_name in enumerate(data_folders):
            temp_subject = []
            for param in subjects[i]:
                temp_subject.extend(subjects[i][param])
            results['Subject'].extend(temp_subject)
            temp_contrast = []
            for param in subjects[i]:
                temp_contrast.extend(contrasts[i][param])
            results['Contrast'].extend(temp_contrast)
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
                temp.extend([param]*len(results_mse[i][param]))
            results['Parameters'].extend(temp)

        from pandas import DataFrame
        df = DataFrame(results)

    print '\n'
    print df

    import seaborn as sns
    import matplotlib.pyplot as plt
    from numpy import argsort

    bandwidth = 0.2

    sns.set(style="whitegrid", palette="pastel", color_codes=True)
    xlabels = ['P' + str(i) for i in range(1, len(perm_params) + 1)]

    for i, folder_name in enumerate(data_folders):
        df_folder = df[df.Dataset == folder_name]
        df_folder_t1 = df_folder[df_folder.Contrast == 't1']
        df_folder_t2 = df_folder[df_folder.Contrast == 't2']

        # computation of mean values by groups
        mean_df_folder_t1_mse = df_folder_t1.groupby('Parameters')['MSE'].mean()
        mean_df_folder_t1_mse_values = mean_df_folder_t1_mse.values
        mean_df_folder_t1_dist = df_folder_t1.groupby('Parameters')['Maximal distance'].mean()
        mean_df_folder_t1_dist_values = mean_df_folder_t1_dist.values
        mean_df_folder_t2_mse = df_folder_t2.groupby('Parameters')['MSE'].mean()
        mean_df_folder_t2_mse_values = mean_df_folder_t2_mse.values
        mean_df_folder_t2_dist = df_folder_t2.groupby('Parameters')['Maximal distance'].mean()
        mean_df_folder_t2_dist_values = mean_df_folder_t2_dist.values

        # sort the params for this particular dataset
        mean_df_folder_t1_mse_values_sorted = [mean_df_folder_t1_mse_values[k] for k in argsort(mean_df_folder_t1_mse_values)]
        x_sorted_df_folder = [mean_df_folder_t1_mse.index[k] for k in argsort(mean_df_folder_t1_mse_values)]
        mean_df_folder_t1_dist_values_sorted = [mean_df_folder_t1_dist_values[i] for i in argsort(mean_df_folder_t1_dist_values)]
        #x_sorted_df_folder_t1_dist = [mean_df_folder_t1_dist.index[i] for i in argsort(mean_df_folder_t1_dist_values)]
        mean_df_folder_t2_mse_values_sorted = [mean_df_folder_t2_mse_values[i] for i in argsort(mean_df_folder_t2_mse_values)]
        #x_sorted_df_folder_t2_mse = [mean_df_folder_t2_mse.index[i] for i in argsort(mean_df_folder_t2_mse_values)]
        mean_df_folder_t2_dist_values_sorted = [mean_df_folder_t2_dist_values[i] for i in argsort(mean_df_folder_t2_dist_values)]
        #x_sorted_df_folder_t2_dist = [mean_df_folder_t2_dist.index[i] for i in argsort(mean_df_folder_t2_dist_values)]

        xlabels_sorted = [xlabels[k] for k in argsort(mean_df_folder_t1_mse_values)]

        # Set up the matplotlib figure
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey='row')
        f.suptitle(folder_name, fontsize=18, fontweight='bold')

        ax1.set_title('T1', fontsize=14, fontweight='bold')
        ax2.set_title('T2', fontsize=14, fontweight='bold')

        plt.hold(True)

        # Draw a violinplot with a narrower bandwidth than the default
        sns.violinplot(x='Parameters', y='MSE', data=df_folder_t1, inner="point", bw=bandwidth, linewidth=1,
                       scale='count', ax=ax1, color='b', order=x_sorted_df_folder)
        ax1.scatter(x=ax1.xaxis.get_majorticklocs(), y=mean_df_folder_t1_mse_values_sorted, s=40, c='red', alpha=0.75)
        sns.violinplot(x='Parameters', y='MSE', data=df_folder_t2, inner="point", bw=bandwidth, linewidth=1,
                       scale='count', ax=ax2, color='y', order=x_sorted_df_folder)
        ax2.scatter(x=ax2.xaxis.get_majorticklocs(), y=mean_df_folder_t2_mse_values_sorted, s=40, c='red', alpha=0.75)
        sns.violinplot(x='Parameters', y='Maximal distance', data=df_folder_t1, inner="point", bw=bandwidth,
                       linewidth=1, scale='count', ax=ax3, color='b', order=x_sorted_df_folder)
        ax3.scatter(x=ax3.xaxis.get_majorticklocs(), y=mean_df_folder_t1_dist_values_sorted, s=40, c='red', alpha=0.75)
        sns.violinplot(x='Parameters', y='Maximal distance', data=df_folder_t2, inner="point", bw=bandwidth,
                       linewidth=1, scale='count', ax=ax4, color='y', order=x_sorted_df_folder)
        ax4.scatter(x=ax4.xaxis.get_majorticklocs(), y=mean_df_folder_t2_dist_values_sorted, s=40, c='red', alpha=0.75)

        # display labels:
        print '\n' + folder_name
        for k, label in enumerate(xlabels_sorted):
            print label + ': ' + x_sorted_df_folder[k]

        # change ylim
        ylim_min_mse = min(df_folder_t1['MSE'].min(), df_folder_t2['MSE'].min())
        ylim_min_dist = min(df_folder_t1['Maximal distance'].min(), df_folder_t2['Maximal distance'].min())
        ylim_max_mse = max(df_folder_t1['MSE'].max(), df_folder_t2['MSE'].max())
        ylim_max_dist = max(df_folder_t1['Maximal distance'].max(), df_folder_t2['Maximal distance'].max())
        ax1.set(ylim=(ylim_min_mse - 0.3, ylim_max_mse + 0.3))
        ax2.set(ylim=(ylim_min_mse - 0.3, ylim_max_mse + 0.3))
        ax3.set(ylim=(ylim_min_dist - 0.5, ylim_max_dist + 0.5))
        ax4.set(ylim=(ylim_min_dist - 0.5, ylim_max_dist + 0.5))

        ax1.set_xticklabels(xlabels_sorted)
        ax2.set_xticklabels(xlabels_sorted)
        ax3.set_xticklabels(xlabels_sorted)
        ax4.set_xticklabels(xlabels_sorted)

        sns.despine(left=True, bottom=True)
        plt.savefig(folder_name + '.eps', bbox_inches='tight')
        plt.show()

        import pickle
        pickle.dump(df, open("results_straightening.p", "wb"))
