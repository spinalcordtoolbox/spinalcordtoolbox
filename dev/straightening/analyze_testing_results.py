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
import numpy as np
import pandas as pd
import pickle


def compare_straightening(filenames, labels):
    frames = []
    for i, data in enumerate(filenames):
        df = pickle.load(open(data, "rb"))
        df.loc[:, 'param'] = pd.Series([labels[i]] * len(df['status']), index=df.index)
        df.loc[:, '% compute accuracy'] = pd.Series(100.0 * df['duration_accuracy_results'] / df['duration'], index=df.index)
        frames.append(df)

    df_results = pd.concat(frames)

    print '\n'
    print df_results

    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(style="whitegrid", palette="pastel", color_codes=True)

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    f.suptitle('Effect of warping field resampling on straightening accuracy', fontsize=18, fontweight='bold')

    plt.hold(True)

    bandwidth = 0.2

    # Draw a violinplot with a narrower bandwidth than the default
    sns.violinplot(x='param', y='rmse', data=df_results, inner="point", bw=bandwidth, linewidth=1,
                   scale='count', ax=ax1, color='y')
    ax1.set_xlabel('')
    ax1.set_ylabel('Mean distance [mm]')
    ax1.set_ylim([0, 1.5])
    sns.violinplot(x='param', y='dist_max', data=df_results, inner="point", bw=bandwidth,
                   linewidth=1, scale='count', ax=ax2, color='y')
    ax2.set_xlabel('')
    ax2.set_ylabel('Maximum distance [mm]')
    ax2.set_ylim([0, 5])
    sns.violinplot(x='param', y='duration', data=df_results, inner="point", bw=bandwidth, linewidth=1,
                   scale='count', ax=ax3, color='y')
    ax3.set_xlabel('')
    ax3.set_ylabel('Duration [sec]')
    ax3.set_ylim([0, 200])
    sns.violinplot(x='param', y='% compute accuracy', data=df_results, inner="point", bw=bandwidth, linewidth=1,
                   scale='count', ax=ax4, color='y')
    ax4.set_xlabel('')
    ax4.set_ylabel('% compute accuracy')
    ax4.set_ylim([0, 20])

    plt.show()


def compare_propseg(filenames, labels):
    frames = []
    for i, data in enumerate(filenames):
        df = pickle.load(open(data, "rb"))
        df.loc[:, 'param'] = pd.Series([labels[i]] * len(df['status']), index=df.index)
        frames.append(df)

    df_results = pd.concat(frames)

    print '\n'
    print df_results

    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(style="whitegrid", palette="pastel", color_codes=True)

    f, (ax1, ax2) = plt.subplots(1, 2)
    f.suptitle('Hough vs OptiC spinal cord detection performances', fontsize=18, fontweight='bold')

    plt.hold(True)

    bandwidth = 0.2

    # Draw a violinplot with a narrower bandwidth than the default
    sns.violinplot(x='param', y='dice_segmentation', data=df_results, inner="point", bw=bandwidth, linewidth=1,
                   scale='count', ax=ax1, color='y')
    ax1.set_xlabel('')
    ax1.set_ylabel('Dice Coefficient')
    ax1.set_ylim([0, 1.5])
    sns.violinplot(x='param', y='duration [s]', data=df_results, inner="point", bw=bandwidth, linewidth=1,
                   scale='count', ax=ax2, color='y')
    ax2.set_xlabel('')
    ax2.set_ylabel('Duration [sec]')
    ax2.set_ylim([0, 200])

    plt.show()

#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    np.seterr(all='ignore')

    folder = '/Users/benjamindeleener/data/results/'
    labels = ['Hough', 'OptiC']

    # T1-weighted
    filenames = [folder + '/results_test_sct_propseg_170415073124.pickle',
                 folder + '/results_test_sct_propseg_170414225351.pickle']
    #compare_propseg(filenames, labels)

    # T2-weighted
    filenames = [folder + '/results_test_sct_propseg_170414232651.pickle',
                 folder + '/results_test_sct_propseg_170414222158.pickle']
    #compare_propseg(filenames, labels)

    # T2*-weighted
    filenames = [folder + '/results_test_sct_propseg_170415082056.pickle',
                 folder + '/results_test_sct_propseg_170415075655.pickle']
    compare_propseg(filenames, labels)



