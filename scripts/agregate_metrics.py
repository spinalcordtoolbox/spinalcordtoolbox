#!/usr/bin/env python

# Script that do some cleaning in the output folder and agregate metrics, then plots nice graphs

import sys, os
import sct_utils as sct
from msct_parser import Parser
import shutil
import fnmatch
import csv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def get_parser():

    parser = Parser(__file__)
    parser.usage.set_description('blabla')
    parser.add_option(name="-i",
                      type_value="folder",
                      description="Folder with csv files inside to agregate",
                      mandatory=True,
                      example="/home/cool_project/results")

    return parser



def main(args=None):

    # Parser :
    if not args:
        args = sys.argv[1:]
    parser = get_parser()
    arguments = parser.parse(args)
    folder = arguments['-i']
    # init global dic :
    dice_metrics = {method: {metric: [] for metric in ["dice_global", "dice_mean", "dice_min", "dice_max", "dice_std"]} for method in ["NoRot", "pca", "hog"]}  # dictionary for metrics in dictionnary for method containing lists of values

    for root, dirnames, filenames in os.walk(folder):  # searching the given directory

        for filename_NoRot in fnmatch.filter(filenames, "*NoRot.csv"):

            filename_pca = os.path.join(root, filename_NoRot.replace("NoRot", "pca"))
            filename_hog = os.path.join(root, filename_NoRot.replace("NoRot", "hog"))
            if not (os.path.isfile(filename_pca) and os.path.isfile(filename_hog)):
                sct.printv("3 csv files not found for file : " + filename_NoRot)
                continue  # this block makes sure that there is csv file for the 3 methods

            # Open and verify presence of 5 metrics
            with open(os.path.join(root, filename_NoRot), 'r') as csvfile:
                reader = csv.reader(csvfile)
                metric_dic_NoRot = {rows[0]: float(rows[1]) for rows in reader}
                if len(metric_dic_NoRot) != 5:
                    sct.printv("5 Metrics not present in csv : " + filename_NoRot)
                    continue
            with open(os.path.join(root, filename_pca), 'r') as csvfile:
                reader = csv.reader(csvfile)
                metric_dic_pca = {rows[0]: float(rows[1]) for rows in reader}
                if len(metric_dic_pca) != 5:
                    sct.printv("5 Metrics not present in csv : " + filename_pca)
                    continue
            with open(os.path.join(root, filename_hog), 'r') as csvfile:
                reader = csv.reader(csvfile)
                metric_dic_hog = {rows[0]: float(rows[1]) for rows in reader}
                if len(metric_dic_hog) != 5:
                    sct.printv("5 Metrics not present in csv : " + filename_hog)
                    continue

            # Now append the metrics to the general dic, if program arrives at this step it means that the 3 .csv exist and have the 5 metrics inside
            for metric in ["dice_global", "dice_mean", "dice_min", "dice_max", "dice_std"]:
                dice_metrics["NoRot"][metric].append(metric_dic_NoRot[metric])
                dice_metrics["pca"][metric].append(metric_dic_pca[metric])
                dice_metrics["hog"][metric].append(metric_dic_hog[metric])


    nb_subjects = len(next(iter(next(iter(dice_metrics.values())).values())))  # just to get number of subjects (we access the first element of dic twice)


    # Cleaning everything :
    # shutil.rmtree(folder)
    # os.mkdir(folder)

    # Processing data
    # matplotlib.use('Agg')  # prevent display figure
    fig = plt.figure(figsize=(20, 40))
    fig.suptitle("Histograms of dataset with " + str(nb_subjects) + " images")
    for k, metric in enumerate(["dice_global", "dice_mean", "dice_min", "dice_max", "dice_std"]):
        plt.subplot(2, 3, k + 1)
        if metric == "dice_std":
            range_metric = None
            xlabel = "std"
        else:
            range_metric = (0, 1)
            xlabel = "dice score"
        plt.hist((dice_metrics["NoRot"][metric], dice_metrics["pca"][metric], dice_metrics["hog"][metric]), bins=10, range=range_metric)
        plt.ylabel("count")
        plt.xlabel(xlabel)
        plt.title(metric + " histogram")
        plt.legend(["NoRot", "pca", "hog"])

    plt.subplot(2, 3, 6)
    #Building the np array to plot the table
    data = np.zeros((6, 3))
    for col, metric in enumerate(["dice_global", "dice_mean", "dice_min", "dice_max", "dice_std"]):
        list_argmax = list(np.argmax((dice_metrics["NoRot"]["dice_global"], dice_metrics["pca"]["dice_global"], dice_metrics["hog"]["dice_global"]), axis=0))
        nb_NoRot_best = list_argmax.count(0)
        nb_PCA_best = list_argmax.count(1)
        nb_HOG_best = list_argmax.count(2)
        for row, method in enumerate(["NoRot", "pca", "hog"]):
            data[col, row] = np.mean(dice_metrics[method][metric])
        data[5, :] = [nb_NoRot_best, nb_PCA_best, nb_HOG_best]

    plt.table(cellText=data, colLabels=["NoRot", "pca", "hog"], rowLabels=["dice_global", "dice_mean", "dice_min", "dice_max", "dice_std", "No times best"], loc="center")

    # Saving everything :
    plt.savefig(folder + "/histograms.png")


if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
