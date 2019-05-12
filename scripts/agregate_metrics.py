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

    # TODO : replace every [dice_glob dice_mean, ...] and [NoRot, pca, ...] by a list defined at the beginning
    # Loading all csv files :

    dice_metrics = {method: {metric: [] for metric in ["dice_global", "dice_mean", "dice_min", "dice_max", "dice_std"]} for method in ["NoRot", "pca", "hog"]}  # dictionary for metrics in dictionnary for method containing lists of values

    for root, dirnames, filenames in os.walk(folder):  # searching the given directory
        for method in ["NoRot", "pca", "hog"]:
            # search for csv file
            for filename in fnmatch.filter(filenames, "*" + method + ".csv"):
                if not (os.path.isfile(os.path.join(root, filename).replace(method, "NoRot")) and os.path.isfile(os.path.join(root, filename).replace(method, "pca")) and os.path.isfile(os.path.join(root, filename).replace(method, "hog"))):
                    sct.printv("3 csv files not found for file : " + filename)
                    continue  # this block makes sure that there is csv file for the 3 methods
                with open(os.path.join(root, filename), 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    metric_dic = {rows[0]: float(rows[1]) for rows in reader}
                    for metric in ["dice_global", "dice_mean", "dice_min", "dice_max", "dice_std"]:
                        dice_metrics[method][metric].append(metric_dic[metric])
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
