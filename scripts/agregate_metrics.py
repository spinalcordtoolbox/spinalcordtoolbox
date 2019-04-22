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

    # Loading all csv files :

    dice_glob = []
    dice_mean = []
    dice_min = []
    dice_max = []
    dice_std = []
    nb_subjects = 0

    for root, dirnames, filenames in os.walk(folder):  # searching the given directory
        # Search for the csv files
        for filename in fnmatch.filter(filenames, "*.csv"):  # if file with nii extension (.nii or .nii.gz) found

            with open(os.path.join(root, filename), 'r') as csvfile:
                reader = csv.reader(csvfile)
                metric_dic = {rows[0]: float(rows[1]) for rows in reader}
                dice_glob.append(metric_dic['Dice global'])
                dice_mean.append(metric_dic['Mean Dice per slice'])
                dice_min.append(metric_dic['Min Dice'])
                dice_max.append(metric_dic['Max Dice'])
                dice_std.append(metric_dic['STD Dice'])
                nb_subjects += 1


    # Cleaning everything :
    # shutil.rmtree(folder)
    # os.mkdir(folder)

    # Processing data
    # matplotlib.use('Agg')  # prevent display figure
    fig = plt.figure(figsize=(8, 10))
    fig.suptitle("Histograms of dataset with " + str(nb_subjects) + " images")
    plt.subplot(231)
    plt.hist(dice_glob, bins=10, range=(0, 1))
    plt.title("Global dice histogram")
    plt.xlabel("dice score")
    plt.ylabel("count")
    plt.subplot(232)
    plt.hist(dice_mean, bins=10, range=(0, 1))
    plt.title("Mean dice histogram")
    plt.xlabel("dice score")
    plt.ylabel("count")
    plt.subplot(233)
    plt.hist(dice_min, bins=10, range=(0, 1))
    plt.title("Min dice histogram")
    plt.xlabel("dice score")
    plt.ylabel("count")
    plt.subplot(234)
    plt.hist(dice_max, bins=10, range=(0, 1))
    plt.title("Max dice histogram")
    plt.xlabel("dice score")
    plt.ylabel("count")
    plt.subplot(235)
    plt.hist(dice_glob, bins=10)
    plt.title("STD dice histogram")
    plt.xlabel("dice score")
    plt.ylabel("count")

    # Saving everything :

    plt.savefig(folder + "/histograms.png")







if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
