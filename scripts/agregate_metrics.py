#!/usr/bin/env python

# Script that do some cleaning in the output folder and agregate metrics, then plots nice graphs

import sys, os
import sct_utils as sct
from msct_parser import Parser
import shutil
import fnmatch
import csv

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

    for root, dirnames, filenames in os.walk(folder):  # searching the given directory
        # Search for the csv files
        for filename in fnmatch.filter(filenames, "*.csv"):  # if file with nii extension (.nii or .nii.gz) found

            with open(os.path.join(root, filename), 'r') as csvfile:
                reader = csv.reader(csvfile)
                metric_dic = {rows[0]: float(rows[1]) for rows in reader}

    # Cleaning everything :
    # shutil.rmtree(folder)
    # os.mkdir(folder)

    # Processing data

    1

    # Saving everything :







if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
