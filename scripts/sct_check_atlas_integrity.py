#!/usr/bin/env python
#########################################################################################
#
# Module converting image files
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener
# Created: 2014-09-22
#
# Dependences:
#   minc-toolkit - http://www.bic.mni.mcgill.ca/ServicesSoftware/ServicesSoftwareMincToolKit
#
# TO DO:
# - check if minc-toolkit is installed. If not, convert files using nibabel
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os
import sys
import commands
import getopt
import sct_utils as sct
import nibabel as nib
import numpy as np

# DEFAULT PARAMETERS
class param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.verbose = 1
        self.file_info_label = 'info_label.txt'
        self.threshold_atlas = 0.5

# constants
ALMOST_ZERO = 0.000001

# main
#=======================================================================================================================
def main():
    
    # Initialization
    path_atlas = ''
    fname_seg = ''
    verbose = param.verbose
    
    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
    
    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hi:o:v:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ('-i'):
            path_atlas = arg
        elif opt in ('-s'):
            fname_seg = arg
        elif opt in ('-v'):
            param.verbose = int(arg)
    
    # display usage if a mandatory argument is not provided
    if path_atlas == '':
        usage()

    # print arguments
    if param.verbose:
        print 'Check input parameters...'
        print '.. Atlas folder path:    '+path_atlas

    # Check for end-caracter of folder path
    if path_atlas[-1] != "/": path_atlas=path_atlas+"/";

    # Check folder existence
    sct.printv('\nCheck atlas existence...', param.verbose)
    sct.check_file_exist(path_atlas)
    
    # Extract atlas info
    atlas_id, atlas_name, atlas_file = read_label_file(path_atlas)
    nb_tracts_total = len(atlas_id)

    # Load atlas
    sct.printv('\nLoad atlas...', param.verbose)
    atlas = np.empty([nb_tracts_total], dtype=object)  # labels(nb_labels_total, x, y, z)
    for i_atlas in range(0, nb_tracts_total):
        atlas[i_atlas] = nib.load(path_atlas+atlas_file[i_atlas]).get_data()

    # Check integrity
    sct.printv('\nCheck atlas integrity...', param.verbose)
    check_integrity(atlas, atlas_name, fname_seg)


#=======================================================================================================================
# Read label.txt file which is located inside label folder
#=======================================================================================================================
def read_label_file(path_info_label):

    # file name of info_label.txt
    fname_label = path_info_label+param.file_info_label

    # Check info_label.txt existence
    sct.check_file_exist(fname_label)

    # Read file
    f = open(fname_label)

    # Extract all lines in file.txt
    lines = [lines for lines in f.readlines() if lines.strip()]

    # separate header from (every line starting with "#")
    lines = [lines[i] for i in range(0, len(lines)) if lines[i][0] != '#']

    # read each line
    label_id = []
    label_name = []
    label_file = []
    for i in range(0, len(lines)-1):
        line = lines[i].split(',')
        label_id.append(int(line[0]))
        label_name.append(line[1])
        label_file.append(line[2][:-1].strip())
    # An error could occur at the last line (deletion of the last character of the .txt file), the 5 following code
    # lines enable to avoid this error:
    line = lines[-1].split(',')
    label_id.append(int(line[0]))
    label_name.append(line[1])
    line[2]=line[2]+' '
    label_file.append(line[2].strip())

    # check if all files listed are present in folder. If not, WARNING.
    print '\nCheck existence of all files listed in '+param.file_info_label+' ...'
    for fname in label_file:
        if os.path.isfile(path_info_label+fname) or os.path.isfile(path_info_label+fname + '.nii') or \
                os.path.isfile(path_info_label+fname + '.nii.gz'):
            print('  OK: '+path_info_label+fname)
            pass
        else:
            print('  WARNING: ' + path_info_label+fname + ' does not exist but is listed in '
                  +param.file_info_label+'.\n')

    # Close file.txt
    f.close()

    return [label_id, label_name, label_file]


#=======================================================================================================================
# Check integrity of the atlas
#=======================================================================================================================
def check_integrity(atlas, atlas_name, method='wath'):

    nb_tracts = len(atlas) # number of tracts

    # if user asks for binary regions, binarize atlas
    if method == 'bin':
        for i in range(0, nb_tracts):
            atlas[i][atlas[i] < param.threshold_atlas] = 0
            atlas[i][atlas[i] >= param.threshold_atlas] = 1

    # if user asks for thresholded weighted-average, threshold atlas
    if method == 'wath':
        for i in range(0, nb_tracts):
            atlas[i][atlas[i] < param.threshold_atlas] = 0

    # Does all the tracts are present?
    sum_tract = []
    for i_atlas in range(0, nb_tracts):
        sum_tract.append(np.sum(atlas[i_atlas]))
        if sum_tract[i_atlas] < ALMOST_ZERO:
            sct.printv('The tract'+atlas_name[i_atlas]+' is non-existent',param.verbose)
    
    # Does the tracts get out the spinal cord?
    # Loading segmentation
    


# Print usage
# ==========================================================================================
def usage():
    print """
        """+os.path.basename(__file__)+"""
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>
            
            DESCRIPTION
            Check the integrity of the spinal cord internal structure (tracts) atlas.

            USAGE
            """+os.path.basename(__file__)+""" -i <data>
                
                MANDATORY ARGUMENTS
                -i <folder>           atlas folder path
                
                OPTIONAL ARGUMENTS
                -s <segmentation>     segmentation image. If not provided, do not check ...
                -v {0,1}              verbose. Default="""+str(param.verbose)+"""
                -h                    help. Show this message
                """
    # exit program
    sys.exit(2)



#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = param()
    # call main function
    main()