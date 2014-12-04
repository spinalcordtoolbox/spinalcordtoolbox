#!/usr/bin/env python
#########################################################################################
#
# Test function sct_propseg
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014/10/09
#
# About the license: see the file LICENSE.TXT
#########################################################################################

#import sct_utils as sct
import commands
import shutil
import getopt
import sys
import time
import sct_utils as sct
import os
import nibabel
import numpy as np

class param:
    def __init__(self):
        self.download = 0
        self.remove_tmp_file = 0
        self.verbose = 1
        self.url_git = 'https://github.com/neuropoly/sct_testing_data.git'
        self.path_data = '/home/django/benjamindeleener/data/PropSeg_data/'

def main():
    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],'h:d:p:f:r:a:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit(0)
        if opt == '-d':
            param.download = int(arg)
        if opt == '-p':
            param.path_data = arg
        if opt == '-r':
            param.remove_tmp_file = int(arg)

    start_time = time.time()

    # download data
    if param.download:
        sct.printv('\nDownloading testing data...', param.verbose)
        # remove data folder if exist
        if os.path.exists('sct_testing_data_propseg'):
            sct.printv('WARNING: sct_testing_data already exists. Removing it...', param.verbose, 'warning')
            sct.run('rm -rf sct_testing_data_propseg')
        # clone git repos
        sct.run('git clone '+param.url_git)
        # update path_data field 
        param.path_data = 'sct_testing_data_propseg'

    # get absolute path and add slash at the end
    param.path_data = sct.slash_at_the_end(os.path.abspath(param.path_data), 1)

    # segment all data in t2 folder
    status = []
    for dirname in os.listdir(param.path_data+"t2/"):
        if dirname not in ['._.DS_Store','.DS_Store']:
            for filename in os.listdir(param.path_data+"t2/"+dirname):
                if filename.startswith('t2_') and not filename.endswith('_seg.nii.gz') and not filename.endswith('_detection.nii.gz'):
                    segmentation(param.path_data+"t2/"+dirname+"/"+filename,param.path_data+"t2/"+dirname,'t2')

    sys.exit(0)
    [status.append(test_function(f)) for f in functions if function_to_test == f]
    if not status:
        for f in functions:
            status.append(test_function(f))
    print 'status: '+str(status)

    # display elapsed time
    elapsed_time = time.time() - start_time
    print 'Finished! Elapsed time: '+str(int(round(elapsed_time)))+'s\n'

    # remove temp files
    if param.remove_tmp_file:
        sct.printv('\nRemove temporary files...', param.verbose)
        sct.run('rm -rf '+param.path_tmp, param.verbose)

    e = 0
    if sum(status) != 0:
        e = 1
    print e

    sys.exit(e)

def segmentation(fname_input, output_dir, image_type):
    # parameters
    path_in, file_in, ext_in = sct.extract_fname(fname_input)

    # define command
    cmd = 'sct_propseg_test -i ' + fname_input \
        + ' -o ' + output_dir \
        + ' -t ' + image_type \
        + ' -detect-nii' \

    status, output = sct.run(cmd)

    # check if spinal cord is correctly detected
    # sct_propseg return one point
    # check existence of input files
    segmentation_filename = path_in + file_in + '_seg' + ext_in
    manual_segmentation_filename = path_in + 'manual_' + file_in + ext_in
    detection_filename = path_in + file_in + '_detection' + ext_in

    sct.check_file_exist(detection_filename)
    sct.check_file_exist(segmentation_filename)

    # read nifti input file
    img = nibabel.load(detection_filename)
    # 3d array for each x y z voxel values for the input nifti image
    data = img.get_data()

    # read nifti input file
    img_seg = nibabel.load(manual_segmentation_filename)
    # 3d array for each x y z voxel values for the input nifti image
    data_seg = img_seg.get_data()

    X, Y, Z = (data>0).nonzero()
    status = 0
    for i in range(0,len(X)):
        if data_seg[X[i],Y[i],Z[i]] == 0:
            status = 1
            break;

    if status is not 0:
        sct.printv('ERROR: detected point is not in segmentation',1,'warning')
    else:
        sct.printv('OK: detected point is in segmentation')
    
    cmd_validation = 'sct_dice_coefficient ' + segmentation_filename \
                + ' ' + manual_segmentation_filename \
                + ' -bzmax'

    status_validation, output = sct.run(cmd_validation)
    print output
    return status


if __name__ == "__main__":
    # call main function
    param = param()
    main()