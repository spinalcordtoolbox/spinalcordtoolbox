#!/usr/bin/env python
#########################################################################################
#
# Extract metrics within spinal labels as defined by the white matter atlas.
# The folder atlas should have a txt file that lists all tract files with labels.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Eddie Magnide, Simon Levy, Charles Naaman, Julien Cohen-Adad
# Modified: 2014-07-30
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# Import common Python libraries
import os
import getopt
import sys
import time
import glob
import re
import commands
import nibabel as nib
import sct_utils as sct
import numpy as np

# get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')
import sct_utils as sct

# constants
ALMOST_ZERO = 0.000001

class param:
    def __init__(self):
        self.debug = 1
        self.method = 'wa'
        self.path_label = path_sct+'/data/template'  # default is toolbox
        self.verbose = 1
        self.labels_of_interest = ''  # list. example: '1,3,4'. . For all labels, leave empty.
        self.vertebral_levels = ''
        self.slices_of_interest = ''  # 2-element list corresponding to zmin,zmax. example: '5,8'. For all slices, leave empty.
        self.average_all_labels = 0  # average all labels together after concatenation
        self.fname_output = 'quantif_metrics.txt'
        self.file_info_label = 'info_label.txt'



#=======================================================================================================================
# main
#=======================================================================================================================
def main():
    # Initialization to defaults parameters
    fname_data = '' # data is empty by default
    path_label = param.path_label
    method = param.method # extraction mode by default
    labels_of_interest = param.labels_of_interest
    slices_of_interest = param.slices_of_interest
    vertebral_levels = param.vertebral_levels
    average_all_labels = param.average_all_labels
    fname_output = param.fname_output
    file_info_label = param.file_info_label
    start_time = time.time()  # save start time for duration
    verbose = param.verbose

    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        fname_data = path_sct+'/testing/data/errsm_23/mt/mtr.nii.gz'
        path_label = path_sct+'/testing/sct_warp_template/results/label/template' #'/testing/data/errsm_23/label/atlas'
        method = 'wa'
        labels_of_interest = '0, 2'
        slices_of_interest = '2:4'
        vertebral_levels = ''#'1:3'
        average_all_labels = 0
        fname_output = path_sct+'/testing/sct_extract_metric/results/quantif_mt_debug.txt'


    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'haf:i:l:m:o:v:z:') # define flags
    except getopt.GetoptError as err: # check if the arguments are defined
        print str(err) # error
        usage() # display usage
    for opt, arg in opts: # explore flags
        if opt in '-a':
            average_all_labels = 1
        elif opt in '-f':
            path_label = os.path.abspath(arg)  # save path of labels folder
        elif opt == '-h': # help option
            usage() # display usage
        elif opt in '-i':
            fname_data = arg
        elif opt in '-l':
            labels_of_interest = arg
        elif opt in '-m': # method for metric extraction
            method = arg
        elif opt in '-o': # output option
            fname_output = arg  # fname of output file
        elif opt in '-v': # vertebral levels option, if the user wants to average the metric across specific vertebral levels
             vertebral_levels = arg
        elif opt in '-z': # slices numbers option
            slices_of_interest = arg # save labels numbers


    #TODO: check if the case where the input images are not in AIL orientation is taken into account (if not, implement it)

    # Display usage with tract parameters by default in case files aren't chosen in arguments inputs
    if fname_data == '' or path_label == '':
        usage()

    # Check existence of data file
    sct.printv('\nCheck file existence...', verbose)
    sct.check_file_exist(fname_data)

    # Extract data path/file/extension
    path_data, file_data, ext_data = sct.extract_fname(fname_data)

    # add slash at the end
    path_label = sct.slash_at_the_end(path_label, 1)
    #if not os.path.isdir(path_label):
    #    print('\nERROR: ' + path_label + ' does not exist. Exit program.\n')
    #    sys.exit(2)
    # TODO

    # Check input parameters
    check_method(method)

    # Extract label info
    label_id, label_name, label_file = read_label_file(path_label)
    nb_labels_total = len(label_id)

    ## update label_id given user input
    #if label_choice == 0:
    #    nb_labels = range(0, nb_labels_total)

    # check consistency of label input parameter
    # TODO: test this
    label_id = check_labels(labels_of_interest, nb_labels_total)
    nb_labels = len(label_id)

    # print parameters
    print '\nCheck parameters:'
    print '  data ................... '+fname_data
    print '  folder label ........... '+path_label


    # Load image
    sct.printv('\nLoad image...', verbose)
    data = nib.load(fname_data).get_data()
    sct.printv('  Done.', verbose)

    # Get dimensions of data
    sct.printv('\nGet dimensions of data...', verbose)
    nx, ny, nz = data.shape
    sct.printv('.. '+str(nx)+' x '+str(ny)+' x '+str(nz), verbose)

    # TODO: check consistency of size between atlas and data
    # open one atlas file and check nx, ny and nz

    # load label
    sct.printv('\nLoad labels...', verbose)
    labels = np.empty([nb_labels, nx, ny, nz], dtype=object)  # labels(nb_labels, x, y, z)
    for i_label in range(0, nb_labels):
        labels[i_label, :, :, :] = nib.load(path_label+label_file[label_id[i_label]]).get_data()
    sct.printv('  Done.', verbose)

    # Get dimensions of atlas
    # TODO: no need to do that if size consistency check is done before
    sct.printv('\nGet dimensions of label...', verbose)
    nx_atlas, ny_atlas, nz_atlas = labels[i_label, :, :, :].shape
    sct.printv('.. '+str(nx_atlas)+' x '+str(ny_atlas)+' x '+str(nz_atlas), verbose)

    # if user selected vertebral levels, then update variable slices_of_interest (i.e., zmin, zmax)
    # TODO: function here

    if vertebral_levels != '':
        if slices_of_interest != '':
            print '\nERROR: You cannot select BOTH vertebral levels AND slice numbers.'
            usage()
        else:
            slices_of_interest = get_slices_matching_with_vertebral_levels(data,vertebral_levels,path_label,label_name,label_file)

    # select slice of interest by cropping data and labels
    if slices_of_interest != '':
        data = remove_slices(data, slices_of_interest)
        labels = remove_slices(labels, slices_of_interest)

    # if user wants to get unique value across labels, then combine all labels together
    if average_all_labels == 1:
        labels = np.sum(labels, axis=0)
        # TODO: instead of 0, make it clear for the user that all labels are concatenated
        label_id = [0]

    # extract metrics within labels
    # labels can be 3d or 4d
    metric_mean, metric_std = extract_metric_within_tract(data, labels, method)  # mean and std are lists.

    # display metrics
    print '\nEstimated metrics:\n'+str(metric_mean)

    # save metrics
    if fname_output != '':
        save_metrics(label_id, label_name, slices_of_interest, vertebral_levels, metric_mean, metric_std, fname_output)

    # Print elapsed time
    print 'Elapsed time : ' + str(int(round(time.time() - start_time))) + ' sec'

    # end of main.
    print



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

    # read each line
    label_id = []
    label_name = []
    label_file = []
    for i in range(0, len(lines)):
        line = lines[i].split(',')
        label_id.append(int(line[0]))
        label_name.append(line[1])
        label_file.append(line[2][:-1].replace(" ", ""))

    # check if all files listed are present in folder. If not, WARNING.
    print 'Check if all files listed in '+param.file_info_label+' are indeed present in +'+path_info_label+' ...\n'
    for fname in label_file:
        if os.path.isfile(path_info_label+fname) or os.path.isfile(path_info_label+fname + '.nii') or os.path.isfile(path_info_label+fname + '.nii.gz'):
            print('  OK: '+path_info_label+fname)
            pass
        else:
            print('  WARNING: ' + path_info_label+fname + ' does not exist but is listed in '+param.file_info_label+'.\n')


    # Close file.txt
    f.close()

    return [label_id, label_name, label_file]


#=======================================================================================================================
# get_slices_matching_with_vertebral_levels
#=======================================================================================================================
def get_slices_matching_with_vertebral_levels(metric_data, vertebral_levels,path_label, label_name, label_file):
    """Return the slices of the input image corresponding to the vertebral levels given as argument."""

    # check existence of a vertebral labeling file
    fname_vertebral_labeling = path_label + label_file[label_name.index(' vertebral labeling')]
    sct.check_file_exist(fname_vertebral_labeling)

    # Convert the selected vertebral levels chosen into a 2-element list [start_level end_level]
    vert_levels_list = [int(x) for x in vertebral_levels.split(':')]

    # If only one vertebral level was selected (n), consider as n:n
    if len(vert_levels_list) == 1:
        vert_levels_list = [vert_levels_list[0], vert_levels_list[0]]

    # Check if there are only two values [start_level, end_level] and if the end level is higher than the start level
    if (len(vert_levels_list) > 2) or (vert_levels_list[0] > vert_levels_list[1]):
        print '\nERROR:  "' + vertebral_levels + '" is not correct. Enter format "1:4". Exit program.\n'
        sys.exit(2)

    # Read files vertebral_labeling.nii.gz
    print '\nRead files '+fname_vertebral_labeling+'...'

    # Load the vertebral labeling file and get the data in array format
    data_vert_labeling = nib.load(fname_vertebral_labeling).get_data()

    # Extract metric data size X, Y, Z
    [mx, my, mz] = metric_data.shape
    # Extract vertebral labeling data size X, Y, Z
    [vx, vy, vz] = data_vert_labeling.shape

    # Initialisation of check error flag
    exit_program = 0

    # Check if sizes along X are the same
    if mx != vx:
        print '\tERROR: Size of vertebral_labeling.nii.gz along X is not the same as the metric data.'
        exit_program = 1
    # Check if sizes along Y are the same
    if my != vy:
        print '\tERROR: Size of vertebral_labeling.nii.gz along Y is not the same as the metric data.'
        exit_program = 1
    # Check if sizes along Z are the same
    if mz != vz:
        print '\tERROR: Size of vertebral_labeling.nii.gz along Z is not the same as the metric data.'
        exit_program = 1

    # Check if the vertebral levels selected are not available in the input image
    if vert_levels_list[0] < int(np.ndarray.min(data_vert_labeling)) or vert_levels_list[1] > int(np.ndarray.max(data_vert_labeling)):
        print '\tERROR: The vertebral levels you selected are not available in the input image.'
        exit_program = 1

    # Exit program if an error was detected
    if exit_program == 1 :
        print '\nExit program.\n'
        sys.exit(2)


    # Extract the X, Y, Z positions of voxels belonging to the first vertebral level
    X_bottom_level, Y_bottom_level, Z_bottom_level = (data_vert_labeling==vert_levels_list[0]).nonzero()
    # Record the bottom of slice of this level
    slice_min_bottom = min(Z_bottom_level)

    # Extract the X, Y, Z positions of voxels belonging to the last vertebral level
    X_top_level, Y_top_level, Z_top_level = (data_vert_labeling==vert_levels_list[1]).nonzero()
    # Record the top slice of this level
    slice_max_top = max(Z_top_level)

    # Take into account the case where the ordering of the slice is reversed compared to the ordering of the vertebral level
    if slice_min_bottom > slice_max_top:
        slice_min = min(Z_top_level)
        slice_max = max(Z_bottom_level)
    else:
        slice_min = min(Z_bottom_level)
        slice_max = max(Z_top_level)

    # Return the slice numbers in the right format ("-1" because the function "remove_slices", which runs next, add 1 to the top slice
    return str(slice_min)+':'+str(slice_max)

#=======================================================================================================================
# Crop data to only keep the slices asked by user
#=======================================================================================================================
def remove_slices(data_to_crop, slices_of_interest):

    # extract slice numbers
    slices_list = [int(x) for x in slices_of_interest.split(':')] # 2-element list

    # Remove slices that are not wanted (+1 is to include the last selected slice as Python "includes -1"
    data_cropped = data_to_crop[..., slices_list[0]:slices_list[1]+1]

    return data_cropped


#=======================================================================================================================
# Save in txt file
#=======================================================================================================================
def save_metrics(ind_labels, label_name, slices_of_interest, vertebral_levels, metric_mean, metric_std, fname_output):
    print '\nWrite results in ' + fname_output + '...'

    # Write mode of file
    fid_metric = open(fname_output, 'w')

    # Write selected vertebral levels
    if vertebral_levels != '':
        fid_metric.write('%s\t%s to %s\n\n'% ('Vertebral levels : ', vertebral_levels.split(':')[0], vertebral_levels.split(':')[1]))
    else:
        fid_metric.write('No vertebral level selected. Considered all vertebral levels.\n\n')

    # # Write selected slices
    if slices_of_interest != '':
        fid_metric.write('%s\t%s to %s\n\n'% ('Slices : ', slices_of_interest.split(':')[0], slices_of_interest.split(':')[1]))
    else:
        fid_metric.write('No particular slice selected. Considered all slices.\n\n')

    # Write header title in file .txt
    fid_metric.write('%s,%s,%s\n' % ('Label', 'mean', 'std'))
    # Write metric for label chosen in file .txt
    for i in range(0, len(ind_labels)):
        fid_metric.write('%i,%s,%f,%f\n' % (ind_labels[i], label_name[ind_labels[i]], metric_mean[i], metric_std[i]))

    # Close file .txt
    fid_metric.close()


#=======================================================================================================================
# Check the consistency of the method asked by the user
#=======================================================================================================================
def check_method(method):
    if (method != 'wa') & (method != 'ml') & (method != 'bin'):
        print '\nERROR: Method "' + method + '" is not correct. See help. Exit program.\n'
        sys.exit(2)

#=======================================================================================================================
# Check the consistency of the labels asked by the user
#=======================================================================================================================
def check_labels(labels_of_interest, nb_labels):
    nb = ''
    # only specific labels are selected
    if not labels_of_interest == '':
        # Check if label chosen is in format : 0,1,2,..
        for char in labels_of_interest:
            if not char in '0123456789, ':
                print '\nERROR: "' + labels_of_interest + '" is not correct. Enter format "1,2,3,4,5,..". Exit program.\n'
                sys.exit(2)
        # Remove redundant values of label chosen and convert in integer
        nb = list(set([int(x) for x in labels_of_interest.split(",")]))
        # Check if label chosen correspond to a tract
        for num in nb:
            if not num in range(0, nb_labels):
                print '\nERROR: "' + str(num) + '" is not a correct tract label. Enter valid number. Exit program.\n'
                sys.exit(2)
    # all labels are selected
    else:
        nb = range(0, nb_labels)

    return nb



#=======================================================================================================================
# Extract metric within labels
#=======================================================================================================================
def extract_metric_within_tract(data, labels, method):

    # convert data to 1d
    data1d = data.ravel()

    # if there is only one tract, add dimension for compatibility of matrix manipulation
    if len(labels.shape) == 3:
        labels = labels[np.newaxis, :, :, :]

    # convert labels to 2d
    # TODO: pythonize this
    nb_labels = len(labels[:, 1, 1, 1])
    labels2d = np.empty([nb_labels, len(data1d)], dtype=object)
    for i_label in range(0, nb_labels):
        labels2d[i_label, :] = labels[i_label, :, :, :].ravel()

    # if user asks for binary regions, binarize atlas
    if method == 'bin':
        labels2d[labels2d < 0.5] = 0
        labels2d[labels2d >= 0.5] = 1

    #  Select non-zero values in the union of all labels
    labels2d_sum = np.sum(labels2d, axis=0)
    ind_nonzero = [i for i, v in enumerate(labels2d_sum) if v > ALMOST_ZERO]
    data1d = data1d[ind_nonzero]
    labels2d = labels2d[:, ind_nonzero]

    # Display number of non-zero values
    sct.printv('\nNumber of non-null voxels: '+str(len(ind_nonzero)), 1)

    # initialization
    metric_mean = np.empty([nb_labels, 1], dtype=object)
    metric_std = np.empty([nb_labels, 1], dtype=object)

    # Estimation with weighted average (also works for binary)
    if method == 'wa' or method == 'bin':
        for i_label in range(0, nb_labels):
            # check if all labels are equal to zero
            if sum(labels2d[i_label, :]) == 0:
                print 'WARNING: labels #'+str(i_label)+' contains only null voxels. Mean and std are set to 0.'
                metric_mean[i_label] = 0
                metric_std[i_label] = 0
            else:
                # estimate the weighted average
                metric_mean[i_label] = sum(data1d * labels2d[i_label, :]) / sum(labels2d[i_label, :])
                # estimate the biased weighted standard deviation
                metric_std[i_label] = np.sqrt( sum(labels2d[i_label, :] * (data1d - metric_mean[i_label])**2 ) / sum(labels2d[i_label, :]) )

    # Estimation with maximum likelihood
    if method == 'ml':
        y = data1d  # [nb_vox x 1]
        x = labels2d.T  # [nb_vox x nb_labels]
        beta = np.linalg.lstsq(np.dot(x.T, x), np.dot(x.T, y))
        for i_label in range(0, nb_labels):
            metric_mean[i_label] = beta[0][i_label]
            metric_std[i_label] = 0  # need to assign a value for writing output file

    return metric_mean, metric_std



#=======================================================================================================================
# Usage
#=======================================================================================================================
def usage():

    # read the .txt files referencing the labels by default
    default_info_label = open(param.path_label+'/'+param.file_info_label, 'r')
    label_references = default_info_label.read()

    # display help
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  This program exlabels metrics (e.g., DTI or MTR) within white matter labels. It requires an atlas,
  in the same space coordinates as the input image. The current methods for computing the average
  metrics are:
  - wa: weighted average (robust and accurate)
  - ml: maximum likelihood (best if >10 slices and low noise)
  - bin: binary masks (poorly accurate)
  The atlas is located in a folder and all labels are defined by .txt file. The label used by default is the template:

Label ID, label name, corresponding file name

"""+label_references+"""

USAGE
  """+os.path.basename(__file__)+""" -i <data> -t <path_label>

MANDATORY ARGUMENTS
  -i <volume>           file to extract metrics from

OPTIONAL ARGUMENTS
  -f <path_label>       path to the folder including labels to extract the metric from.
                        Default = """+param.path_label+"""
  -l <label_id>         Label number to extract the metric from. Default = all labels.
  -m <method>           ml (maximum likelihood), wa (weighted average), bin (binary)
  -a                    average all selected labels.
  -o <output>           File containing the results of metrics extraction.
                        Default = """+param.fname_output+"""
  -v <vert_level>       Vertebral levels to estimate the metric accross.
  -z <slice>            Slices to estimate the metric from. Begin at 0. Example: -z 3:6.
  -h                    help. Show this message

EXAMPLE
  """+os.path.basename(__file__)+""" -i t1.nii.gz\n"""

    #Exit program
    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    param = param()
    # call main function
    main()