#!/usr/bin/env python
#########################################################################################
#
# Extract metrics within spinal labels as defined by the white matter atlas and the
# template
# The folder atlas should have a .txt file that lists all tract files with labels.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Eddie Magnide, Simon Levy, Charles Naaman, Julien Cohen-Adad
# Modified: 2014-07-30
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: remove class color and use sct_printv
# TODO: add documentation for new features
# TODO (not urgent): vertebral levels selection should only consider voxels of the selected levels in slices where two different vertebral levels coexist (and not the whole slice)
# TODO: rename fname_vertebral_labeling

# Import common Python libraries
import os
import getopt
import sys
import time
import commands
import nibabel as nib
import numpy as np
import sct_utils as sct
from sct_orientation import get_orientation, set_orientation

# get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')


# constants
ALMOST_ZERO = 0.000001

class Param:
    def __init__(self):
        self.debug = 0
        self.method = 'wath'
        self.path_label = ''
        self.verbose = 1
        self.labels_of_interest = ''  # list. Example: '1,3,4'. . For all labels, leave empty.
        self.vertebral_levels = ''
        self.slices_of_interest = ''  # 2-element list corresponding to zmin:zmax. example: '5:8'. For all slices, leave empty.
        self.average_all_labels = 0  # average all labels together after concatenation
        self.fname_output = 'metric_label.txt'
        self.file_info_label = 'info_label.txt'
        self.fname_vertebral_labeling = 'MNI-Poly-AMU_level.nii.gz'
        self.threshold = 0.5  # threshold for the estimation methods 'wath' and 'bin'

class color:
    purple = '\033[95m'
    cyan = '\033[96m'
    darkcyan = '\033[36m'
    blue = '\033[94m'
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    bold = '\033[1m'
    underline = '\033[4m'
    end = '\033[0m'

class Color:
    def __init__(self):
        self.purple = '\033[95m'
        self.cyan = '\033[96m'
        self.darkcyan = '\033[36m'
        self.blue = '\033[94m'
        self.green = '\033[92m'
        self.yellow = '\033[93m'
        self.red = '\033[91m'
        self.bold = '\033[1m'
        self.underline = '\033[4m'
        self.end = '\033[0m'

#=======================================================================================================================
# main
#=======================================================================================================================
def main():
    # Initialization to defaults parameters
    fname_data = ''  # data is empty by default
    path_label = ''  # empty by default
    method = param.method # extraction mode by default
    labels_of_interest = param.labels_of_interest
    slices_of_interest = param.slices_of_interest
    vertebral_levels = param.vertebral_levels
    average_all_labels = param.average_all_labels
    fname_output = param.fname_output
    fname_vertebral_labeling = param.fname_vertebral_labeling
    fname_normalizing_label = ''  # optional then default is empty
    normalization_method = ''  # optional then default is empty
    actual_vert_levels = None  # variable used in case the vertebral levels asked by the user don't correspond exactly to the vertebral levels available in the metric data
    warning_vert_levels = None  # variable used to warn the user in case the vertebral levels he asked don't correspond exactly to the vertebral levels available in the metric data
    threshold = param.threshold  # default defined in class "param"
    start_time = time.time()  # save start time for duration
    verbose = param.verbose
    flag_h = 0

    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        fname_data = '/home/django/slevy/data/handedness_asymmetries/errsm_03/t2/t2_crop.nii.gz'  #path_sct+'/data/template/MNI-Poly-AMU_T2.nii.gz' #path_sct+'/testing/data/errsm_23/mt/mtr.nii.gz'
        path_label = '/home/django/slevy/data/handedness_asymmetries/errsm_03/t2/template/atlas'  #path_sct+'/data/atlas' #path_sct+'/testing/data/errsm_23/label/atlas'
        method = 'wa'
        labels_of_interest = '' #'0,1,2,21'  #'0, 2, 5, 7, 15, 22, 27, 29'
        slices_of_interest = '' #'102:134' #'117:127' #'2:4'
        vertebral_levels = '4:5'
        average_all_labels = 1
        fname_output = '/home/django/slevy/data/handedness_asymmetries/errsm_03/metric_extraction/left_averaged_estimations/atlas/t2'  #path_sct+'/testing/sct_extract_metric/results/quantif_mt_debug.txt'
        fname_normalizing_label = ''#'/home/django/slevy/data/handedness_asymmetries/errsm_03/t2/template/template/MNI-Poly-AMU_CSF.nii.gz'  #path_sct+'/data/template/MNI-Poly-AMU_CSF.nii.gz'
        normalization_method = '' #'whole'

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'haf:i:l:m:n:o:t:v:w:z:') # define flags
    except getopt.GetoptError as err: # check if the arguments are defined
        print str(err) # error
        usage() # display usage
    for opt, arg in opts: # explore flags
        if opt in '-a':
            average_all_labels = 1
        elif opt in '-f':
            path_label = os.path.abspath(arg)  # save path of labels folder
        elif opt == '-h':  # help option
            flag_h = 1
        elif opt in '-i':
            fname_data = arg
        elif opt in '-l':
            labels_of_interest = arg
        elif opt in '-m':  # method for metric extraction
            method = arg
        elif opt in '-n':  # filename of the label by which the user wants to normalize
            fname_normalizing_label = arg
        elif opt in '-o': # output option
            fname_output = arg  # fname of output file
        elif opt in '-t':  # threshold for the estimation methods 'wath' and 'bin' (see flag -m)
            threshold = float(arg)
        elif opt in '-v':
            # vertebral levels option, if the user wants to average the metric across specific vertebral levels
             vertebral_levels = arg
        elif opt in '-w':
            # method used for the normalization by the metric estimation into the normalizing label (see flag -n): 'sbs' for slice-by-slice or 'whole' for normalization after estimation in the whole labels
            normalization_method = arg
        elif opt in '-z':  # slices numbers option
            slices_of_interest = arg # save labels numbers

    # Display usage with tract parameters by default in case files aren't chosen in arguments inputs
    if fname_data == '' or path_label == '' or flag_h:
        param.path_label = path_label
        usage()

    # Check existence of data file
    sct.printv('\nCheck data files existence...', verbose)
    sct.check_file_exist(fname_data)
    sct.check_file_exist(path_label)
    if fname_normalizing_label:
        sct.check_file_exist(fname_normalizing_label)

    # add slash at the end
    path_label = sct.slash_at_the_end(path_label, 1)

    # Find path to the vertebral labeling file if vertebral levels were specified by the user
    if vertebral_levels:
        if slices_of_interest:  # impossible to select BOTH specific slices and specific vertebral levels
            print '\nERROR: You cannot select BOTH vertebral levels AND slice numbers.'
            usage()
        else:
            fname_vertebral_labeling_list = sct.find_file_within_folder(fname_vertebral_labeling, path_label + '..')
            if len(fname_vertebral_labeling_list) > 1:
                print color.red + 'ERROR: More than one file named \'' + fname_vertebral_labeling + ' were found in ' + path_label + '. Exit program.' + color.end
                sys.exit(2)
            elif len(fname_vertebral_labeling_list) == 0:
                print color.red + 'ERROR: No file named \'' + fname_vertebral_labeling + '\' were found in ' + path_label + '. Exit program.' + color.end
                sys.exit(2)
            else:
                fname_vertebral_labeling = os.path.abspath(fname_vertebral_labeling_list[0])

    # Check input parameters
    check_method(method, fname_normalizing_label, normalization_method)

    # Extract label info
    label_id, label_name, label_file = read_label_file(path_label)
    nb_labels_total = len(label_id)

    # check consistency of label input parameter.
    label_id_user = check_labels(labels_of_interest, nb_labels_total)  # If 'labels_of_interest' is empty, then
    # 'label_id_user' contains the index of all labels in the file info_label.txt

    # print parameters
    print '\nChecked parameters:'
    print '  data ...................... '+fname_data
    print '  folder label .............. '+path_label
    print '  selected labels ........... '+str(label_id_user)
    print '  estimation method ......... '+method
    print '  slices of interest ........ '+slices_of_interest
    print '  vertebral levels .......... '+vertebral_levels
    print '  vertebral labeling file.... '+fname_vertebral_labeling

    # Check if the orientation of the data is RPI
    orientation_data = get_orientation(fname_data)

    # we assume here that the orientation of the label files to extract the metric from is the same as the orientation
    # of the metric file (since the labels were registered to the metric)
    if orientation_data != 'RPI':
        sct.printv('\nCreate temporary folder to change the orientation of the NIFTI files into RPI...', verbose)
        path_tmp = sct.slash_at_the_end('tmp.'+time.strftime("%y%m%d%H%M%S"), 1)
        sct.create_folder(path_tmp)
        # change orientation and load data
        sct.printv('\nChange image orientation and load it...', verbose)
        data = nib.load(set_orientation(fname_data, 'RPI', path_tmp+'orient_data.nii')).get_data()
        # Do the same for labels
        sct.printv('\nChange labels orientation and load them...', verbose)
        labels = np.empty([nb_labels_total], dtype=object)  # labels(nb_labels_total, x, y, z)
        for i_label in range(0, nb_labels_total):
            labels[i_label] = nib.load(set_orientation(path_label+label_file[i_label], 'RPI', path_tmp+'orient_'+label_file[i_label])).get_data()
        if fname_normalizing_label:  # if the "normalization" option is wanted,
            normalizing_label = np.empty([1], dtype=object)  # choose this kind of structure so as to keep easily the
            # compatibility with the rest of the code (dimensions: (1, x, y, z))
            normalizing_label[0] = nib.load(set_orientation(fname_normalizing_label, 'RPI', path_tmp+'orient_normalizing_volume.nii')).get_data()
        if vertebral_levels:  # if vertebral levels were selected,
            data_vertebral_labeling = nib.load(set_orientation(fname_vertebral_labeling, 'RPI', path_tmp+'orient_vertebral_labeling.nii.gz')).get_data()

        # Remove the temporary folder used to change the NIFTI files orientation into RPI
        sct.printv('\nRemove the temporary folder...', verbose)
        status, output = commands.getstatusoutput('rm -rf ' + path_tmp)
    else:
        # Load image
        sct.printv('\nLoad image...', verbose)
        data = nib.load(fname_data).get_data()

        # Load labels
        sct.printv('\nLoad labels...', verbose)
        labels = np.empty([nb_labels_total], dtype=object)  # labels(nb_labels_total, x, y, z)
        for i_label in range(0, nb_labels_total):
            labels[i_label] = nib.load(path_label+label_file[i_label]).get_data()
        if fname_normalizing_label:  # if the "normalization" option is wanted,
            normalizing_label = np.empty([1], dtype=object)  # choose this kind of structure so as to keep easily the
            # compatibility with the rest of the code (dimensions: (1, x, y, z))
            normalizing_label[0] = nib.load(fname_normalizing_label).get_data()  # load the data of the normalizing label
        if vertebral_levels:  # if vertebral levels were selected,
            data_vertebral_labeling = nib.load(fname_vertebral_labeling).get_data()

    # Change metric data type into floats for future manipulations (normalization)
    data = np.float64(data)

    # Get dimensions of data
    sct.printv('\nGet dimensions of data...', verbose)
    nx, ny, nz = data.shape
    sct.printv('  ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz), verbose)

    # Get dimensions of labels
    sct.printv('\nGet dimensions of label...', verbose)
    nx_atlas, ny_atlas, nz_atlas = labels[0].shape
    sct.printv('.. '+str(nx_atlas)+' x '+str(ny_atlas)+' x '+str(nz_atlas)+' x '+str(nb_labels_total), verbose)

    # Check dimensions consistency between atlas and data
    if (nx, ny, nz) != (nx_atlas, ny_atlas, nz_atlas):
        print '\nERROR: Metric data and labels DO NOT HAVE SAME DIMENSIONS.'
        sys.exit(2)

    # Update the flag "slices_of_interest" according to the vertebral levels selected by user (if it's the case)
    if vertebral_levels:
        slices_of_interest, actual_vert_levels, warning_vert_levels = \
            get_slices_matching_with_vertebral_levels(data, vertebral_levels, data_vertebral_labeling)

    # select slice of interest by cropping data and labels
    if slices_of_interest:
        data = remove_slices(data, slices_of_interest)
        for i_label in range(0, nb_labels_total):
            labels[i_label] = remove_slices(labels[i_label], slices_of_interest)
        if fname_normalizing_label:  # if the "normalization" option was selected,
            normalizing_label[0] = remove_slices(normalizing_label[0], slices_of_interest)

    # if user wants to get unique value across labels, then combine all labels together
    if average_all_labels == 1:
        sum_labels_user = np.sum(labels[label_id_user]) # sum the labels selected by user
        if method == 'ml':  # in case the maximum likelihood and the average across different labels are wanted
            # TODO: make the below code cleaner (no use of tmp variable)
            labels_tmp = np.empty([nb_labels_total - len(label_id_user) + 1], dtype=object)
            labels = np.delete(labels, label_id_user)  # remove the labels selected by user
            labels_tmp[0] = sum_labels_user # put the sum of the labels selected by user in first position of the tmp
            # variable
            for i_label in range(1, len(labels_tmp)):
                labels_tmp[i_label] = labels[i_label-1]  # fill the temporary array with the values of the non-selected
                # labels
            labels = labels_tmp  # replace the initial labels value by the updated ones (with the summed labels)
            del labels_tmp  # delete the temporary labels
        else:  # in other cases than the maximum likelihood, we don't need to keep the other labels than those that were
        # selected
            labels = np.empty(1, dtype=object)
            labels[0] = sum_labels_user  # we create a new label array that includes only the summed labels

    if fname_normalizing_label:  # if the "normalization" option is wanted
        sct.printv('\nExtract normalization values...', verbose)
        if normalization_method == 'sbs':  # case: the user wants to normalize slice-by-slice
            for z in range(0, data.shape[-1]):
                normalizing_label_slice = np.empty([1], dtype=object)  # in order to keep compatibility with the function
                # 'extract_metric_within_tract', define a new array for the slice z of the normalizing labels
                normalizing_label_slice[0] = normalizing_label[0][..., z]
                metric_normalizing_label = extract_metric_within_tract(data[..., z], normalizing_label_slice, method, 0, threshold)
                # estimate the metric mean in the normalizing label for the slice z
                if metric_normalizing_label[0][0] != 0:
                    data[..., z] = data[..., z]/metric_normalizing_label[0][0]  # divide all the slice z by this value

        elif normalization_method == 'whole':  # case: the user wants to normalize after estimations in the whole labels
            metric_mean_norm_label, metric_std_norm_label = extract_metric_within_tract(data, normalizing_label, method, param.verbose, threshold)  # mean and std are lists


    # extract metrics within labels
    sct.printv('\nExtract metric within labels...', verbose)
    metric_mean, metric_std = extract_metric_within_tract(data, labels, method, verbose, threshold)  # mean and std are lists

    if fname_normalizing_label and normalization_method == 'whole':  # case: user wants to normalize after estimations in the whole labels
        metric_mean, metric_std = np.divide(metric_mean, metric_mean_norm_label), np.divide(metric_std, metric_std_norm_label)

    # update label name if average
    if average_all_labels == 1:
        label_name[0] = 'AVERAGED'+' -'.join(label_name[i] for i in label_id_user)  # concatenate the names of the
        # labels selected by the user if the average tag was asked
        label_id_user = [0]  # update "label_id_user" to select the "averaged" label (which is in first position)

    metric_mean = metric_mean[label_id_user]
    metric_std = metric_std[label_id_user]

    # display metrics
    print color.bold + '\nEstimation results:'
    for i in range(0, metric_mean.size):
        print color.bold+str(label_id_user[i])+', '+str(label_name[label_id_user[i]])+':    '+str(metric_mean[i])\
              +' +/- '+str(metric_std[i])+color.end

    # save and display metrics
    save_metrics(label_id_user, label_name, slices_of_interest, metric_mean, metric_std, fname_output, fname_data,
                 method, fname_normalizing_label, actual_vert_levels, warning_vert_levels)

    # Print elapsed time
    print '\nElapsed time : ' + str(int(round(time.time() - start_time))) + 's\n'


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
# Return the slices of the input image corresponding to the vertebral levels given as argument
#=======================================================================================================================
def get_slices_matching_with_vertebral_levels(metric_data, vertebral_levels, data_vertebral_labeling):

    sct.printv('\nFind slices corresponding to vertebral levels...', param.verbose)

    # Convert the selected vertebral levels chosen into a 2-element list [start_level end_level]
    vert_levels_list = [int(x) for x in vertebral_levels.split(':')]

    # If only one vertebral level was selected (n), consider as n:n
    if len(vert_levels_list) == 1:
        vert_levels_list = [vert_levels_list[0], vert_levels_list[0]]

    # Check if there are only two values [start_level, end_level] and if the end level is higher than the start level
    if (len(vert_levels_list) > 2) or (vert_levels_list[0] > vert_levels_list[1]):
        print '\nERROR:  "' + vertebral_levels + '" is not correct. Enter format "1:4". Exit program.\n'
        sys.exit(2)

    # Extract the vertebral levels available in the metric image
    vertebral_levels_available = np.array(list(set(data_vertebral_labeling[data_vertebral_labeling > 0])))

    # Check if the vertebral levels selected are available
    warning=[]  # list of strings gathering the potential following warning(s) to be written in the output .txt file
    min_vert_level_available = min(vertebral_levels_available)  # lowest vertebral level available
    max_vert_level_available = max(vertebral_levels_available)  # highest vertebral level available
    if vert_levels_list[0] < min_vert_level_available:
        vert_levels_list[0] = min_vert_level_available
        warning.append('WARNING: the bottom vertebral level you selected is lower to the lowest level available --> '
                       'Selected the lowest vertebral level available: ' + str(int(vert_levels_list[0])))  # record the
                       # warning to write it later in the .txt output file
        print color.yellow + 'WARNING: the bottom vertebral level you selected is lower to the lowest ' \
                                          'level available \n--> Selected the lowest vertebral level available: '+\
              str(int(vert_levels_list[0])) + color.end

    if vert_levels_list[0] > max_vert_level_available:
        vert_levels_list[1] = max_vert_level_available
        warning.append('WARNING: the top vertebral level you selected is higher to the highest level available --> '
                       'Selected the highest vertebral level available: ' + str(int(vert_levels_list[1])))  # record the
        # warning to write it later in the .txt output file

        print color.yellow + 'WARNING: the top vertebral level you selected is higher to the highest ' \
                                          'level available --> Selected the highest vertebral level available: ' + \
              str(int(vert_levels_list[1])) + color.end

    if vert_levels_list[0] not in vertebral_levels_available:
        distance = vertebral_levels_available - vert_levels_list[0]  # relative distance
        distance_min_among_negative_value = min(abs(distance[distance < 0]))  # minimal distance among the negative
        # relative distances
        vert_levels_list[0] = vertebral_levels_available[distance == distance_min_among_negative_value]  # element
        # of the initial list corresponding to this minimal distance
        warning.append('WARNING: the bottom vertebral level you selected is not available --> Selected the nearest '
                       'inferior level available: ' + str(int(vert_levels_list[0])))
        print color.yellow + 'WARNING: the bottom vertebral level you selected is not available \n--> Selected the ' \
                             'nearest inferior level available: '+str(int(vert_levels_list[0]))  # record the
        # warning to write it later in the .txt output file

    if vert_levels_list[1] not in vertebral_levels_available:
        distance = vertebral_levels_available - vert_levels_list[1]  # relative distance
        distance_min_among_positive_value = min(abs(distance[distance > 0]))  # minimal distance among the negative
        # relative distances
        vert_levels_list[1] = vertebral_levels_available[distance == distance_min_among_positive_value]  # element
        # of the initial list corresponding to this minimal distance
        warning.append('WARNING: the top vertebral level you selected is not available --> Selected the nearest superior'
                       ' level available: ' + str(int(vert_levels_list[1])))  # record the warning to write it later in the .txt output file

        print color.yellow + 'WARNING: the top vertebral level you selected is not available \n--> Selected the ' \
                             'nearest superior level available: ' + str(int(vert_levels_list[1]))

    # Extract metric data size X, Y, Z
    [mx, my, mz] = metric_data.shape
    # Extract vertebral labeling data size X, Y, Z
    [vx, vy, vz] = data_vertebral_labeling.shape

    sct.printv('  Check consistency of data size...', param.verbose)

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

    # Exit program if an error was detected
    if exit_program == 1:
        print '\nExit program.\n'
        sys.exit(2)
    else:
        print '    OK!'

    sct.printv('  Find slices corresponding to vertebral levels...', param.verbose)
    # Extract the X, Y, Z positions of voxels belonging to the first vertebral level
    X_bottom_level, Y_bottom_level, Z_bottom_level = (data_vertebral_labeling == vert_levels_list[0]).nonzero()
    # Record the bottom and top slices of this level
    slice_min_bottom = min(Z_bottom_level)
    slice_max_bottom = max(Z_bottom_level)

    # Extract the X, Y, Z positions of voxels belonging to the last vertebral level
    X_top_level, Y_top_level, Z_top_level = (data_vertebral_labeling == vert_levels_list[1]).nonzero()
    # Record the bottom and top slices of this level
    slice_min_top = min(Z_top_level)
    slice_max_top = max(Z_top_level)

    # Take into account the case where the ordering of the slice is reversed compared to the ordering of the vertebral
    # levels (usually the case) and if several slices include two different vertebral levels
    if slice_min_bottom >= slice_min_top or slice_max_bottom >= slice_max_top:
        slice_min = slice_min_top
        slice_max = slice_max_bottom
    else:
        slice_min = slice_min_bottom
        slice_max = slice_max_top

    # display info
    sct.printv('    '+str(slice_min)+':'+str(slice_max), param.verbose)

    # Return the slice numbers in the right format ("-1" because the function "remove_slices", which runs next, add 1
    # to the top slice
    return str(slice_min)+':'+str(slice_max), vert_levels_list, warning


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
def save_metrics(ind_labels, label_name, slices_of_interest, metric_mean, metric_std, fname_output, fname_data, method, fname_normalizing_label, actual_vert=None, warning_vert_levels=None):

    # CSV format, header lines start with "#"

    # Save metric in a .txt file
    print '\nWrite results in ' + fname_output + '...'

    # Write mode of file
    fid_metric = open(fname_output, 'w')

    # WRITE HEADER:
    # Write date and time
    fid_metric.write('# Date - Time: '+ time.strftime('%Y/%m/%d - %H:%M:%S'))
    # Write metric data file path
    fid_metric.write('\n'+'# Metric file: '+ os.path.abspath(fname_data))
    # If it's the case, write the label used to normalize the metric estimation:
    if fname_normalizing_label:
        fid_metric.write('\n' + '# Label used to normalize the metric estimation slice-by-slice: ' +
                         fname_normalizing_label)
    # Write method used for the metric estimation
    fid_metric.write('\n'+'# Extraction method: '+method)

    # Write selected vertebral levels
    if actual_vert:
        if warning_vert_levels:
            for i in range(0, len(warning_vert_levels)):
                fid_metric.write('\n# '+str(warning_vert_levels[i]))
        fid_metric.write('\n# Vertebral levels: '+'%s to %s' % (int(actual_vert[0]), int(actual_vert[1])))
    else:
        fid_metric.write('\n# Vertebral levels: ALL')

    # Write selected slices
    fid_metric.write('\n'+'# Slices (z): ')
    if slices_of_interest != '':
        fid_metric.write('%s to %s' % (slices_of_interest.split(':')[0], slices_of_interest.split(':')[1]))
    else:
        fid_metric.write('ALL')

    # label info
    fid_metric.write('%s' % ('\n'+'# ID, label name, mean, std\n\n'))

    # WRITE RESULTS

    # Write metric for label chosen in file .txt
    for i in range(0, len(ind_labels)):
        fid_metric.write('%i, %s, %f, %f\n' % (ind_labels[i], label_name[ind_labels[i]], metric_mean[i], metric_std[i]))

    # Close file .txt
    fid_metric.close()



#=======================================================================================================================
# Check the consistency of the methods asked by the user
#=======================================================================================================================
def check_method(method, fname_normalizing_label, normalization_method):
    if (method != 'wa') & (method != 'ml') & (method != 'bin') & (method != 'wath'):
        print '\nERROR: Method "' + method + '" is not correct. See help. Exit program.\n'
        sys.exit(2)

    if normalization_method and not fname_normalizing_label:
        print color.red + '\nERROR: You selected a normalization method (' + color.bold + str(normalization_method) \
              + color.end + color.red + ') but you didn\'t selected any label to be used for the normalization.' \
              + color.end
        usage()

    if fname_normalizing_label and normalization_method != 'sbs' and normalization_method != 'whole':
        print color.red + '\nERROR: The normalization method you selected is incorrect:' + color.bold + \
              str(normalization_method) + color.end
        usage()


#=======================================================================================================================
# Check the consistency of the labels asked by the user
#=======================================================================================================================
def check_labels(labels_of_interest, nb_labels):

    # by default, all labels are selected
    list_label_id = range(0, nb_labels)

    # only specific labels are selected
    if labels_of_interest != '':
        # Check if label chosen is in format : 0,1,2,..
        for char in labels_of_interest:
            if not char in '0123456789, ':
                print '\nERROR: "' + labels_of_interest + '" is not correct. Enter format "1,2,3,4,5,..". Exit program.\n'
                sys.exit(2)

        # Remove redundant values of label chosen and convert in integer
        list_label_id = list(set([int(x) for x in labels_of_interest.split(",")]))
        list_label_id.sort()

        # Check if label chosen correspond to a label
        for num in list_label_id:
            if not num in range(0, nb_labels):
                print '\nERROR: "' + str(num) + '" is not a correct label. Enter valid number. Exit program.\n'
                sys.exit(2)

    return list_label_id


#=======================================================================================================================
# Extract metric within labels
#=======================================================================================================================
def extract_metric_within_tract(data, labels, method, verbose, threshold=0.5):

    nb_labels = len(labels) # number of labels

    # if user asks for binary regions, binarize atlas
    if method == 'bin':
        for i in range(0, nb_labels):
            labels[i][labels[i] < threshold] = 0
            labels[i][labels[i] >= threshold] = 1

    # if user asks for thresholded weighted-average, threshold atlas
    if method == 'wath':
        for i in range(0, nb_labels):
            labels[i][labels[i] < threshold] = 0

    #  Select non-zero values in the union of all labels
    labels_sum = np.sum(labels)
    ind_positive_labels = labels_sum > ALMOST_ZERO
    ind_positive_data = data > 0
    ind_positive = ind_positive_labels & ind_positive_data
    data1d = data[ind_positive]
    labels2d = np.empty([nb_labels, len(data1d)], dtype=float)
    for i in range(0, nb_labels):
        labels2d[i] = labels[i][ind_positive]

    # clear memory
    del data, labels

    # Display number of non-zero values
    sct.printv('  Number of non-null voxels: '+str(len(data1d)), verbose=verbose)

    # initialization
    metric_mean = np.empty([nb_labels], dtype=object)
    metric_std = np.empty([nb_labels], dtype=object)

    # Estimation with weighted average (also works for binary)
    if method == 'wa' or method == 'bin' or method == 'wath':
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
                metric_std[i_label] = np.sqrt(sum(labels2d[i_label, :] * (data1d - metric_mean[i_label])**2 ) /
                                               sum(labels2d[i_label, :]))

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

    # read the .txt files referencing the labels
    if param.path_label != '':
        file_label = param.path_label+'/'+param.file_info_label
        sct.check_file_exist(file_label, 0)
        default_info_label = open(file_label, 'r')
        label_references = default_info_label.read()
    else:
        label_references = ''

    # display help
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  This program extracts metrics (e.g., DTI or MTR) within labels. The labels are generated with
  'sct_warp_template'. The label folder contains a file (info_label.txt) that describes all labels.
  The labels should be in the same space coordinates as the input image.

USAGE
  """+os.path.basename(__file__)+""" -i <data> -f <folder_label>

MANDATORY ARGUMENTS
  -i <data>             file to extract metrics from
  -f <folder_label>     folder including labels to extract the metric from.

OPTIONAL ARGUMENTS
  -l <label_id>         Label number to extract the metric from. Example: 1,3 for left fasciculus
                        cuneatus and left ventral spinocerebellar tract in folder '/atlas'.
                        Default = all labels.
  -m {ml,wa,wath,bin}   method to extract metrics. Default = """+param.method+"""
                          ml: maximum likelihood (only use with well-defined regions and low noise)
                          wa: weighted average
                          wath: weighted average (only consider values >0.5)
                          bin: binarize mask (threshold=0.5)
  -a                    average all selected labels.
  -o <output>           File containing the results of metrics extraction.
                        Default = """+param.fname_output+"""
  -v <vmin:vmax>        Vertebral levels to estimate the metric across. Example: 2:9 for C2 to T2.
  -z <zmin:zmax>        Slice range to estimate the metric from. First slice is 0. Example: 5:23
                        You can also select specific slices using commas. Example: 0,2,3,5,12
  -h                    help. Show this message

EXAMPLE
  To list template labels:
    """+os.path.basename(__file__)+""" -f $SCT_DIR/data/template

  To list white matter atlas labels:
    """+os.path.basename(__file__)+""" -f $SCT_DIR/data/atlas

  To compute FA within labels 0, 2 and 3 within vertebral levels C2 to C7 using binary method:
    """+os.path.basename(__file__)+""" -i dti_FA.nii.gz -f label/atlas -l 0,2,3 -v 2:7 -m bin"""

    # display list of labels
    if label_references != '':
        print """
List of labels in: """+file_label+""":
==========
"""+label_references+"""
=========="""
    print

    #Exit program
    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    param = Param()
    color = Color()
    # call main function
    main()