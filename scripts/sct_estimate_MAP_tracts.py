#!/usr/bin/env python

## @package sct_estimate_MAP_tracts
#
# - from spinal cord MRI volume 3D (as niftii format) and atlas of white matter tracts, estimate metrics (FA, MTR,...)
# of each tract
#
# Description about how the function works:
#
# 1. pretreatment
# ----------------------------------------------------------------------------------------------------------------------
# Inputs
# - data_start : data array of metrics
# - tracts_start : tracts arrays of partial volumes
# - nb_slice : slices start and end
#
# Outputs
# - data_adjust : data array of metrics adjusted
# - tracts_adjust : tracts arrays of partial volumes adjusted
# - numtracts : total number of tracts selected
#
# Description
# This function checks if the volume size metric MRI is the same as the volume of tracts. It then applies a mask to
# select only non-zero values in tracts. Finally, it select only slices chosen in tracts and data.
#
# 2. read_name
# ----------------------------------------------------------------------------------------------------------------------
# Inputs
# - fname_tracts : tracts files folder
#
# Outputs
# - label_title : title of labels
# - label_name : file names of each tracts
# - label_num : number associated with each tracts
#
# Description
# This function checks if there are errors in read file .nii.gz of each tracts and file .txt associated with these
# tracts. It extract files
#
# 3. weighted_average
# ----------------------------------------------------------------------------------------------------------------------
# Inputs
# - data_wa : data array of metrics
# - tracts_wa : cell array containing partial volumes of each tracts
#
# Outputs
# - X_wa : metric values estimation for each tract
# - std_wa : standard deviations for each tract
#
# Description
# This function make estimation of each tract metric with weighted average and compute the standard deviation in each
# tract.
#
# 4. estimate_parameters
# ----------------------------------------------------------------------------------------------------------------------
# Inputs
# - Y : data array of metrics resize to 1D
# - P : matrix of linear transformation (y=Px+n)
# - R_X : covariance normalised of metrics
# - U_comp*X0 : matrix of mean
#
# Outputs
# - sigmaN : standard deviation estimation of noise in metrics
# - sigmaX : standard deviation estimation in all values of metrics by tracts
#
# Description
# This function makes estimation of standard deviations (noise and x) for maximum a posteriori method by minimisation.
#
# 5. bayesian
# ----------------------------------------------------------------------------------------------------------------------
# Inputs
# - data : data array of metrics
# - sigmaN : standard deviation estimation of noise in metrics
# - sigmaX : standard deviation estimation in all values of metrics by tracts
# - tracts : cell array containing the white matter atlas
#
# Outputs
# - X_map : metric value estimation for each tract
# - std_map : standard deviation for each tract
#
# Description
# This function makes estimation of each tract metric with maximum a posteriori and compute the standard deviation in
# each tract.
#
# 6. MAP
# ----------------------------------------------------------------------------------------------------------------------
# Inputs
# - Y : data array of metrics resize to 1D
# - P : matrix of linear transformation (y=Px+n)
# - R_X : covariance normalised of metrics
# - U_comp*X0 : matrix of mean
# - sigmaX : standard deviation of metrics
# - sigmaN : standard deviation of noise
#
# Outputs
# - sigma_map : standard deviation estimation of noise after MAP
# - sigma_map : standard deviation estimation in metrics by tracts
# - X_map : metric value estimation for each tract
#
# Description
# This function computes maximum a posteriori method.
#
# DEPENDENCIES
# ----------------------------------------------------------------------------------------------------------------------
#
# EXTERNAL PYTHON PACKAGES
# - nibabel : <http://nipy.sourceforge.net/nibabel/>
# - numpy : <http://www.numpy.org>
# - scipy : <www.scipy.org>
#
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2014 NeuroPoly, Polytechnique Montreal <www.neuropoly.info>
# Authors: Julien Cohen-Adad, Eddie Magnide
#
# License: see the LICENSE.TXT
# ======================================================================================================================



# Initialisation of default parameters

class param:
    def __init__(self):
        # debugger parameter of test program
        self.debug = 0
        # iterations number for estimation of standard deviations
        self.iter = 100
        # extraction mode by default is weighted average
        self.mode = "weightedaverage"
        # tract folder by default is atlas white matter of spinalcordtoolbox_dev project
        self.tract = '../data/atlas'
        # by default, labels choice is deactivated and program use all labels
        self.label_choice = 0
        # by defaults, the estimation is made accross all vertebral levels
        self.vertebral_levels = ''
        # by default, slices choice is desactivated and program use all slices
        self.slice_choice = 0
        # by default, program don't export data results in file .txt
        self.output_choice = 0

# Import common Python libraries
import os
import getopt
import sys
import time
import glob
import re
import sct_utils as sct
import numpy

# Check if special Python libraries are installed or not
try:
    # library of nifty format for read data and tracts files
    from nibabel import load
except ImportError:
    print '--- nibabel not installed! Exit program. ---'
    sys.exit(2)
try:
    # library of calculations and processing matrices
    from numpy import mean, asarray, std, zeros, sum, ones, dot, eye, sqrt, empty, size, linspace, abs, amin, argmin, concatenate, array
    from numpy.linalg import solve,pinv
except ImportError:
    print '--- numpy not installed! Exit program. ---'
    sys.exit(2)
try:
    # library of processing imaging
    from scipy.ndimage.filters import gaussian_filter
except ImportError:
    print '--- scipy not installed! Exit program. ---'
    sys.exit(2)

#=======================================================================================================================
# main
#=======================================================================================================================

def main():

    # Extract path of the script
    path_script = os.path.dirname(__file__) + '/'

    # Initialization to defaults parameters
    fname_data = '' # data is empty by default
    fname_tracts = path_script + param.tract # tracts path by default
    mode = param.mode # extraction mode by default
    label_choice = param.label_choice # no select label by default
    vertebral_levels = param.vertebral_levels # no vertebral level selected by default
    slice_choice = param.slice_choice # no select label by default
    output_choice = param.output_choice # no select slice by default
    start_time = time.time() # save start time for duration

    # Parameters for debug mode
    if param.debug == 1:
        print '\n** WARNING: DEBUG MODE ON **'
        fname_data = '/home/django/emagnide/code/spinalcordtoolbox_dev/testing/data/errsm_24/mt/mtr.nii.gz'
        fname_tracts = '/home/django/emagnide/code/spinalcordtoolbox_dev/testing/data/errsm_24/atlas'
        mode = param.mode
        label_choice = 1
        label_number = '2,6'
        output_choice = 1
        slice_choice = 1
        slice_number = '1'
        fname_output = '../result.txt'

    # Extract title, tract names and label numbers because label names are used by the function "usage"
    [label_title, label_name, label_num] = read_name(fname_tracts)[:-1]

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hi:l:m:o:t:v:z:') # define flags
    except getopt.GetoptError as err: # check if the arguments are defined
        print str(err) # error
        usage(label_title, label_name, label_num,fname_tracts) # display usage
    for opt, arg in opts: # explore flags
        if opt == '-h': # help option
            usage(label_title, label_name, label_num,fname_tracts) # display usage
        elif opt in '-i': # MRI metric to input
            fname_data = arg # save path of metric MRI
        elif opt in '-l': # labels numbers option
            label_choice = 1 # label choice is activate
            label_number = arg # save labels numbers
        elif opt in '-m': # extraction mode option
            mode = arg # save extraction mode
        elif opt in '-o': # output option
            output_choice = 1 # output choice is activate
            fname_output = os.path.abspath(arg) # save path of output
        elif opt in '-t': # tracts to input
            fname_tracts = os.path.abspath(arg) # save path of tracts folder
        elif opt in '-v': # vertebral levels option, if the user wants to average the metric accross specific vertebral levels
            vertebral_levels = arg
        elif opt in '-z': # slices numbers option
            slice_choice = 1 # slice choice is activate
            slice_number = arg # save labels numbers
        else: # verify that all entries are correct
            print('\nERROR: Option {} unknown. Exit program.\n'.format(opt)) # error
            sys.exit(2) # exit programme

    #TODO: check if the case where the input images are not in AIL orientation is taken into account (if not, implement it)

    # Display usage with tract parameters by default in case files aren't chosen in arguments inputs
    if fname_data == '' or fname_tracts == '':
        usage(label_title, label_name, label_num,fname_tracts)

    # Check existence of data file
    sct.check_file_exist(fname_data)

    # Extract path/file/extension
    path_data, file_data, ext_data = sct.extract_fname(fname_data)

    # Add extensions file if there are not
    if ext_data == '':
        fname_data += '.nii.gz'
        path_data, file_data, ext_data = sct.extract_fname(fname_data)

    # Check if data extension is correct
    if ext_data != '.nii.gz':
        print '\nERROR: Data format ' + ext_data + ' not correct, use ".nii.gz". Exit program.\n'
        sys.exit(2)

    # Extract title, tract names and label numbers
    [label_title, label_name, label_num, fname_tract] = read_name(fname_tracts)

    # Check if mode is correct : "weightedaverage" or "bayesian"
    if (mode != "weightedaverage") & (mode != "bayesian"):
        print '\nERROR: Mode "' + mode + '" is not correct. Enter "weightedaverage" or "bayesian". Exit program.\n'
        sys.exit(2)

    # Extract label chosen
    if label_choice == 1:

        # Check if label chosen is in format : 0,1,2,..
        for char in label_number:
            if not char in '0123456789, ':
                print '\nERROR: "' + label_number + '" is not correct. Enter format "1,2,3,4,5,..". Exit program.\n'
                sys.exit(2)

        # Remove redundant values of label chosen and convert in integer
        nb = list(set([int(x) for x in label_number.split(",")]))

        # Check if label chosen correspond to a tract
        for num in nb:
            if not num in range(0, len(fname_tract)):
                print '\nERROR: "' + str(num) + '" is not a correct tract label. Enter valid number. Exit program.\n'
                sys.exit(2)

    # Consider all tracts by default whereas in their names
    if label_choice == 0:
        nb = range(0, len(fname_tract))

    # Read files
    print '\nRead files...'

    # Load data metric MRI
    data = load(fname_data)

    # Make data in array format
    data = data.get_data()


    # Select the input image slices corresponding to the selected vertebral levels
    if vertebral_levels != '':
        if slice_choice == 1:
            print '\nERROR: You cannot select BOTH vertebral levels AND slice numbers.'
            sys.exit(2)
        else:
            # Convert the selected vertebral levels chosen into a vector of two elements [start_level end_level]
            vert_levels_list = [int(x) for x in vertebral_levels.split(':')]

            # If only one vertebral level was selected (n), consider as n:n
            if len(vert_levels_list) == 1:
                vert_levels_list = [vert_levels_list[0], vert_levels_list[0]]

            # Check if there are only two values [start_level, end_level] and if the end level is higher than the start level
            if (len(vert_levels_list) > 2) or (vert_levels_list[0] > vert_levels_list[1]):
                print '\nERROR: "' + vertebral_levels + '" is not correct. Enter format "1:4". Exit program.\n'
                sys.exit(2)

            # Select the slices of the input image corresponding to the vertebral levels asked
            slice_choice = 1
            slice_number = get_slices_matching_with_vertebral_levels(data,fname_tracts,vert_levels_list)


    # Extract slices chosen
    if slice_choice == 1:

        # Check if slices chosen is in format : "0:5"
        for char in slice_number:
            if not char in '0123456789:':
                print '\nERROR: "' + slice_number + '" is not correct. Enter format "0:5". Exit program.\n'
                sys.exit(2)

        # Convert slices chosen in integer
        nb_slice = [int(x) for x in slice_number.split(':')]

        # If slice chosen is single (n), program consider as n:n
        if len(nb_slice) == 1:
            nb_slice = [nb_slice[0], nb_slice[0]]

        # Check if there are only two values (start, end) and if end number is more than start number
        if (len(nb_slice) > 2) or (nb_slice[0] > nb_slice[1]):
            print '\nERROR: "' + slice_number + '" is not correct. Enter format "0:5". Exit program.\n'
            sys.exit(2)

    # Verify file output
    if output_choice == 1:

        # Check if repertory of file output exist
        if not (os.path.isdir((os.path.split(fname_output))[0])):
            print('\nERROR: Folder output ' + (os.path.split(fname_output))[0] + ' does not exist. Exit program.\n')
            sys.exit(2)

    # Display arguments
    print '\nCheck input arguments...'

    # Display data file
    print '\tSpinal cord MRI : ' + fname_data

    # Display mode extraction
    print '\tExtraction mode : ' + mode

    # Display tracts path
    print '\tTracts atlas : ' + fname_tracts

    # Display vertebral levels where the metric will be estimated
    if vertebral_levels != '':
        print '\tVertebral levels selected : ' + (str(vert_levels_list)[1:-1]).replace(',', ':').replace(' ', '')
    else:
        print '\tNo vertebral level selected.'

    # Display slices where the metric sill be estimated
    if slice_choice ==1:
        print '\tSlices : ' + (str(nb_slice)[1:-1]).replace(',', ':').replace(' ', '')

    # Display output file for results
    if output_choice == 1:
        print '\tOutput : ' + fname_output

    # Display labels chosen for results
    if label_choice == 1:
        print '\tSelected tracts numbers : ' + (str(nb)[1:-1]).replace(' ', '')

    # Display label number, label name and file .nii.gz corresponding
    print '\n\t'+label_title+'\n'
    for label in range(0,len(fname_tract)):
        print '\tLabel ' + str(label_num[label]) + ' \t\t' + fname_tract[label][(len(fname_tracts) + 1):]+ \
              '\t\t' + label_name[label]

    # Initialise tracts variable as object because there are 4 dimensions
    tracts = empty([len(fname_tract), 1], dtype=object)

    # Load each partial volumes of each tracts
    for label in range(0, len(fname_tract)):
        tracts[label, 0] = load(fname_tract[label]).get_data()

    # Reshape data if it is the 2D image instead of 3D
    if data.ndim == 2:
        data=data.reshape(int(size(data,0)), int(size(data,1)),1)

    # Reshape tracts if it is the 2D image instead of 3D
    for label in range(0, len(fname_tract)):
        if (tracts[label,0]).ndim == 2:
            tracts[label,0] = tracts[label,0].reshape(int(size(tracts[label,0],0)), int(size(tracts[label,0],1)),1)

    # Initialization slices if slice choice is off by considering the z size of tracts
    if slice_choice ==0:
        nb_slice = [0,int(size(tracts[0, 0],2)-1)]

    # Pretreatment before extraction
    [data_new,tracts_new, number_tracts] = pretreatment(data, tracts, nb_slice)

    #TODO: only estimate the metric value for selected tracts AND NOT: for all and then display the metric value for the selected tracts (what this script currently does)
    # Extraction with weighted_average
    if mode == "weightedaverage":

        # Do extraction with weighted average
        [X, stand] = weighted_average(data_new, tracts_new, number_tracts)
        print'\nWeighted average results \n'

        # Display results
        for i in range(0, len(nb)):
            print'\tLabel ' + str(nb[i]) + ' \tX = ' + str(X[nb[i], 0]) + ' \tSTD = ' + str(stand[nb[i], 0])

    # Extraction with bayesian model
    if mode == "bayesian":

        # Do extraction with maximum a posteriori method
        [X, stand] = bayesian(data_new, tracts_new, number_tracts)

        # Display results
        print'\nBayesian estimation results \n'
        for i in range(0, len(nb)):
            print'\tLabel ' + str(nb[i]) + ' \tX = ' + str(X[nb[i], 0]) + ' \tSTD = ' + str(stand[nb[i], 0])

    # Save data output in file .txt
    if output_choice == 1:
        print '\nWrite results in ' + fname_output + '...'

        # Write mode of file
        fid_metric = open(fname_output, 'w')

        # Write selected vertebral levels
        if vertebral_levels != '':
            fid_metric.write('%s\t%i to %i\n\n'% ('Vertebral levels : ',vert_levels_list[0],vert_levels_list[1]))
        else:
            fid_metric.write('No vertebral level selected.\n\n')

        # Write slices chosen
        fid_metric.write('%s\t%i to %i\n\n'% ('Slices : ',nb_slice[0],nb_slice[1]))

        # Write header title in file .txt
        fid_metric.write('%s\t\t%s\t\t\t\t\t\t%s\t\t\t\t%s\n\n' % ('Label', 'Name', 'Metric', 'STD'))

        # Write metric for label chosen in file .txt
        for i in range(0, len(nb)):
            fid_metric.write('%i\t%s\t\t\t%f\t\t\t%f\n' % (nb[i], label_name[nb[i]], X[nb[i]], stand[nb[i]]))

        # Close file .txt
        fid_metric.close()

        # Display success message
        print '\tExport successful'

    # Calculate elapsed time
    elapsed_time = int(round(time.time() - start_time))

    # Extract time in minutes and seconds
    sec = elapsed_time % 60
    mte = (elapsed_time - sec) / 60

    # Display time in seconds and minutes for more than one minute
    if mte != 0:
        print '\nFinished! Elapsed time: ' + str(mte) + 'min' + str(sec) + 's'

    # Display time in seconds for less than one minute
    else:
        print '\nFinished! Elapsed time: ' + str(sec) + 's'

#=======================================================================================================================
# Read file of tract names and extract names and labels
#=======================================================================================================================

def read_name(fname_tracts):

    # Check if tracts folder exist
    if not os.path.isdir(fname_tracts):
        print('\nERROR: ' + fname_tracts + ' does not exist. Exit program.\n')
        sys.exit(2)

    # Save path of each tracts
    fname_tract = glob.glob(fname_tracts + '/*.nii.gz')

    # Check if tracts exist in folder
    if len(fname_tract) == 0:
        print '\nERROR: There are not tracts in this folder. Exit program.\n'
        sys.exit(2)

    # Save path of file list.txt
    fname_list = glob.glob(fname_tracts + '/*.txt')

    # Check if tracts list.txt exist in folder
    if len(fname_list) == 0:
        print '\nWARNING: There are no file txt in this folder. File list.txt will be create in folder \n'

    # Check if tracts list.txt is only txt in folder
    if len(fname_list) > 1:
        print '\nWARNING: There are more than one file txt in this folder. File list.txt will be create in folder \n'

    # Create list.txt default in list.txt in case there are not file or file is not define correctly
    if len(fname_list) == 0 or len(fname_list) > 1:

        # New file list : list.txt
        fname_list = fname_tracts + '/list.txt'

        # Write mode
        fid_list = open(fname_list, 'w')

        # Write "Title : Name of tracts" by default
        fid_list.write('%s : %s\n' % ('Title', 'Name of tracts'))

        # Write "Label XX : Label XX" by default for "XX" tract number
        for j in range(0, len(fname_tract)):
            fid_list.write('%s %i : %s %i\n' % ('Label', j, 'Label', j))

        # Close file txt
        fid_list.close()

    # Take the value of string instead of array string
    else:
        fname_list = fname_list[0]

    # Read file list.txt
    f = open(fname_list)

    # Extract all lines in file.txt
    lines = [lines for lines in f.readlines() if lines.strip()]

    # Close file.txt
    f.close()

    # Check if file contain data
    if len(lines) == 0:
        print '\nWARNING: File txt is empty. File list.txt will be create in folder. \n'

    # Initialisation of label number
    label_num = [[]] * len(lines)

    # Initialisation of label name
    label_name = [[]] * len(lines)

    # Extract of label title, label name and label number
    for k in range(0, len(lines)):

        # Check if file.txt contains ":" because it is necessary for split label name to label number
        if not ':' in lines[k]:
            print '\nERROR: File txt is not in correct form. File list.txt must be in this form :\n'
            print '\t\tTitle : Name of tracts'
            print '\t\tLabel 0 : Name Label 0'
            print '\t\tLabel 1 : Name Label 1'
            print '\t\tLabel 2 : Name Label 2'
            print '\t\t...\n '
            print '\t\tExample of file.txt'
            print '\t\tTitle : List of tracts names for the white matter atlas'
            print '\t\tLabel 0 : left fasciculus gracilis'
            print '\t\tLabel 1 : left fasciculus cuneatus'
            print '\t\tLabel 2 : left lateral corticospinal tract'
            print '\nExit program. \n'
            sys.exit(2)

        # Split label name to label number without "['" (begin) and "']" (end) (so 2 to end-2)
        else:
            [label_num[k],label_name[k]] = lines[k].split(':')
            label_name[k] = str(label_name[k].splitlines())[2:-2]

    # Extract label title as the first line in file.txt
    label_title = label_name[0]

    # Extract label name from the following lines
    label_name = label_name[1:]

    # Extract label number from the following lines
    label_num = str(label_num[1:])
    label_num = [int(x.group()) for x in re.finditer(r'\d+',label_num)]

    # Check corresponding between label name and tract file
    if label_num != range(0, len(fname_tract)):
        print '\nERROR: File txt and tracts are not corresponding. Change file txt or tracts .nii.gz. Exit program. \n'
        sys.exit(2)

    return [label_title, label_name, label_num, fname_tract]

#=======================================================================================================================
# Pretreatment before extraction
#=======================================================================================================================

def pretreatment(data_start, tracts_start, nb_slice):

    # Size of tracts
    print '\nVerify tract size...'
    # Extract total number of tracts
    numtracts = len(tracts_start)

    # Initialisation of size axis X, Y, Z
    nx = zeros(numtracts)
    ny = zeros(numtracts)
    nz = zeros(numtracts)

    # Initialisation of check error flag
    exit_program = 0

    # Save size X, Y, Z of each tracts
    for i in range(0, numtracts):
        [nx[i], ny[i], nz[i]] = (tracts_start[i, 0]).shape

    # Convert list to integer
    nx = nx.astype(int)

    # Display the different sizes X of tracts
    print '\tSize X : ' + str(list(set(nx)))[1:-1]

    # Check if all sizes X of tracts is same and display error message
    if sum(nx) != nx[0] * numtracts:
        print '\tERROR: Size X is not the same for all tracts.'
        exit_program = 1

    # Convert list to integer
    ny = ny.astype(int)

    # Display the different sizes Y of tracts
    print '\tSize Y : ' + str(list(set(ny)))[1:-1]

    # Check if all sizes Y of tracts is same and display error message
    if sum(ny) != ny[0] * numtracts:
        print '\tERROR: Size Y is not the same for all tracts.'
        exit_program = 1

    # Convert list to integer
    nz = nz.astype(int)

    # Display the different sizes Z of tracts
    print '\tSize Z : ' + str(list(set(nz)))[1:-1]

    # Check if all sizes Z of tracts is same and display error message
    if sum(nz) != nz[0] * numtracts:
        print '\tERROR: Size Z is not the same for all tracts.'
        exit_program = 1

    # Take size in integer instead of integer array
    nx = nx[0]
    ny = ny[0]
    nz = nz[0]

    # Size of data
    print '\nVerify data size...'

    # Extract data size X, Y, Z
    [mx, my, mz] = data_start.shape

    # Display values of size data
    print '\tSize : ' + str(int(mx)) + 'x' + str(int(my)) + 'x' + str(int(mz))

    # Check if sizes X is same with tract
    if mx != nx:
        print '\tERROR: Size X is not the same for tracts and data.'
        exit_program = 1

    # Check if sizes Y is same with tract
    if my != ny:
        print '\tERROR: Size Y is not the same for tracts and data.'
        exit_program = 1

    # Check if sizes Z is same with tract
    if mz != nz:
        print '\tERROR: Size Z is not the same for tracts and data.'
        exit_program = 1

    # Exit program if error is detect in sizes
    if exit_program == 1 :
        print '\nExit program.\n'
        sys.exit(2)

    # Check if slices chosen are presents in tract slices
    for slice in nb_slice:
        if not slice in range(0,mz):
            print '\nERROR: Slice z = ' + str(slice) + ' does not exist. Exit program.\n'
            sys.exit(2)

    # Compute and apply binary mask
    print '\nSelection of tracts areas...'

    # Initialisation of mask
    mask = zeros([nx, ny, nz])

    # Select non-zero values in the tracts
    for i in range(0, numtracts):
        mask[(tracts_start[i, 0]) > 0] = 1
    data_adjust = mask * data_start

    # Display number of non-zero values
    print '\tThere are ' + str(int(sum(mask))) + \
          ' voxels that have non-zero values in the tracts.'

    # Select slices chosen in data
    data_adjust = data_adjust[:,:,nb_slice[0]:nb_slice[1]+1]

    # Initialisation of new tracts
    tracts_adjust = empty([numtracts,1], dtype=object)

    # Select slices chosen in tracts
    for i in range(0, numtracts):
        tracts_adjust[i,0] = tracts_start[i, 0][:,:,nb_slice[0]:nb_slice[1]+1]

    # Return data corrected and tracts number
    return [data_adjust,tracts_adjust, numtracts]

#=======================================================================================================================
# Weighted average
#=======================================================================================================================

def weighted_average(data_wa, tracts_wa, numtracts_wa):

    # Estimation with weighted average
    print '\nEstimation with weighted average ...'

    # Initialisation of metrics variable
    X_wa = zeros([numtracts_wa, 1])

    # Initialisation of standard deviation
    std_wa = zeros([numtracts_wa, 1])

    # Calculate partial fractions ignoring the values of zeros
    for i in range(0, numtracts_wa):
        partial_data = data_wa [tracts_wa[i, 0]>0]*tracts_wa[i, 0][tracts_wa[i, 0]>0]

        # Make weighted average if tracts are not zero everywhere
        if sum(tracts_wa[i, 0]) != 0:
            X_wa[i] = sum(partial_data) / sum(tracts_wa[i, 0])

        # Set the metric estimation to 0 for tracts that are zero everywhere
        else:
            print '\tWARNING: Tract number ' + str(i) + ' is zero everywhere . Metric value will be set to 0 for this tract.'
            X_wa[i] = 0.0
            #TODO: the program displays a warning message with this warning, it would be cleaner not to display it

        # Determinate standard deviation
        std_wa[i] = std(partial_data)

    return [X_wa, std_wa]

#=======================================================================================================================
# Estimation of standard deviations
#=======================================================================================================================

def estimate_parameters(P, Y, R_X, U_comp, X0):

    # Initialisation of iterations number
    iter = param.iter

    # Computing standard deviation of data
    sigmaY = std(Y)

    # Determinate range of noise standard deviation
    sigmaN = linspace(0, sigmaY, iter)

    # Exclude zero value (to avoid division by 0)
    sigmaN = sigmaN[1:]

    # Determinate range of metric standard deviation
    sigmaX = linspace(0, sigmaY, iter)

    # Exclude zero value (to avoid division by 0)
    sigmaX = sigmaX[1:]

    # Initialisation of standard deviation errors
    d_n = zeros([iter-1, iter-1])
    d_x = zeros([iter-1, iter-1])

    # Iterations for standard deviations
    for i in range(0, iter-1):
        for j in range(0, iter-1):

            # Estimate metrics with MAP
            [sigma_map, sigma_noise] = MAP(P, Y, R_X, sigmaX[j], sigmaN[i], U_comp, X0)[1:]

            # Errors between sigma
            d_n[i, j] = abs(sigmaN[i] - sigma_noise)
            d_x[i, j] = abs(sigmaX[j] - sigma_map)

    # Determinate sigma noise for minimise error
    min_n = int(argmin(d_n)/(iter-1))
    sigmaN = sigmaN[min_n]

    # Determinate sigma metric for minimise error
    min_x = argmin(d_x[min_n,:])
    sigmaX = sigmaX [min_x]

    # Display sigmas
    print '\nSTD estimations...'
    print '\tsigmaN = ' + str(sigmaN) + '; ' + 'sigmaX = ' + str(sigmaX)

    return [sigmaX, sigmaN]

#=======================================================================================================================
# Estimation of map tracts
#=======================================================================================================================

def bayesian(data, tracts, numtracts):

    # Choice one slice for simplification MAP
    slice_mid = int(data.shape[2]/2)

    # Resizing of data in 1D
    Y = asarray(data[:, :, slice_mid]).reshape(-1, 1)

    # Initialization of matrix linear transformation
    P = zeros([len(Y), numtracts])

    # Matrix of linear transformation
    for label in range(0, numtracts):
        P[:, label] = ((tracts[label, 0])[:,:,slice_mid]).reshape(-1)

    # Save linear transformation before take inverse
    P_save = P
    Py = P.shape[1]

    # Normalisation of linear transformation
    for label in range(0, Py):
        label_sum = sum(P[:,label])
        P[:,label] = P[:, label] / label_sum

    # Mean of data
    X0 = mean(Y)

    # Inverse linear transformation because x=P*y so y=inv(P)*x
    P = pinv(P)

    # Matrix of mean
    U_comp = ones([numtracts, 1])

    # Covariance normalised
    R_X= eye(numtracts)

    # Estimate sigmas before MAP
    [sigmaX, sigmaN] = estimate_parameters(P, Y, R_X, U_comp, X0)

    # Compute MAP
    X_map = MAP(P, Y, R_X, sigmaX, sigmaN, U_comp, X0)[0]

    # Standard deviation of MAP
    std_map = zeros([numtracts, 1])
    for label in range(0, numtracts):
        temp = P_save[:,label]
        sum_label=sum(temp)
        temp = temp[temp>0]
        temp = (temp - X_map[label])
        temp = (temp * temp)
        temp = ((sum(temp)/(sum_label)))
        std_map[label] = sqrt(temp)

    return [X_map, std_map]

#=======================================================================================================================
# MAP
#=======================================================================================================================
def MAP(P, Y, R_X, sigmaX, sigmaN, U_comp, X0):

    # Computing of MAP
    A = dot(P, P.transpose())+(sigmaN/sigmaX)*(sigmaN/sigmaX)*R_X
    B = dot(P, (Y - X0 * dot(P.transpose(), U_comp)))
    X_map = X0 + solve(A,B)

    # Standard deviation in metrics
    sigma_map = std(X_map)

    # Noise estimation after MAP
    noise = Y-dot(P.transpose(),X_map)

    # Noise standard deviation
    sigma_noise = std(noise)

    return [X_map, sigma_map, sigma_noise]

#=======================================================================================================================
# get_slices_matching_with_vertebral_levels
#=======================================================================================================================
def get_slices_matching_with_vertebral_levels(metric_data,fname_tracts,vert_levels_list=None):
    """Return the slices of the input image corresponding to the vertebral levels given as argument."""

    # check existence of "vertebral_labeling.nii.gz" file
    fname_vertebral_labeling = fname_tracts + '/../vertebral_labeling.nii.gz'
    sct.check_file_exist(fname_vertebral_labeling)

     # Read files vertebral_labeling.nii.gz
    print '\nRead files vertebral_labeling.nii.gz...'

    # Load vertebral_labeling.nii.gz
    file_vert_labeling = load(fname_vertebral_labeling)

    # Make data in array format
    data_vert_labeling = file_vert_labeling.get_data()

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

    # Compute the minimum and maximum vertebral levels available in the input image
    min_vert_level, max_vert_level = int(numpy.amin(concatenate(data_vert_labeling,axis=None))), int(numpy.amax(concatenate(data_vert_labeling,axis=None)))

    if vert_levels_list!=None:
        # Check if the vertebral levels selected are available in the input image
        if (vert_levels_list[0] < min_vert_level or vert_levels_list[1] > max_vert_level):
            print '\tERROR: The vertebral levels you selected are not available in the input image.'
            print 'Minimum level asked: '+ str(vert_levels_list[0])
            print '...minimum level available in the input image: '+str(min_vert_level)
            print 'Maximum level asked: '+ str(vert_levels_list[1])
            print '...maximum level available in the input image: '+str(max_vert_level)
            exit_program = 1

        # Exit program if error is detect in sizes
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

        # Return the slice numbers in the right format
        return str(slice_min)+':'+str(slice_max)

    else:

        # Exit program if error is detect in sizes
        if exit_program == 1 :
            print '\nExit program.\n'
            sys.exit(2)

        # Return the minimum and maximum vertebral levels available in the input image
        return [min_vert_level, max_vert_level]

#=======================================================================================================================
# usage
#=======================================================================================================================
def usage(label_title, label_name, label_num, fname_tracts):

    print '\n' \
        'sct_estimate_MAP_tracts\n' \
        '----------------------------------------------------------------------------------------------------------\n'\
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        ' This program extracts metrics (e.g., DTI or MTR) within white matter tracts. It requires an atlas, in the \n' \
        ' same space coordinates as the input image. The current methods for computing the average metrics are: \n' \
        ' (i) weighted average or (ii) bayesian maximum a posteriori (MAP). The atlas is located in a folder and all \n' \
        ' tracts are defined by .txt file. \n' \
        '\n' + label_title + ': \n' \
        ' Label - Tract'
    for label in range(0,len(label_num)):
        print '\t ' + str(label_num[label]) + '\t - ' + label_name[label]
    print '\n'\
        'USAGE\n' \
        ' sct_estimate_MAP_tracts.py -i <data> -t <tracts> -m <mode> -l <label> -z <slice> -o <output>\n' \
        '\n'\
        'MANDATORY ARGUMENTS\n' \
        ' -i <data> : File to extract metrics from.\n' \
        '\n' \
        'OPTIONAL ARGUMENTS\n' \
        ' -l <label> : Label(s) corresponding to the tract(s) to extract the metric from. Begin at 0. ' \
        ' Example: -l 0,5,6,7. By default, all labels are selected.\n' \
        ' -m <method> : Extraction mode : "weightedaverage", "bayesian". Default = weightedaverage. !!! ' \
        ' CURRENTLY, THE bayesian MODE DOES NOT WORK.\n' \
        ' -o <output> : File containing the results of metrics extraction.\n'\
        ' -t <tracts> : Folder that contains atlas. Default = '+fname_tracts+'\n' \
        ' -v <vertebral_levels> : Vertebral levels to estimate the metric accross. Example: \"-v 6:8\" for C6, C7, T1.' \
        ' By defaults, all levels are ' \
        ' selected.'\
        ' -z <slice> : Slices to estimate the metric from. Begin at 0. Example: -z 3:6. By default, all ' \
        ' slices are selected.\n'


    sys.exit(2)

#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = param()
    # call main function
    main()