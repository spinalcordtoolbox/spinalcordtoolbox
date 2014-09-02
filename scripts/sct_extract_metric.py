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

# Import common Python libraries
import os
import getopt
import sys
import time
import commands
import nibabel as nib
import numpy as np
import sct_utils as sct

# get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')


# constants
ALMOST_ZERO = 0.000001

class param:
    def __init__(self):
        self.debug = 0
        self.method = 'wa'
        self.path_label = ''
        self.verbose = 1
        self.labels_of_interest = ''  # list. Example: '1,3,4'. . For all labels, leave empty.
        self.vertebral_levels = ''
        self.slices_of_interest = ''  # 2-element list corresponding to zmin:zmax. example: '5:8'. For all slices, leave
        # empty.
        self.average_all_labels = 0  # average all labels together after concatenation
        self.fname_output = 'metric_label.txt'
        self.file_info_label = 'info_label.txt'
        self.vertebral_labeling_file = path_sct+'/data/template/MNI-Poly-AMU_level.nii.gz'



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
    vertebral_labeling_path = param.vertebral_labeling_file
    fname_normalizing_label = ''  # optional then default is empty
    start_time = time.time()  # save start time for duration
    verbose = param.verbose

    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        fname_data = path_sct+'/data/template/MNI-Poly-AMU_T2.nii.gz' #path_sct+'/testing/data/errsm_23/mt/mtr.nii.gz'
        path_label = path_sct+'/data/atlas' #path_sct+'/testing/data/errsm_23/label/atlas'
        method = 'wa'
        labels_of_interest = '0,1,4,7'  #'0, 2, 5, 7, 15, 22, 27, 29'
        slices_of_interest = '200:210' #'2:4'
        vertebral_levels = ''
        average_all_labels = 0
        fname_output = path_sct+'/testing/sct_extract_metric/results/quantif_mt_debug.txt'
        fname_normalizing_label = path_sct+'/data/template/MNI-Poly-AMU_CSF.nii.gz'


    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'haf:i:l:m:n:o:v:z:') # define flags
    except getopt.GetoptError as err: # check if the arguments are defined
        print str(err) # error
        usage() # display usage
    for opt, arg in opts: # explore flags
        if opt in '-a':
            average_all_labels = 1
        elif opt in '-f':
            path_label = os.path.abspath(arg)  # save path of labels folder
        elif opt == '-h':  # help option
            usage() # display usage
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
        elif opt in '-v': # vertebral levels option, if the user wants to average the metric across specific vertebral
        # levels
             vertebral_levels = arg
        elif opt in '-z': # slices numbers option
            slices_of_interest = arg # save labels numbers


    #TODO: check if the case where the input images are not in AIL orientation is taken into account (if not, implement it)

    # Display usage with tract parameters by default in case files aren't chosen in arguments inputs
    if fname_data == '' or path_label == '':
        #param.path_label = path_label
        usage()

    # Check existence of data file
    sct.printv('\nCheck data file existence...', verbose)
    sct.check_file_exist(fname_data)
    sct.check_file_exist(path_label)

    # add slash at the end
    path_label = sct.slash_at_the_end(path_label, 1)

    # Check input parameters
    check_method(method)

    # Extract label info
    label_id, label_name, label_file = read_label_file(path_label)
    nb_labels_total = len(label_id)

    # check consistency of label input parameter.
    label_id_user = check_labels(labels_of_interest, nb_labels_total)  # If 'labels_of_interest' is empty, then
    # 'label_id_user' contains the index of all labels in the file info_label.txt

    # print parameters
    print '\nChecked parameters:'
    print '  data ................... '+fname_data
    print '  folder label ........... '+path_label
    print '  selected labels ........ '+str(label_id_user)
    print '  estimation method ...... '+method
    print '  vertebral levels ....... '+vertebral_levels
    print '  slices of interest ..... '+slices_of_interest

    # Load image
    sct.printv('\nLoad image...', verbose)
    data = nib.load(fname_data).get_data()
    sct.printv('  Done.', verbose)

    # Get dimensions of data
    sct.printv('\nGet dimensions of data...', verbose)
    nx, ny, nz = data.shape
    sct.printv('.. '+str(nx)+' x '+str(ny)+' x '+str(nz), verbose)

    # load label
    sct.printv('\nLoad labels...', verbose)
    labels = np.empty([nb_labels_total], dtype=object)  # labels(nb_labels_total, x, y, z)
    for i_label in range(0, nb_labels_total):
        labels[i_label] = nib.load(path_label+label_file[i_label]).get_data()
    if fname_normalizing_label:  # if the "normalization" option is wanted,
        normalizing_label = np.empty([1], dtype=object)  # choose this kind of structure so as to keep easily the
        # compatibility with the rest of the code (dimensions: (1, x, y, z))
        normalizing_label[0] = nib.load(fname_normalizing_label).get_data()  # load the data of the normalizing label
    sct.printv('  Done.', verbose)

    # Get dimensions of labels
    sct.printv('\nGet dimensions of label...', verbose)
    nx_atlas, ny_atlas, nz_atlas = labels[0].shape
    sct.printv('.. '+str(nx_atlas)+' x '+str(ny_atlas)+' x '+str(nz_atlas)+' x '+str(nb_labels_total), verbose)

    # Check dimensions consistency between atlas and data
    if (nx, ny, nz) != (nx_atlas, ny_atlas, nz_atlas):
        print '\nERROR: Metric data and labels DO NOT HAVE SAME DIMENSIONS.'
        sys.exit(2)

    # Update the flag "slices_of_interest" according to the vertebral levels selected by user (if it's the case)
    if vertebral_levels != '':
        if slices_of_interest != '':
            print '\nERROR: You cannot select BOTH vertebral levels AND slice numbers.'
            usage()
        else:
            if path_label.endswith('atlas/'):
                vertebral_labeling_path = path_label+'../template/MNI-Poly-AMU_level.nii.gz'
            elif path_label.endswith('template/'):
                vertebral_labeling_path = path_label+'MNI-Poly-AMU_level.nii.gz'
            slices_of_interest = get_slices_matching_with_vertebral_levels(data, vertebral_levels,
                                                                           vertebral_labeling_path)

    # select slice of interest by cropping data and labels
    if slices_of_interest != '':
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

    if fname_normalizing_label:  # if the "normalization" option is wanted,
        #metric_normalizing_label = np.empty([data.shape[-1]], dtype=float)
        for z in range(0, data.shape[-1]):
            normalizing_label_slice = np.empty([1], dtype=object)  # in order to keep compatibility with the function
            # 'extract_metric_within_tract', define a new array for the slice z of the normalizing labels
            normalizing_label_slice[0] = normalizing_label[0][..., z]
            metric_normalizing_label = extract_metric_within_tract(data[..., z], normalizing_label_slice, method,
                                                                   verbose=0)
            # estimate the metric mean in the normalizing label for the slice z
            if metric_normalizing_label[0][0] != 0:
                data[..., z] = data[..., z]/metric_normalizing_label[0][0]  # divide all the slice z by this value

    # extract metrics within labels
    metric_mean, metric_std = extract_metric_within_tract(data, labels, method, verbose=param.verbose)  # mean and std are lists.

    # update label name if average
    if average_all_labels == 1:
        label_name[0] = 'AVERAGED'+' -'.join(label_name[i] for i in label_id_user)  # concatenate the names of the
        # labels selected by the user if the average tag was asked
        label_id_user = [0]  # update "label_id_user" to select the "averaged" label (which is in first position)

    metric_mean = metric_mean[label_id_user]
    metric_std = metric_std[label_id_user]

    # display metrics
    print '\033[1m\nEstimation results:\n'
    for i in range(0, metric_mean.size):
        print '\033[1m'+str(label_id_user[i])+', '+str(label_name[label_id_user[i]])+':    '+str(metric_mean[i])\
              +' +/- '+str(metric_std[i])+'\033[0m'

    # save and display metrics
    save_metrics(label_id_user, label_name, slices_of_interest, vertebral_levels, metric_mean, metric_std, fname_output,
                 fname_data, method, fname_normalizing_label)

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
    # An error could occur at the last line (deletion of the last character of the .txt file), the 5 following code l
    # ines enable to avoid this error:
    line = lines[-1].split(',')
    label_id.append(int(line[0]))
    label_name.append(line[1])
    line[2]=line[2]+' '
    label_file.append(line[2].strip())

    # check if all files listed are present in folder. If not, WARNING.
    print '\nCheck if all files listed in '+param.file_info_label+' are indeed present in '+path_info_label+' ...'
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
def get_slices_matching_with_vertebral_levels(metric_data, vertebral_levels,vertebral_labeling_path):

    sct.printv('\nFind slices corresponding to vertebral levels:', param.verbose)

    # check existence of a vertebral labeling file
    sct.printv('Check file existence...', param.verbose)
    sct.check_file_exist(vertebral_labeling_path)

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
    sct.printv('Load vertebral labeling...', param.verbose)

    # Load the vertebral labeling file and get the data in array format
    data_vert_labeling = nib.load(vertebral_labeling_path).get_data()

    # Extract metric data size X, Y, Z
    [mx, my, mz] = metric_data.shape
    # Extract vertebral labeling data size X, Y, Z
    [vx, vy, vz] = data_vert_labeling.shape

    sct.printv('Check consistency of data size...', param.verbose)

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
    if vert_levels_list[0] < int(np.ndarray.min(data_vert_labeling)) or \
                    vert_levels_list[1] > int(np.ndarray.max(data_vert_labeling)):
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

    # Take into account the case where the ordering of the slice is reversed compared to the ordering of the vertebral
    # levels
    if slice_min_bottom > slice_max_top:
        slice_min = min(Z_top_level)
        slice_max = max(Z_bottom_level)
    else:
        slice_min = min(Z_bottom_level)
        slice_max = max(Z_top_level)

    # display info
    sct.printv('  Vertebral levels correspond to slices: '+str(slice_min)+':'+str(slice_max), param.verbose)

    # Return the slice numbers in the right format ("-1" because the function "remove_slices", which runs next, add 1
    # to the top slice
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
def save_metrics(ind_labels, label_name, slices_of_interest, vertebral_levels, metric_mean, metric_std, fname_output,
                 fname_data, method, fname_normalizing_label):

    # CSV format, header lines start with "#"

    # Save metric in a .txt file
    print '\nWrite results in ' + fname_output + ' ...'

    # Write mode of file
    fid_metric = open(fname_output, 'w')

    # WRITE HEADER:
    # Write date and time
    fid_metric.write('# Date - Time: '+ time.strftime('%Y/%m/%d - %H:%M:%S'))
    # Write metric data file path
    fid_metric.write('\n'+'# Metric file: '+ os.path.abspath(fname_data))
    # If it's the case, write the label used to normalize the metric estimation:
    if fname_normalizing_label:
        fid_metric.write('\n' + '# Label used to normalize the metric estimation slice-by-slice: ' + fname_normalizing_label)
    # Write method used for the metric estimation
    fid_metric.write('\n'+'# Extraction method: '+method)

    # Write selected vertebral levels
    fid_metric.write('\n'+'# Vertebral levels: ')
    if vertebral_levels != '':
        fid_metric.write('%s to %s' % (vertebral_levels.split(':')[0], vertebral_levels.split(':')[1]))
    else:
        fid_metric.write('ALL')

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
# Check the consistency of the method asked by the user
#=======================================================================================================================
def check_method(method):
    if (method != 'wa') & (method != 'ml') & (method != 'bin') & (method != 'wath'):
        print '\nERROR: Method "' + method + '" is not correct. See help. Exit program.\n'
        sys.exit(2)



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
                print '\nERROR: "' + labels_of_interest + '" is not correct. Enter format "1,2,3,4,5,..". Exit program' \
                                                          '.\n'
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
def extract_metric_within_tract(data, labels, method, verbose):

    sct.printv('\nExtract metrics:', verbose=verbose)

    nb_labels = len(labels) # number of labels

    # if user asks for binary regions, binarize atlas
    if method == 'bin':
        for i in range(0, nb_labels):
            labels[i][labels[i] < 0.5] = 0
            labels[i][labels[i] >= 0.5] = 1

    # if user asks for thresholded weighted-average, threshold atlas
    if method == 'wath':
        for i in range(0, nb_labels):
            labels[i][labels[i] < 0.5] = 0

    #  Select non-zero values in the union of all labels
    labels_sum = np.sum(labels)
    ind_nonzero = labels_sum > ALMOST_ZERO
    data1d = data[ind_nonzero]
    labels2d = np.empty([nb_labels, len(data1d)], dtype=float)
    for i in range(0, nb_labels):
        labels2d[i] = labels[i][ind_nonzero]

    # clear memory
    del data, labels

    # Display number of non-zero values
    sct.printv('Number of non-null voxels: '+str(len(data1d)), verbose=verbose)

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
  The labels should be in the same space coordinates as the input image."""
    if label_references != '':
        print """
Current label is:
==========
"""+label_references+"""
=========="""
    print """
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
                          bin: binary masks
  -a                    average all selected labels.
  -o <output>           File containing the results of metrics extraction.
                        Default = """+param.fname_output+"""
  -v <vmin:vmax>        Vertebral levels to estimate the metric across. Example: 2:9 for C2 to T2.
  -z <zmin:zmax>        Slices to estimate the metric from. Example: 5:23. First slice is 0 (not 1)
  -h                    help. Show this message

EXAMPLE
  To see the list of template labels in the template space:
    """+os.path.basename(__file__)+""" -f """+path_sct+"""/data/template

  To see the list of white matter atlas labels in the template space:
    """+os.path.basename(__file__)+""" -f """+path_sct+"""/data/atlas

  To compute FA within labels 0, 2 and 3 within vertebral levels C2 to C7 using binary method:
    """+os.path.basename(__file__)+""" -i dti_FA.nii.gz -f label/atlas -l 0,2,3 -v 2:7 -m bin\n"""

    #Exit program
    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    param = param()
    # call main function
    main()