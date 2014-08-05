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

# TODO: print label name in txt file


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
        self.debug = 0
        self.method = 'wa'
        self.path_label = path_sct+'/data'  # default is toolbox
        self.folder_label = 'template'  # default is template (WM, GM, CSF...)
        self.verbose = 1
        self.labels_of_interest = ''  # list. example: '1,3,4'. . For all labels, leave empty.
        self.slices_of_interest = ''  # 2-element list corresponding to zmin,zmax. example: '5,8'. For all slices, leave empty.
        self.average_all_labels = 0  # average all labels together after concatenation
        self.fname_output = 'metrics.txt'
        self.file_info_label = 'info_label.txt'
        # # by default, labels choice is deactivated and program use all labels
        #self.label_choice = 0
        # # by defaults, the estimation is made accross all vertebral levels
        # self.vertebral_levels = ''
        # # by default, slices choice is desactivated and program use all slices
        # self.slice_choice = 0
        # # by default, program don't export data results in file .txt
        # self.output_choice = 0



#=======================================================================================================================
# main
#=======================================================================================================================
def main():
    # Initialization to defaults parameters
    fname_data = '' # data is empty by default
    path_label = param.path_label
    folder_label = param.folder_label
    method = param.method # extraction mode by default
    labels_of_interest = param.labels_of_interest
    slices_of_interest = param.slices_of_interest
    average_all_labels = param.average_all_labels
    fname_output = param.fname_output
    verbose = param.verbose
    file_info_label = param.file_info_label

    # label_choice = param.label_choice # no select label by default
    # vertebral_levels = param.vertebral_levels # no vertebral level selected by default
    # slice_choice = param.slice_choice # no select label by default
    # output_choice = param.output_choice # no select slice by default
    start_time = time.time()  # save start time for duration
    verbose = param.verbose

    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        fname_data = path_sct+'/testing/data/errsm_23/mt/mtr.nii.gz'
        path_label = path_sct+'/testing/data/errsm_23/label'
        folder_label = 'atlas'
        method = 'wa'
        labels_of_interest = '0, 2'
        average_all_labels = 0
        # label_number = '2,6'
        output_choice = 1
        slice_choice = 1
        slice_number = '1'
        fname_output = 'results.txt'

#    label_id, label_name, label_file = read_label_file(path_atlas+folder_label)

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'haf:i:l:m:o:t:v:z:') # define flags
    except getopt.GetoptError as err: # check if the arguments are defined
        print str(err) # error
        usage() # display usage
    for opt, arg in opts: # explore flags
        if opt in '-a':
            average_all_labels = 1
        elif opt in '-f':
            folder_label = arg
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
        elif opt in '-t':
            path_label = os.path.abspath(arg)  # save path of labels folder
        # elif opt in '-v': # vertebral levels option, if the user wants to average the metric accross specific vertebral levels
        #     vertebral_levels = arg
        elif opt in '-z': # slices numbers option
            slice_choice = 1 # slice choice is activate
            slice_number = arg # save labels numbers
        # TODO: add flag for folder

    #TODO: check if the case where the input images are not in AIL orientation is taken into account (if not, implement it)

    # Display usage with tract parameters by default in case files aren't chosen in arguments inputs
    if fname_data == '' or path_label == '':
        usage()

    # Check existence of data file
    sct.printv('\nCheck file existence...', verbose)
    sct.check_file_exist(fname_data)

    # Extract data path/file/extension
    path_data, file_data, ext_data = sct.extract_fname(fname_data)

    # Check existence of path_label
    path_label = sct.slash_at_the_end(path_label, 1)
    folder_label = sct.slash_at_the_end(folder_label, 1)
    #if not os.path.isdir(path_label):
    #    print('\nERROR: ' + path_label + ' does not exist. Exit program.\n')
    #    sys.exit(2)
    # TODO

    # Check input parameters
    check_method(method)

    # Extract label info
    label_id, label_name, label_file = read_label_file(path_label+folder_label)
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
    print '  folder label ........... '+path_label+folder_label


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
        labels[i_label, :, :, :] = nib.load(path_label+folder_label+label_file[label_id[i_label]]).get_data()
    sct.printv('  Done.', verbose)

    # Get dimensions of atlas
    # TODO: no need to do that if size consistency check is done before
    sct.printv('\nGet dimensions of atlas...', verbose)
    nx_atlas, ny_atlas, nz_atlas = labels[i_label, :, :, :].shape
    sct.printv('.. '+str(nx_atlas)+' x '+str(ny_atlas)+' x '+str(nz_atlas), verbose)

    #
    ## if user selected labels of interest, then update tract list
    #if not labels_of_interest == '':
    #    labels = labels[label_id, :, :, :]

    # select slice of interest by cropping data and atlas
    # if not slices_of_interest == '':
        # TODO

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
    if not fname_output == '':
        save_metrics(label_id, metric_mean, metric_std, fname_output)

    # end of main.
    print



#=======================================================================================================================
# Read label.txt file which is located inside label folder
#=======================================================================================================================
def read_label_file(path_label):
    # TODO



    # Save path of each labels
    #fname_tract = glob.glob(path_atlas + '/*.nii.gz')

    # TODO
    # Check if labels exist in folder
    #if len(fname_tract) == 0:
    #    print '\nERROR: There are not labels in this folder. Exit program.\n'
    #    sys.exit(2)

    # file name of info_label.txt
    fname_label = path_label+param.file_info_label

    # Check info_label.txt existence
    #if len(fname_list) == 0:
    #    print '\nWARNING: There are no file txt in this folder. File list.txt will be create in folder \n'
    sct.check_file_exist(fname_label)

    # Check if labels list.txt is only txt in folder
    #if len(fname_list) > 1:
    #    print '\nWARNING: There are more than one file txt in this folder. File list.txt will be create in folder \n'

    # Read file
    f = open(fname_label)
#    nb = list(set([int(x) for x in labels_of_interest.split(",")]))

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

    # Close file.txt
    f.close()

    # check if files exist
    # TODO
    #
    ## Check if file contain data
    #if len(lines) == 0:
    #    print '\nWARNING: File txt is empty. File list.txt will be create in folder. \n'
    #
    ## Initialisation of label number
    #label_num = [[]] * len(lines)
    #
    ## Initialisation of label name
    #label_name = [[]] * len(lines)
    #
    ## Extract of label title, label name and label number
    #for k in range(0, len(lines)):
    #
    #    # Check if file.txt contains ":" because it is necessary for split label name to label number
    #    if not ':' in lines[k]:
    #        print '\nERROR: File txt is not in correct form. File list.txt must be in this form :\n'
    #        print '\t\tTitle : Name of labels'
    #        print '\t\tLabel 0 : Name Label 0'
    #        print '\t\tLabel 1 : Name Label 1'
    #        print '\t\tLabel 2 : Name Label 2'
    #        print '\t\t...\n '
    #        print '\t\tExample of file.txt'
    #        print '\t\tTitle : List of labels names for the white matter atlas'
    #        print '\t\tLabel 0 : left fasciculus gracilis'
    #        print '\t\tLabel 1 : left fasciculus cuneatus'
    #        print '\t\tLabel 2 : left lateral corticospinal tract'
    #        print '\nExit program. \n'
    #        sys.exit(2)
    #
    #    # Split label name to label number without "['" (begin) and "']" (end) (so 2 to end-2)
    #    else:
    #        [label_num[k],label_name[k]] = lines[k].split(':')
    #        label_name[k] = str(label_name[k].splitlines())[2:-2]
    #
    ## Extract label title as the first line in file.txt
    #label_title = label_name[0]
    #
    ## Extract label name from the following lines
    #label_name = label_name[1:]
    #
    ## Extract label number from the following lines
    #label_num = str(label_num[1:])
    #label_num = [int(x.group()) for x in re.finditer(r'\d+',label_num)]
    #
    ## Check corresponding between label name and tract file
    #if label_num != range(0, len(fname_tract)):
    #    print '\nERROR: File txt and labels are not corresponding. Change file txt or labels .nii.gz. Exit program. \n'
    #    sys.exit(2)

    return [label_id, label_name, label_file]




#=======================================================================================================================
# Save in txt file
#=======================================================================================================================
def save_metrics(ind_labels, metric_mean, metric_std, fname_output):
    print '\nWrite results in ' + fname_output + '...'

    # Write mode of file
    fid_metric = open(fname_output, 'w')

    # Write selected vertebral levels
    # if vertebral_levels != '':
    #     fid_metric.write('%s\t%i to %i\n\n'% ('Vertebral levels : ',vert_levels_list[0],vert_levels_list[1]))
    # else:
    #     fid_metric.write('No vertebral level selected.\n\n')

    # # Write slices chosen
    # fid_metric.write('%s\t%i to %i\n\n'% ('Slices : ',nb_slice[0],nb_slice[1]))

    # Write header title in file .txt
    fid_metric.write('%s,%s,%s\n' % ('Label', 'mean', 'std'))
    # Write metric for label chosen in file .txt
    for i in range(0, len(ind_labels)):
        fid_metric.write('%i,%f,%f\n' % (ind_labels[i], metric_mean[i], metric_std[i]))

    # Close file .txt
    fid_metric.close()



#=======================================================================================================================
def check_method(method):
    if (method != 'wa') & (method != 'ml') & (method != 'bin'):
        print '\nERROR: Method "' + method + '" is not correct. See help. Exit program.\n'
        sys.exit(2)


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
# Read file of tract names and extract names and labels
#=======================================================================================================================
#def read_name(path_atlas):
#
#    # Check if labels folder exist
#    if not os.path.isdir(path_atlas):
#        print('\nERROR: ' + path_atlas + ' does not exist. Exit program.\n')
#        sys.exit(2)
#
#    # Save path of each labels
#    fname_tract = glob.glob(path_atlas + '/*.nii.gz')
#
#    # Check if labels exist in folder
#    if len(fname_tract) == 0:
#        print '\nERROR: There are not labels in this folder. Exit program.\n'
#        sys.exit(2)
#
#    # Save path of file list.txt
#    fname_list = glob.glob(path_atlas + '/*.txt')
#
#    # Check if labels list.txt exist in folder
#    if len(fname_list) == 0:
#        print '\nWARNING: There are no file txt in this folder. File list.txt will be create in folder \n'
#
#    # Check if labels list.txt is only txt in folder
#    if len(fname_list) > 1:
#        print '\nWARNING: There are more than one file txt in this folder. File list.txt will be create in folder \n'
#
#    # Create list.txt default in list.txt in case there are not file or file is not define correctly
#    if len(fname_list) == 0 or len(fname_list) > 1:
#
#        # New file list : list.txt
#        fname_list = path_atlas + '/list.txt'
#
#        # Write mode
#        fid_list = open(fname_list, 'w')
#
#        # Write "Title : Name of labels" by default
#        fid_list.write('%s : %s\n' % ('Title', 'Name of labels'))
#
#        # Write "Label XX : Label XX" by default for "XX" tract number
#        for j in range(0, len(fname_tract)):
#            fid_list.write('%s %i : %s %i\n' % ('Label', j, 'Label', j))
#
#        # Close file txt
#        fid_list.close()
#
#    # Take the value of string instead of array string
#    else:
#        fname_list = fname_list[0]
#
#    # Read file list.txt
#    f = open(fname_list)
#
#    # Extract all lines in file.txt
#    lines = [lines for lines in f.readlines() if lines.strip()]
#
#    # Close file.txt
#    f.close()
#
#    # Check if file contain data
#    if len(lines) == 0:
#        print '\nWARNING: File txt is empty. File list.txt will be create in folder. \n'
#
#    # Initialisation of label number
#    label_num = [[]] * len(lines)
#
#    # Initialisation of label name
#    label_name = [[]] * len(lines)
#
#    # Extract of label title, label name and label number
#    for k in range(0, len(lines)):
#
#        # Check if file.txt contains ":" because it is necessary for split label name to label number
#        if not ':' in lines[k]:
#            print '\nERROR: File txt is not in correct form. File list.txt must be in this form :\n'
#            print '\t\tTitle : Name of labels'
#            print '\t\tLabel 0 : Name Label 0'
#            print '\t\tLabel 1 : Name Label 1'
#            print '\t\tLabel 2 : Name Label 2'
#            print '\t\t...\n '
#            print '\t\tExample of file.txt'
#            print '\t\tTitle : List of labels names for the white matter atlas'
#            print '\t\tLabel 0 : left fasciculus gracilis'
#            print '\t\tLabel 1 : left fasciculus cuneatus'
#            print '\t\tLabel 2 : left lateral corticospinal tract'
#            print '\nExit program. \n'
#            sys.exit(2)
#
#        # Split label name to label number without "['" (begin) and "']" (end) (so 2 to end-2)
#        else:
#            [label_num[k],label_name[k]] = lines[k].split(':')
#            label_name[k] = str(label_name[k].splitlines())[2:-2]
#
#    # Extract label title as the first line in file.txt
#    label_title = label_name[0]
#
#    # Extract label name from the following lines
#    label_name = label_name[1:]
#
#    # Extract label number from the following lines
#    label_num = str(label_num[1:])
#    label_num = [int(x.group()) for x in re.finditer(r'\d+',label_num)]
#
#    # Check corresponding between label name and tract file
#    if label_num != range(0, len(fname_tract)):
#        print '\nERROR: File txt and labels are not corresponding. Change file txt or labels .nii.gz. Exit program. \n'
#        sys.exit(2)
#
#    return [label_title, label_name, label_num, fname_tract]



#=======================================================================================================================
# extract metric within labels
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
# usage
#=======================================================================================================================
# TODO: read default path label and display it
    # TODO
    #"""
    #for label in range(0,len(label_num)):
    #    print '\t ' + str(label_num[label]) + '\t - ' + label_name[label]
    #
    #print """
def usage():
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
  The atlas is located in a folder and all labels are defined by .txt file. By default, the atlas of
  the MNI-Poly-AMU template is used:

  label_title:
  Label - Tract

USAGE
  """+os.path.basename(__file__)+""" -i <data> -t <path_label>

MANDATORY ARGUMENTS
  -i <volume>           file to extract metrics from

OPTIONAL ARGUMENTS
  -t <path_label>       path to the collection of label folders.
                        Default = """+param.path_label+"""
  -f {atlas,template}   folder of label
                        Default = """+param.folder_label+"""
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

    #Exit Program
    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    param = param()
    # call main function
    main()