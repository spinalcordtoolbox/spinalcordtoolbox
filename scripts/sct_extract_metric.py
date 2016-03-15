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
# Created: 2014-07-30
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: find another method to update label in case average_all_labels == 1. E.g., recreate tmp label file.
# TODO: remove class color and use sct_printv
# TODO: add documentation for new features
# TODO (not urgent): vertebral levels selection should only consider voxels of the selected levels in slices where two different vertebral levels coexist (and not the whole slice)

# Import common Python libraries
import os
import getopt
import sys
import commands
import time

import nibabel as nib
import numpy as np

import sct_utils as sct
from sct_image import get_orientation_3d, set_orientation
from msct_image import Image
from msct_parser import Parser


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
        self.ml_clusters = '0:29,30,31'  # three classes: WM, GM and CSF
        self.adv_param = ['10',  # STD of the metric value across labels, in percentage of the mean (mean is estimated using cluster-based ML)
                          '10'] # STD of the assumed gaussian-distributed noise

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


def get_parser():

    param_default = Param()

    # read the .txt files referencing the labels
    if param_default.path_label != '':
        file_label = param_default.path_label + '/' + param_default.file_info_label
        sct.check_file_exist(file_label, 0)
        default_info_label = open(file_label, 'r')
        label_references = default_info_label.read()
    else:
        label_references = ''

    parser = Parser(__file__)
    parser.usage.set_description("""This program extracts metrics (e.g., DTI or MTR) within labels. The labels are generated with 'sct_warp_template'. The label folder contains a file (info_label.txt) that describes all labels. The labels should be in the same space coordinates as the input image.""")
    parser.add_option(name='-i',
                      type_value='image_nifti',
                      description='File to extract metrics from.',
                      mandatory=True,
                      example='fmri.nii.gz')
    parser.add_option(name='-f',
                      type_value='folder',
                      description='Folder including labels to extract the metric from.',
                      mandatory=True,
                      example='$SCT_DIR/data/template')
    parser.add_option(name='-l',
                      type_value='str',
                      description="""Label number to extract the metric from. Example: 1,3 for left fasciculus cuneatus and left ventral spinocerebellar tract in folder '/atlas'. Default = all labels.
You can also select labels using 1:3 to get labels 1,2,3.
Following shortcuts are also available for the folder label "atlas/":
 -l sc: extract in the spinal cord cord
 -l wm: extract in the white matter
 -l gm: extract in the gray matter""",
                      mandatory=False,
                      default_value=param_default.labels_of_interest)
    parser.add_option(name='-method',
                      type_value='multiple_choice',
                      description="""Method to extract metrics.
ml: maximum likelihood (only use with well-defined regions and low noise)
  N.B. ONLY USE THIS METHOD WITH THE WHITE MATTER ATLAS!
map: maximum a posteriori. Mean priors are estimated by maximum likelihood within three clusters (white matter, gray matter and CSF). Tract and  noise variance are set with flag -p.
  N.B. ONLY USE THIS METHOD WITH THE WHITE MATTER ATLAS!
wa: weighted average
wath: weighted average (only consider values >0.5)
bin: binarize mask (threshold=0.5)""",
                      example=['ml', 'map', 'wa', 'wath', 'bin'],
                      mandatory=False,
                      default_value=param_default.method)
    parser.add_option(name='-m',
                      type_value='multiple_choice',
                      description="""Method to extract metrics.
ml: maximum likelihood (only use with well-defined regions and low noise)
  N.B. ONLY USE THIS METHOD WITH THE WHITE MATTER ATLAS!
map: maximum a posteriori. Mean priors are estimated by maximum likelihood within three clusters (white matter, gray matter and CSF). Tract and  noise variance are set with flag -p.
  N.B. ONLY USE THIS METHOD WITH THE WHITE MATTER ATLAS!
wa: weighted average
wath: weighted average (only consider values >0.5)
bin: binarize mask (threshold=0.5)""",
                      mandatory=False,
                      default_value=param_default.method,
                      deprecated_by='-method')
    parser.add_option(name='-param',
                      type_value='str',
                      description="""Advanced parameters for the 'map' method. Separate with comma. All items must be listed (separated with comma).
#1: standard deviation of metrics across labels
#2: standard deviation of the noise (assumed Gaussian)""",
                      mandatory=False,
                      example=param_default.adv_param[0]+','+param_default.adv_param[1])
    parser.add_option(name='-p',
                      type_value=None,
                      description="""Advanced parameters for the 'map' method. Separate with comma. All items must be listed (separated with comma).
#1: standard deviation of metrics across labels
#2: standard deviation of the noise (assumed Gaussian)""",
                      mandatory=False,
                      deprecated_by='-param')
    parser.add_option(name='-o',
                      type_value='file_output',
                      description='File containing the results of metrics extraction.',
                      mandatory=False,
                      default_value=param_default.fname_output)
    parser.add_option(name='-a',
                      type_value=None,
                      description='Average all selected labels.',
                      mandatory=False)
    parser.add_option(name='-vert',
                      type_value='str',
                      description='Vertebral levels to estimate the metric across. Example: 2:9 for C2 to T2.',
                      mandatory=False,
                      example='2:9',
                      default_value=param_default.vertebral_levels)
    parser.add_option(name='-v',
                      type_value='str',
                      description='Vertebral levels to estimate the metric across. Example: 2:9 for C2 to T2.',
                      mandatory=False,
                      example='2:9',
                      deprecated_by='-vert')
    parser.add_option(name='-z',
                      type_value='str',
                      description='Slice range to estimate the metric from. First slice is 0. Example: 5:23\nYou can also select specific slices using commas. Example: 0,2,3,5,12',
                      mandatory=False,
                      example='5:23',
                      default_value=param_default.slices_of_interest)
    parser.add_option(name='-norm-file',
                      type_value='image_nifti',
                      description='Filename of the label by which the user wants to normalize',
                      mandatory=False)
    parser.add_option(name='-n',
                      type_value='image_nifti',
                      description='Filename of the label by which the user wants to normalize',
                      mandatory=False,
                      deprecated_by='-norm-file')
    parser.add_option(name='-norm-method',
                      type_value='multiple_choice',
                      description='Method to use for normalization:\n- sbs: normalization slice-by-slice\n- whole: normalization by the metric value in the whole label for all slices.',
                      example=['sbs', 'whole'],
                      mandatory=False)
    parser.add_option(name='-w',
                      type_value='image_nifti',
                      description='Filename of the label by which the user wants to normalize',
                      mandatory=False,
                      deprecated_by='-norm-method')

    str_section = """\n
To list template labels:
""" + os.path.basename(__file__) + """ -f $SCT_DIR/data/template

To list white matter atlas labels:
""" + os.path.basename(__file__) + """ -f $SCT_DIR/data/atlas

To compute FA within labels 0, 2 and 3 within vertebral levels C2 to C7 using binary method:
""" + os.path.basename(__file__) + """ -i dti_FA.nii.gz -f label/atlas -l 0,2,3 -v 2:7 -m bin"""
    if label_references != '':
        str_section += """
List of labels in: """ + file_label + """:
==========
""" + label_references + """
=========="""

    parser.usage.addSection(str_section)

    return parser



#=======================================================================================================================
# main
#=======================================================================================================================
def main(fname_data, path_label, method, labels_of_interest, slices_of_interest, vertebral_levels, average_all_labels, fname_output, fname_normalizing_label, normalization_method, adv_param_user):
    # Initialization to defaults parameters
    fname_vertebral_labeling = param.fname_vertebral_labeling
    actual_vert_levels = None  # variable used in case the vertebral levels asked by the user don't correspond exactly to the vertebral levels available in the metric data
    warning_vert_levels = None  # variable used to warn the user in case the vertebral levels he asked don't correspond exactly to the vertebral levels available in the metric data
    verbose = param.verbose
    flag_h = 0
    ml_clusters = param.ml_clusters
    adv_param = param.adv_param
    clustered_labels = []
    matching_cluster_labels = []

    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        status, path_sct_data = commands.getstatusoutput('echo $SCT_TESTING_DATA_DIR')
        fname_data = '/Users/julien/data/temp/sct_example_data/mt/mtr.nii.gz'
        path_label = '/Users/julien/data/temp/sct_example_data/mt/label/atlas/'
        method = 'map'
        ml_clusters = '0:29,30,31'
        labels_of_interest = '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29'
        slices_of_interest = ''
        vertebral_levels = ''
        average_all_labels = 1
        fname_normalizing_label = ''  #path_sct+'/testing/data/errsm_23/mt/label/template/MNI-Poly-AMU_CSF.nii.gz'
        normalization_method = ''  #'whole'

    # add slash at the end
    path_label = sct.slash_at_the_end(path_label, 1)

    # Find path to the vertebral labeling file if vertebral levels were specified by the user
    if vertebral_levels:
        if slices_of_interest:  # impossible to select BOTH specific slices and specific vertebral levels
            sct.printv(parser.usage.generate(error='ERROR: You cannot select BOTH vertebral levels AND slice numbers.'))
        else:
            fname_vertebral_labeling_list = sct.find_file_within_folder(fname_vertebral_labeling, path_label + '..')
            if len(fname_vertebral_labeling_list) > 1:
                sct.printv(parser.usage.generate(error='ERROR: More than one file named "' + fname_vertebral_labeling + '" were found in ' + path_label + '. Exit program.'))
            elif len(fname_vertebral_labeling_list) == 0:
                sct.printv(parser.usage.generate(error='ERROR: No file named "' + fname_vertebral_labeling + '" were found in ' + path_label + '. Exit program.'))
            else:
                fname_vertebral_labeling = os.path.abspath(fname_vertebral_labeling_list[0])

    # Check input parameters
    check_method(method, fname_normalizing_label, normalization_method)

    # parse argument for param
    if not adv_param_user == '':
        adv_param = adv_param_user.replace(' ', '').split(',')  # remove spaces and parse with comma
        del adv_param_user  # clean variable
        # TODO: check integrity of input

    # Extract label info
    label_id, label_name, label_file = read_label_file(path_label, param.file_info_label, method)
    nb_labels_total = len(label_id)

    # check consistency of label input parameter.
    label_id_user, average_all_labels = check_labels(labels_of_interest, nb_labels_total, average_all_labels, method)  # If 'labels_of_interest' is empty, then
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
    print '  advanced parameters ....... '+str(adv_param)

    # Check if the orientation of the data is RPI
    input_im = Image(fname_data)
    orientation_data = get_orientation_3d(input_im)

    # If orientation is not RPI, change to RPI
    if orientation_data != 'RPI':
        sct.printv('\nCreate temporary folder to change the orientation of the NIFTI files into RPI...', verbose)
        path_tmp = sct.tmp_create()
        # change orientation and load data
        sct.printv('\nChange image orientation and load it...', verbose)
        im_orient = set_orientation(input_im, 'RPI', fname_out=path_tmp+'metric_RPI.nii')
        data = im_orient.data
        # Do the same for labels
        sct.printv('\nChange labels orientation and load them...', verbose)
        labels = np.empty([nb_labels_total], dtype=object)  # labels(nb_labels_total, x, y, z)
        for i_label in range(0, nb_labels_total):
            im_label = set_orientation(Image(path_label+label_file[i_label]), 'RPI', fname_out=path_tmp+'label_'+str(i_label)+'_RPI.nii')
            labels[i_label] = im_label.data
        if fname_normalizing_label:  # if the "normalization" option is wanted,
            normalizing_label = np.empty([1], dtype=object)  # choose this kind of structure so as to keep easily the
            # compatibility with the rest of the code (dimensions: (1, x, y, z))
            im_normalizing_label = set_orientation(Image(fname_normalizing_label), 'RPI', fname_out=path_tmp+'normalizing_label_RPI.nii')
            normalizing_label[0] = im_normalizing_label.data
        if vertebral_levels:  # if vertebral levels were selected,
            im_vertebral_labeling = set_orientation(Image(fname_vertebral_labeling), 'RPI', fname_out=path_tmp+'vertebral_labeling_RPI.nii')
            data_vertebral_labeling = im_vertebral_labeling.data
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
    data[np.isneginf(data)] = 0.0
    data[data < 0.0] = 0.0
    data[np.isnan(data)] = 0.0
    data[np.isposinf(data)] = np.nanmax(data)

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
            get_slices_matching_with_vertebral_levels(data, vertebral_levels, data_vertebral_labeling, verbose)

    # select slice of interest by cropping data and labels
    if slices_of_interest:
        data = remove_slices(data, slices_of_interest)
        for i_label in range(0, nb_labels_total):
            labels[i_label] = remove_slices(labels[i_label], slices_of_interest)
        if fname_normalizing_label:  # if the "normalization" option was selected,
            normalizing_label[0] = remove_slices(normalizing_label[0], slices_of_interest)

    if method == 'map':
        # get clustered labels
        clustered_labels, matching_cluster_labels = get_clustered_labels(ml_clusters, labels, label_id_user, average_all_labels, verbose)

    # if user wants to get unique value across labels, then combine all labels together
    if average_all_labels == 1:
        sum_labels_user = np.sum(labels[label_id_user])  # sum the labels selected by user
        if method == 'ml' or method == 'map':  # in case the maximum likelihood and the average across different labels are wanted

            # merge labels
            labels_tmp = np.empty([nb_labels_total - len(label_id_user) + 1], dtype=object)
            labels = np.delete(labels, label_id_user)  # remove the labels selected by user
            labels_tmp[0] = sum_labels_user  # put the sum of the labels selected by user in first position of the tmp variable
            for i_label in range(1, len(labels_tmp)):
                labels_tmp[i_label] = labels[i_label-1]  # fill the temporary array with the values of the non-selected labels
            labels = labels_tmp  # replace the initial labels value by the updated ones (with the summed labels)
            del labels_tmp  # delete the temporary labels

        else:  # in other cases than the maximum likelihood, we can remove other labels (not needed for estimation)
            labels = np.empty(1, dtype=object)
            labels[0] = sum_labels_user  # we create a new label array that includes only the summed labels

    if fname_normalizing_label:  # if the "normalization" option is wanted
        sct.printv('\nExtract normalization values...', verbose)
        if normalization_method == 'sbs':  # case: the user wants to normalize slice-by-slice
            for z in range(0, data.shape[-1]):
                normalizing_label_slice = np.empty([1], dtype=object)  # in order to keep compatibility with the function
                # 'extract_metric_within_tract', define a new array for the slice z of the normalizing labels
                normalizing_label_slice[0] = normalizing_label[0][..., z]
                metric_normalizing_label = extract_metric_within_tract(data[..., z], normalizing_label_slice, method, 0)
                # estimate the metric mean in the normalizing label for the slice z
                if metric_normalizing_label[0][0] != 0:
                    data[..., z] = data[..., z]/metric_normalizing_label[0][0]  # divide all the slice z by this value

        elif normalization_method == 'whole':  # case: the user wants to normalize after estimations in the whole labels
            metric_mean_norm_label, metric_std_norm_label = extract_metric_within_tract(data, normalizing_label, method, param.verbose)  # mean and std are lists

    # extract metrics within labels
    sct.printv('\nExtract metric within labels...', verbose)
    metric_mean, metric_std = extract_metric_within_tract(data, labels, method, verbose, clustered_labels, matching_cluster_labels, adv_param)  # mean and std are lists

    if fname_normalizing_label and normalization_method == 'whole':  # case: user wants to normalize after estimations in the whole labels
        metric_mean, metric_std = np.divide(metric_mean, metric_mean_norm_label), np.divide(metric_std, metric_std_norm_label)

    # update label name if average
    if average_all_labels == 1:
        label_name[0] = 'AVERAGED: '+' -'.join(label_name[i] for i in label_id_user)  # concatenate the names of the labels selected by the user if the average tag was asked
        label_id_user = [0]  # update "label_id_user" to select the "averaged" label (which is in first position)

    metric_mean = metric_mean[label_id_user]
    metric_std = metric_std[label_id_user]

    # display metrics
    sct.printv('\nEstimation results:', 1)
    for i in range(0, metric_mean.size):
        sct.printv(str(label_id_user[i])+', '+str(label_name[label_id_user[i]])+':    '+str(metric_mean[i])+' +/- '+str(metric_std[i]), 1, 'info')

    # save metrics
    save_metrics(label_id_user, label_name, slices_of_interest, metric_mean, metric_std, fname_output, fname_data,
                 method, fname_normalizing_label, actual_vert_levels, warning_vert_levels)



#=======================================================================================================================
# Read label.txt file which is located inside label folder
#=======================================================================================================================
def read_label_file(path_info_label, file_info_label, method=''):

    # file name of info_label.txt
    fname_label = path_info_label+file_info_label

    # Check info_label.txt existence
    sct.check_file_exist(fname_label)

    # Read file
    f = open(fname_label)

    # Extract all lines in file.txt
    lines = [lines for lines in f.readlines() if lines.strip()]

    # In case the method 'ml' or 'map' was selected, check if the White matter atlas was provided by the user
    if method in ['ml', 'map']:
        # look at first line
        header_lines = [lines[i] for i in range(0, len(lines)) if lines[i][0] == '#']
        info_label_title = header_lines[0].split('-')[0].strip()
        if '# White matter atlas' not in info_label_title:
            sct.printv("ERROR: You selected the method \'"+method+"\' but you did not provide the White matter atlas. You provided the "+info_label_title+". Please provide the White matter atlas to use the method \'"+method+"\'.", type='error' )

    # remove header lines (every line starting with "#")
    lines = [lines[i] for i in range(0, len(lines)) if lines[i][0] != '#']

    # read each line
    label_id = []
    label_name = []
    label_file = []
    for i in range(0, len(lines)-1):
        line = lines[i].split(',')
        label_id.append(int(line[0]))
        label_name.append(line[1].strip())
        label_file.append(line[2][:-1].strip())
    # An error could occur at the last line (deletion of the last character of the .txt file), the 5 following code
    # lines enable to avoid this error:
    line = lines[-1].split(',')
    label_id.append(int(line[0]))
    label_name.append(line[1].strip())
    line[2]=line[2]+' '
    label_file.append(line[2].strip())

    # check if all files listed are present in folder. If not, WARNING.
    # print '\nCheck existence of all files listed in '+file_info_label+' ...'
    for fname in label_file:
        sct.check_file_exist(path_info_label+fname)
        # if os.path.isfile(path_info_label+fname) or os.path.isfile(path_info_label+fname + '.nii') or \
        #         os.path.isfile(path_info_label+fname + '.nii.gz'):
        #     sct.printv('  OK: '+path_info_label+fname, param.verbose)
        #     pass
        # else:
        #     sct.printv('  ERROR: ' + path_info_label+fname + ' does not exist\n', 1, 'error')

    # Close file.txt
    f.close()

    return [label_id, label_name, label_file]


#=======================================================================================================================
# Return the slices of the input image corresponding to the vertebral levels given as argument
#=======================================================================================================================
def get_slices_matching_with_vertebral_levels(metric_data, vertebral_levels, data_vertebral_labeling, verbose=1):
    
    # TODO: use sct_utils instead (jcohenadad)
    color = Color()

    sct.printv('\nFind slices corresponding to vertebral levels...', verbose)

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

    if vert_levels_list[1] > max_vert_level_available:
        vert_levels_list[1] = max_vert_level_available
        warning.append('WARNING: the top vertebral level you selected is higher to the highest level available --> '
                       'Selected the highest vertebral level available: ' + str(int(vert_levels_list[1])))  # record the
        # warning to write it later in the .txt output file

        print color.yellow + 'WARNING: the top vertebral level you selected is higher to the highest ' \
                                          'level available \n--> Selected the highest vertebral level available: ' + \
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

    sct.printv('  Check consistency of data size...', verbose)

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

    sct.printv('  Find slices corresponding to vertebral levels...', verbose)
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
    sct.printv('    '+str(slice_min)+':'+str(slice_max), verbose)

    # Return the slice numbers in the right format ("-1" because the function "remove_slices", which runs next, add 1
    # to the top slice
    return str(slice_min)+':'+str(slice_max), vert_levels_list, warning


#=======================================================================================================================
# Crop data to only keep the slices asked by user
#=======================================================================================================================
def remove_slices(data_to_crop, slices_of_interest):

    # check if user selected specific slices using delimitor ','
    if not slices_of_interest.find(',') == -1:
        slices_list = [int(x) for x in slices_of_interest.split(',')]  # n-element list
    else:
        slices_range = [int(x) for x in slices_of_interest.split(':')]  # 2-element list
        # if only one slice (z) was selected, consider as z:z
        if len(slices_range) == 1:
            slices_range = [slices_range[0], slices_range[0]]
        slices_list = [i for i in range(slices_range[0], slices_range[1]+1)]

    # Remove slices that are not wanted (+1 is to include the last selected slice as Python "includes -1"
    data_cropped = data_to_crop[..., slices_list]

    return data_cropped


#=======================================================================================================================
# Save in txt file
#=======================================================================================================================
def save_metrics(ind_labels, label_name, slices_of_interest, metric_mean, metric_std, fname_output, fname_data, method, fname_normalizing_label, actual_vert=None, warning_vert_levels=None):

    # CSV format, header lines start with "#"

    # Save metric in a .txt file
    sct.printv('\nWrite results in ' + fname_output + '...')

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
        fid_metric.write(slices_of_interest)
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
    sct.printv('\tDone.')



#=======================================================================================================================
# Check the consistency of the methods asked by the user
#=======================================================================================================================
def check_method(method, fname_normalizing_label, normalization_method):
    if (method != 'wa') & (method != 'ml') & (method != 'bin') & (method != 'wath') & (method != 'map'):
        sct.printv(parser.usage.generate(error='ERROR: Method "' + method + '" is not correct. See help. Exit program.\n'))

    if normalization_method and not fname_normalizing_label:
        sct.printv(parser.usage.generate(error='ERROR: You selected a normalization method ('+ str(normalization_method)+ ') but you didn\'t selected any label to be used for the normalization.'))

    if fname_normalizing_label and normalization_method != 'sbs' and normalization_method != 'whole':
        sct.printv(parser.usage.generate(error='\nERROR: The normalization method you selected is incorrect:'+str(normalization_method)))



#=======================================================================================================================
# Check the consistency of the labels asked by the user
#=======================================================================================================================
def check_labels(labels_of_interest, nb_labels, average_labels, method):

    # by default, all labels are selected
    list_label_id = range(0, nb_labels)

    if labels_of_interest:
        # Check if label chosen is in the right format
        for char in labels_of_interest:
            if not char in '0123456789,:scwmg':
                sct.printv(parser.usage.generate(error='\nERROR: ' + labels_of_interest + ' is not the correct format to select labels.\n Exit program.\n'))

        # if spinal cord was selected, need all 32 labels from folder atlas
        if labels_of_interest == 'sc':
            if nb_labels < 32 and (method == 'ml' or method == 'map'):
                sct.printv(parser.usage.generate(error='\nERROR: You\'ve asked to extract metric in the all spinal cord using the method '+method+' but your atlas folder containing'
                           ' the labels only contains '+nb_labels+' labels. You need all 32 labels from the folder /atlas of'
                            ' the SpinalCordToolbox (files WMtract_XX, with XX from 00 to 31).\nExit program.\n\n'))

            if nb_labels < 31 and (method != 'ml' and method != 'map'):
                sct.printv(parser.usage.generate(error='\nERROR: You\'ve asked to extract metric in the all spinal cord using the method '+method+' but your atlas folder containing'
                           ' the labels only contains '+nb_labels+' labels. You need all 30 white matter tracts and the gray matter from the folder /atlas of'
                            ' the SpinalCordToolbox (files WMtract_XX, with XX from 00 to 30).\nExit program.\n\n'))

            else:
                list_label_id = range(0, 31)
                average_labels = 1

        elif labels_of_interest == 'gm':
            if nb_labels < 32 and (method == 'ml' or method == 'map'):
                sct.printv(parser.usage.generate(error='\nERROR: You\'ve asked to extract metric in the gray matter using the method '+method+' but your atlas folder containing'
                           ' the labels only contains '+nb_labels+' labels. You need all 32 labels from the folder /atlas of'
                            ' the SpinalCordToolbox (files WMtract_XX, with XX from 00 to 31).\nExit program.\n\n'))
            else:
                list_label_id = [30]

        elif labels_of_interest == 'wm':
            if nb_labels < 32 and (method == 'ml' or method == 'map'):
                sct.printv(parser.usage.generate(error='\nERROR: You\'ve asked to extract metric in the white matter using the method '+method+' but your atlas folder containing'
                           ' the labels only contains '+nb_labels+' labels. You need all 32 labels from the folder /atlas of'
                            ' the SpinalCordToolbox (files WMtract_XX, with XX from 00 to 31).\nExit program.\n\n'))

            else:
                list_label_id = range(0, 30)
                average_labels = 1

        elif ':' in labels_of_interest:
            label_ids_range = [int(x) for x in labels_of_interest.split(':')]
            if len(label_ids_range)>2:
                sct.printv(parser.usage.generate(error='\nERROR: label IDs selection must be in format X:Y, with X and Y between 0 and 31.\nExit program.\n\n'))

            else:
                label_ids_range.sort()
                list_label_id = [int(x) for x in range(label_ids_range[0], label_ids_range[1]+1)]

        else:
            list_label_id = list(set([int(x) for x in labels_of_interest.split(",")]))

        # Sort labels ID and remove redundant values
        list_label_id.sort()
        list_label_id = list(set(list_label_id))

    return list_label_id, average_labels


#=======================================================================================================================
# Extract metric within labels
#=======================================================================================================================
def extract_metric_within_tract(data, labels, method, verbose, clustered_labels=[], matching_cluster_labels=[], adv_param=[]):
    """
    :data: (nx,ny,nz) numpy array
    :labels: nlabel tuple of (nx,ny,nz) array
    """


    nb_labels = len(labels)  # number of labels

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
    ind_positive_labels = labels_sum > ALMOST_ZERO  # labels_sum > ALMOST_ZERO
    # ind_positive_data = data > -9999999999  # data > 0
    ind_positive = ind_positive_labels  # & ind_positive_data
    data1d = data[ind_positive]
    nb_vox = len(data1d)
    labels2d = np.empty([nb_labels, nb_vox], dtype=float)
    for i in range(0, nb_labels):
        labels2d[i] = labels[i][ind_positive]

    # # display labels
    # import matplotlib.pyplot as plt
    # plt.imshow(labels_sum[:,:,3])
    # plt.show()
    # plt.imshow(data[:,:,3])
    # plt.show()

    # Display number of non-zero values
    sct.printv('  Number of non-null voxels: '+str(nb_vox), verbose=verbose)

    # initialization
    metric_mean = np.empty([nb_labels], dtype=object)
    metric_std = np.empty([nb_labels], dtype=object)

    # Estimation with 3-class maximum likelihood
    if method == 'map':
        sct.printv('Estimation maximum likelihood within clustered labels...', verbose=verbose)

        nb_clusters = len(clustered_labels)

        #  Select non-zero values in the union of the clustered labels
        clustered_labels_sum = np.sum(clustered_labels)
        ind_positive_clustered_labels = clustered_labels_sum > ALMOST_ZERO  # labels_sum > ALMOST_ZERO

        y = data[ind_positive_clustered_labels]  # [nb_vox x 1]
        # create matrix X to use ML and estimate beta_0
        x = np.zeros([len(y), nb_clusters])
        for i_cluster in range(0, nb_clusters):
            x[:, i_cluster] = clustered_labels[i_cluster][ind_positive_clustered_labels]

        # estimate values using ML for each cluster
        beta = np.dot( np.linalg.pinv(np.dot(x.T, x)), np.dot(x.T, y) )  # beta = (Xt . X)-1 . Xt . y
        # display results
        sct.printv('  Estimated beta per cluster: '+str(beta), verbose=verbose)

    # clear memory
    del data, labels

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
                metric_std[i_label] = np.sqrt(sum(labels2d[i_label, :] * (data1d - metric_mean[i_label])**2 ) / sum(labels2d[i_label, :]))

    # Estimation with maximum likelihood
    if method == 'ml':
        y = data1d  # [nb_vox x 1]
        x = labels2d.T  # [nb_vox x nb_labels]
        beta = np.dot( np.linalg.pinv(np.dot(x.T, x)), np.dot(x.T, y) )  # beta = (Xt . X)-1 . Xt . y
        #beta, residuals, rank, singular_value = np.linalg.lstsq(np.dot(x.T, x), np.dot(x.T, y), rcond=-1)
        #beta, residuals, rank, singular_value = np.linalg.lstsq(x, y)
        #print beta, residuals, rank, singular_value
        for i_label in range(0, nb_labels):
            metric_mean[i_label] = beta[i_label]
            metric_std[i_label] = 0  # need to assign a value for writing output file

    # Estimation with maximum a posteriori (map)
    if method == 'map':
        # perc_var_label = int(adv_param[0])^2  # variance within label, in percentage of the mean (mean is estimated using cluster-based ML)
        var_label = int(adv_param[0]) ^ 2  # variance within label
        var_noise = int(adv_param[1]) ^ 2  # variance of the noise (assumed Gaussian)

        y = data1d  # [nb_vox x 1]
        x = labels2d.T  # [nb_vox x nb_labels]
        # construct beta0
        beta0 = np.zeros(nb_labels)
        for i_cluster in range(nb_clusters):
            beta0[np.where(np.asarray(matching_cluster_labels) == i_cluster)[0]] = beta[i_cluster]
        # construct covariance matrix (variance between tracts). For simplicity, we set it to be the identity.
        Rlabel = np.diag(np.ones(nb_labels))
        # Vlabel =  np.diag(np.ones(nb_labels) * var_label)
        # Vlabel =  np.diag(beta0 * perc_var_label * 0.01)  # [nb_labels x nb_labels]
        # construct noise matrix
        # Vnoise = np.diag(np.ones(nb_labels) * var_noise)
        # beta = beta0 + (Xt . X + var_noise/Var_label * Rlabel^-1)^-1 . Xt . ( y - X . beta0 )
        # beta = beta0 +                      A                        . B  .         C
        # A = np.linalg.pinv(np.dot(x.T, x) + np.dot(Vnoise, np.linalg.pinv(Vlabel)))
        A = np.linalg.pinv(np.dot(x.T, x) + np.linalg.pinv(Rlabel) * var_noise/var_label)
        B = x.T
        C = y - np.dot(x, beta0)
        beta = beta0 + np.dot(A, np.dot(B, C))
        for i_label in range(0, nb_labels):
            metric_mean[i_label] = beta[i_label]
            metric_std[i_label] = 0  # need to assign a value for writing output file

    return metric_mean, metric_std

# ======================================================================================================================
def get_clustered_labels(ml_clusters, labels, labels_user, averaging_flag, verbose):
    """
    Cluster labels according to selected options (labels and averaging).
    :ml_clusters: clusters in form: '0:29,30,31'
    :labels: all labels data
    :labels_user: label IDs selected by the user
    :averaging_flag: flag -a (0 or 1)
    :return: clustered_labels: labels summed by clustered
    """

    # get the label IDs included in each cluster
    clusters_list = ml_clusters.split(',')
    nb_clusters = len(clusters_list)
    clusters_all_labels = []
    for cluster in clusters_list:
        limits = cluster.split(':')
        clusters_all_labels.append(range(int(limits[0]), int(limits[-1])+1))

    # find matching between labels and clusters in the label id list selected by the user
    matching_cluster_label_id_user = np.zeros(len(labels_user), dtype=int)
    for i_label in range(0, len(labels_user)):
        for i_cluster in range(0, nb_clusters):
            if labels_user[i_label] in clusters_all_labels[i_cluster]:
                matching_cluster_label_id_user[i_label] = i_cluster

    # reorganize the cluster according to the averaging flag chosen
    if averaging_flag:
        matching_cluster_label_id_unique = np.unique(matching_cluster_label_id_user)
        if matching_cluster_label_id_unique.size != 1:
            merged_cluster = []
            for i_cluster in matching_cluster_label_id_unique:
                merged_cluster = merged_cluster + clusters_all_labels[i_cluster]
            clusters_all_labels = list(np.delete(np.asarray(clusters_all_labels), matching_cluster_label_id_unique))
            clusters_all_labels.insert(matching_cluster_label_id_unique[0], merged_cluster)
    nb_clusters = len(clusters_all_labels)
    sct.printv('  Number of clusters: '+str(nb_clusters), verbose=verbose)

    # sum labels within each cluster
    clustered_labels = np.empty([nb_clusters], dtype=object)  # labels(nb_labels_total, x, y, z)
    for i_cluster in range(0, nb_clusters):
        clustered_labels[i_cluster] = np.sum(labels[clusters_all_labels[i_cluster]])

    # find matching between labels and clusters in the whole label id list
    matching_cluster_label_id = np.zeros(len(labels), dtype=int)
    for i_label in range(0, len(labels)):
        for i_cluster in range(0, nb_clusters):
            if i_label in clusters_all_labels[i_cluster]:
                matching_cluster_label_id[i_label] = i_cluster
    if averaging_flag:
        cluster_averaged_labels = matching_cluster_label_id[labels_user]
        matching_cluster_label_id = list(np.delete(np.asarray(matching_cluster_label_id), labels_user))
        matching_cluster_label_id.insert(0, cluster_averaged_labels[0])  # because the average of labels will be placed in the first position

    return clustered_labels, matching_cluster_label_id



#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    param_default = Param()
    param = Param()
    color = Color()

    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])

    # Initialization to defaults parameters
    fname_data = arguments['-i']
    path_label = arguments['-f']
    labels_of_interest = ''
    vertebral_levels = ''

    if '-l' in arguments:
        labels_of_interest = arguments['-l']
    method = arguments['-method']
    adv_param_user = ''
    if '-param' in arguments:
        adv_param_user = arguments['-param']
    slices_of_interest = ''
    if '-z' in arguments:
        slices_of_interest = arguments['-z']
    if '-vert' in arguments:
        vertebral_levels = arguments['-vert']
    average_all_labels = param.average_all_labels
    if '-a' in arguments:
        average_all_labels = 1
    fname_output = arguments['-o']
    fname_normalizing_label = ''
    if '-norm-file' in arguments:
        fname_normalizing_label = arguments['-norm-file']
    normalization_method = ''
    if '-norm-method' in arguments:
        normalization_method = arguments['-norm-method']

    # call main function
    main(fname_data, path_label, method, labels_of_interest, slices_of_interest, vertebral_levels, average_all_labels, fname_output, fname_normalizing_label, normalization_method, adv_param_user)
