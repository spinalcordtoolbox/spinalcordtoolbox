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

# TODO: add unit tests: perlevel/perslice, overwrite,
# TODO: check: not sure flag perslice is working 
# TODO: use argparse
# TODO: revisit the flags normalization and weighted mask-- useful?
# TODO: move to csv output. However, we need to change the way z is represented: currently it is a list separated by ,. Maybe we can change it for: ;. e.g.: 0;1;2;3
# TODO: remove fix_label_value() usage because it is used in isolated case and introduces confusion.
# TODO (not urgent): vertebral levels selection should only consider voxels of the selected levels in slices where two different vertebral levels coexist (and not the whole slice)

from __future__ import division, absolute_import

import sys, os, glob, time

import numpy as np

from spinalcordtoolbox.metadata import read_label_file
from spinalcordtoolbox.utils import parse_num_list
from spinalcordtoolbox.template import get_slices_from_vertebral_levels, get_vertebral_level_from_slice

import sct_utils as sct
from spinalcordtoolbox.image import Image
from msct_parser import Parser

# get path of the script and the toolbox
path_script = os.path.dirname(__file__)
path_sct = os.path.dirname(path_script)

# constants
ALMOST_ZERO = 0.000001


class Param:
    def __init__(self):
        self.method = 'wath'
        self.path_label = os.path.join(path_sct, "data", "PAM50", "atlas")
        self.verbose = 1
        self.vertebral_levels = ''
        self.slices_of_interest = ''  # 2-element list corresponding to zmin:zmax. example: '5:8'. For all slices, leave empty.
        self.average_all_labels = 0  # average all labels together after concatenation
        self.fname_output = 'metric_label.txt'
        self.file_info_label = 'info_label.txt'
        self.adv_param = ['10',  # STD of the metric value across labels, in percentage of the mean (mean is estimated using cluster-based ML)
                          '10']  # STD of the assumed gaussian-distributed noise


def get_parser():

    param_default = Param()

    parser = Parser(__file__)
    parser.usage.set_description("""This program extracts metrics (e.g., DTI or MTR) within labels. Labels could be a single file or a folder generated with 'sct_warp_template' and containing multiple label files and a label description file (info_label.txt). The labels should be in the same space coordinates as the input image.""")
    # Mandatory arguments
    parser.add_option(name='-i',
                      type_value='image_nifti',
                      description='File to extract metrics from.',
                      mandatory=True,
                      example='FA.nii.gz')
    # Optional arguments
    parser.add_option(name='-f',
                      type_value='folder',
                      description='Single label file, or folder that contains WM tract labels.',
                      mandatory=False,
                      default_value=os.path.join("label", "atlas"),
                      check_file_exist=False,
                      example=os.path.join(path_sct, 'data', 'atlas'))
    parser.add_option(name='-l',
                      type_value='str',
                      description='Label IDs to extract the metric from. Default = all labels. Separate labels with ",". To select a group of consecutive labels use ":". Example: 1:3 is equivalent to 1,2,3. Maximum Likelihood (or MAP) is computed using all tracts, but only values of the selected tracts are reported.',
                      mandatory=False,
                      default_value='')
    parser.add_option(name='-method',
                      type_value='multiple_choice',
                      description="""Method to extract metrics.
ml: maximum likelihood (only use with well-defined regions and low noise)
  N.B. ONLY USE THIS METHOD WITH THE WHITE MATTER ATLAS! The sum of all tracts should be 1 in all voxels (the algorithm doesn't normalize the atlas).
map: maximum a posteriori. Mean priors are estimated by maximum likelihood within three clusters (white matter, gray matter and CSF). Tract and  noise variance are set with flag -p.
  N.B. ONLY USE THIS METHOD WITH THE WHITE MATTER ATLAS! The sum of all tracts should be 1 in all voxels (the algorithm doesn't normalize the atlas).
wa: weighted average
wath: weighted average (only consider values >0.5)
bin: binarize mask (threshold=0.5)
max: for each z-slice of the input data, extract the max value for each slice of the input data. This mode is useful to extract CSA from an interpolated image (ignore partial volume effect).""",
                      example=['ml', 'map', 'wa', 'wath', 'bin', 'max'],
                      mandatory=False,
                      default_value=param_default.method)
    parser.add_option(name='-overwrite',
                      type_value='int',
                      description="""In the case you choose \".xls\" for the output file extension and you specify a pre-existing output file (see flag \"-o\"),
                      this option will allow you to overwrite this .xls file (\"-overwrite 1\") or to append the results at the end (last line) of the file (\"-overwrite 0\").""",
                      mandatory=False,
                      default_value=0)
    parser.add_option(name='-o',
                      type_value='file_output',
                      description="""File name (including the file extension) of the output result file collecting the metric estimation results. \nThree file types are available: a CSV text file (extension .txt), a MS Excel file (extension .xls) and a pickle file (extension .pickle). Default: """ + param_default.fname_output,
                      mandatory=False,
                      default_value=param_default.fname_output)
    parser.add_option(name='-output-map',
                      type_value='file_output',
                      description="""File name for an image consisting of the atlas labels multiplied by the estimated metric values yielding the metric value map, useful to assess the metric estimation and especially partial volume effects.""",
                      mandatory=False,
                      default_value='')
    parser.add_option(name='-z',
                      type_value='str',
                      description='Slice range to estimate the metric from. First slice is 0. Example: 5:23\nYou can also select specific slices using commas. Example: 0,2,3,5,12',
                      mandatory=False,
                      default_value=param_default.slices_of_interest)
    parser.add_option(name='-perslice',
                      type_value='int',
                      description='Set to 1 to output one metric per slice instead of a single output metric.'
                                  'Please note that when methods ml or map is used, outputing a single '
                                  'metric per slice and then averaging them all is not the same as outputting a single'
                                  'metric at once across all slices.',
                      mandatory=False,
                      default_value=0)
    parser.add_option(name='-vert',
                      type_value='str',
                      description='Vertebral levels to estimate the metric across. Example: 2:9 for C2 to T2.',
                      mandatory=False,
                      example='2:5',
                      default_value=param_default.vertebral_levels)
    parser.add_option(name='-vertfile',
                      type_value='str',  # note: even though it's a file, we cannot put the type='file' otherwise the full path will be added in sct_testing and it will crash
                      description='Vertebral labeling file. Only use with flag -vert',
                      default_value='./label/template/PAM50_levels.nii.gz',
                      mandatory=False)
    parser.add_option(name='-perlevel',
                      type_value='int',
                      description='Set to 1 to output one metric per vertebral level instead of a single '
                                  'output metric.',
                      mandatory=False,
                      default_value=0)
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="1: display on, 0: display off (default)",
                      mandatory=False,
                      example=["0", "1"],
                      default_value="1")
    parser.usage.addSection("\nFOR ADVANCED USERS")
    parser.add_option(name='-param',
                      type_value='str',
                      description="""Advanced parameters for the 'map' method. Separate with comma. All items must be listed (separated with comma).
    #1: standard deviation of metrics across labels
    #2: standard deviation of the noise (assumed Gaussian)""",
                      mandatory=False)
    parser.add_option(name='-p',
                      type_value=None,
                      description="""Advanced parameters for the 'map' method. Separate with comma. All items must be listed (separated with comma).
    #1: standard deviation of metrics across labels
    #2: standard deviation of the noise (assumed Gaussian)""",
                      mandatory=False,
                      deprecated_by='-param')
    parser.add_option(name='-fix-label',
                      type_value=[[','], 'str'],
                      description='When using ML or MAP estimations, if you do not want to estimate the metric in one label and fix its value to avoid effects on other labels, specify <label_ID>,<metric_value. Example to fix the CSF value to 0: -fix-label 36,0.',
                      mandatory=False,
                      default_value='')
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
    parser.add_option(name='-mask-weighted',
                      type_value='image_nifti',
                      description='Nifti mask to weight each voxel during ML or MAP estimation.',
                      example='PAM50_wm.nii.gz',
                      mandatory=False)
    parser.add_option(name='-discard-neg-val',
                      type_value='multiple_choice',
                      mandatory=False,
                      description='Discard voxels with negative value when computing metrics statistics.',
                      example=["0", "1"],
                      default_value="0")

    # read the .txt files referencing the labels
    file_label = os.path.join(param_default.path_label, param_default.file_info_label)
    sct.check_file_exist(file_label, 0)
    default_info_label = open(file_label, 'r')
    label_references = default_info_label.read()
    default_info_label.close()

    str_section = """\n
To list white matter atlas labels:
""" + os.path.basename(__file__) + """ -f """ + os.path.join(path_sct, "data", "atlas") + """

To compute FA within labels 0, 2 and 3 within vertebral levels C2 to C7 using binary method:
""" + os.path.basename(__file__) + """ -i dti_FA.nii.gz -f label/atlas -l 0,2,3 -v 2:7 -m bin"""
    if label_references != '':
        str_section += """

To compute average MTR in a region defined by a single label file (could be binary or 0-1 weighted mask) between slices 1 and 4:
""" + os.path.basename(__file__) + """ -i mtr.nii.gz -f my_mask.nii.gz -z 1:4 -m wa"""
    if label_references != '':
        str_section += """

\nList of labels in """ + file_label + """:
--------------------------------------------------------------------------------------
""" + label_references + """
--------------------------------------------------------------------------------------
"""

    parser.usage.addSection(str_section)

    return parser


def main(fname_data, path_label, method, slices_of_interest, vertebral_levels, fname_output, labels_user, overwrite,
         fname_normalizing_label, normalization_method, label_to_fix, adv_param_user, fname_output_metric_map,
         fname_mask_weight, fname_vertebral_labeling="", perslice=1, perlevel=1, discard_negative_values=False):
    """
    Extract metrics from MRI data based on mask (could be single file of folder to atlas)
    :param fname_data: data to extract metric from
    :param path_label: mask: could be single file or folder to atlas (which contains info_label.txt)
    :param method:
    :param slices_of_interest. Accepted format:
           "0,1,2,3": slices 0,1,2,3
           "0:3": slices 0,1,2,3
    :param vertebral_levels: Vertebral levels to extract metrics from. Should be associated with a template
           (e.g. PAM50/template/) or a specified file: fname_vertebral_labeling. Same format as slices_of_interest.
    :param fname_output:
    :param labels_user:
    :param overwrite:
    :param fname_normalizing_label:
    :param normalization_method:
    :param label_to_fix:
    :param adv_param_user:
    :param fname_output_metric_map:
    :param fname_mask_weight:
    :param fname_vertebral_labeling: vertebral labeling to be used with vertebral_levels
    :param perslice: if user selected several slices, then the function outputs a metric within each slice
           instead of a single average output.
    :param perlevel: if user selected several levels, then the function outputs a metric within each vertebral level
           instead of a single average output.
    :param discard_negative_values: Bool: Discard negative voxels when computing metrics statistics
    :return:
    """

    # Initialization
    verbose = param_default.verbose
    adv_param = param_default.adv_param
    normalizing_label = []
    fixed_label = []
    label_to_fix_fract_vol = None
    im_weight = None

    # check if path_label is a file instead of a folder
    if os.path.isfile(path_label):
        single_label = 1
    elif os.path.isdir(path_label):
        single_label = 0
    else:
        sct.printv('\nERROR: ' + path_label + ' does not exist.', 1, 'error')

    # adjust file names and parameters for old MNI-Poly-AMU template
    if not single_label:
        if not len(glob.glob(os.path.join(path_label, 'WMtract*.*'))) == 0:
            # MNI-Poly-AMU
            suffix_vertebral_labeling = '*_level.nii.gz'
        else:
            # PAM50 and later
            suffix_vertebral_labeling = '*_levels.nii.gz'

    # Find path to the vertebral labeling file if vertebral levels were specified by the user
    if vertebral_levels:
        # check if user selected both specific slices and specific vertebral levels
        if slices_of_interest:
            sct.printv(parser.usage.generate(error='ERROR: You cannot select BOTH vertebral levels AND slice numbers.'))
        # check if user specified folder or single file as label
        if single_label:
            # check if user selected vert but failed to provide a vertebral labeling file
            if not fname_vertebral_labeling:
                sct.printv(
                    '\nYou should indicate a vertebral labeling file with flag -vert.',
                    1, 'error')
        else:
            # if folder is specified, then the vertebral labeling file should be in there. Searching for it...
            fname_vertebral_labeling_list = sct.find_file_within_folder(suffix_vertebral_labeling, os.path.dirname(path_label))
            if len(fname_vertebral_labeling_list) > 1:
                sct.printv(parser.usage.generate(error='ERROR: More than one file named "' + suffix_vertebral_labeling + '" were found in ' + path_label + '. Exit program.'))
            elif len(fname_vertebral_labeling_list) == 0:
                sct.printv(parser.usage.generate(error='ERROR: No file named "' + suffix_vertebral_labeling + '" were found in ' + path_label + '. Exit program.'))
            else:
                fname_vertebral_labeling = os.path.abspath(fname_vertebral_labeling_list[0])

    # Check input parameters
    check_method(method, fname_normalizing_label, normalization_method)

    # parse argument for param
    if not adv_param_user == '':
        adv_param = adv_param_user.replace(' ', '').split(',')  # remove spaces and parse with comma
        del adv_param_user  # clean variable
        # TODO: check integrity of input

    # sct.printv(parameters)
    sct.printv('\nChecked parameters:')
    sct.printv('  data ...................... ' + fname_data)
    sct.printv('  path to label ............. ' + path_label)
    sct.printv('  label ..................... ' + labels_user)
    sct.printv('  method .................... ' + method)
    sct.printv('  slices of interest ........ ' + slices_of_interest)
    sct.printv('  vertebral levels .......... ' + vertebral_levels)
    sct.printv('  vertebral labeling file.... ' + fname_vertebral_labeling)
    sct.printv('  advanced parameters ....... ' + str(adv_param) + '\n')

    # parse labels according to the file info_label.txt
    # note: the "combined_labels_*" is a list of single labels that are defined in the section defined by the keyword
    # "# Keyword=CombinedLabels" in info_label.txt.
    # TODO: redirect to appropriate Sphinx documentation
    if not single_label:
        indiv_labels_ids, indiv_labels_names, indiv_labels_files, \
        combined_labels_ids, combined_labels_names, combined_labels_id_groups, ml_clusters \
            = read_label_file(path_label, param_default.file_info_label)
        # check syntax of labels asked by user
        labels_id_user = check_labels(indiv_labels_ids + combined_labels_ids, parse_num_list(labels_user))
    else:
        indiv_labels_ids = [0]
        labels_id_user = [0]
        indiv_labels_names = [path_label]
        indiv_labels_files = [path_label]
        combined_labels_ids = []
        combined_labels_names = []
        combined_labels_id_groups = []
        ml_clusters = []
        # set path_label to empty string, because indiv_labels_files will replace it from now on
        path_label = ''
    nb_labels = len(indiv_labels_files)

    # Load data and systematically reorient to RPI because we need the 3rd dimension to be z
    sct.printv('\nLoad metric image...', verbose)
    input_im = Image(fname_data).change_orientation("RPI")

    data = input_im.data
    # Load labels
    labels = np.empty([nb_labels], dtype=object)
    for i_label in range(nb_labels):
        im_label = Image(os.path.join(path_label, indiv_labels_files[i_label])).change_orientation("RPI")
        labels[i_label] = im_label.data
    # Load vertebral levels
    if vertebral_levels:
        im_vertebral_labeling = Image(fname_vertebral_labeling).change_orientation("RPI")
    # if the "normalization" option is wanted,
    if fname_normalizing_label:
        normalizing_label = np.empty([1], dtype=object)  # choose this kind of structure so as to keep easily the compatibility with the rest of the code (dimensions: (1, x, y, z))
        im_normalizing_label = Image(fname_normalizing_label).change_orientation("RPI")
        normalizing_label[0] = im_normalizing_label.data
    # if flag "-mask-weighted" is specified
    if fname_mask_weight:
        im_weight = Image(fname_mask_weight).change_orientation("RPI")

    # Change metric data type into floats for future manipulations (normalization)
    data = np.float64(data)
    # loop across labels and set voxel to zero if...
    for i_label in range(nb_labels):
        labels[i_label][np.isneginf(data)] = 0  # ...data voxel is -inf
        labels[i_label][np.isnan(data)] = 0  # ...data voxel is nan
        if discard_negative_values:
            labels[i_label][data < 0.0] = 0  # ...data voxel is negative
        labels[i_label][np.isposinf(data)] = 0  # ...data voxel is +inf

    # Get dimensions of data and labels
    nx, ny, nz = data.shape
    nx_atlas, ny_atlas, nz_atlas = labels[0].shape

    # Check dimensions consistency between atlas and data
    if (nx, ny, nz) != (nx_atlas, ny_atlas, nz_atlas):
        sct.printv('\nERROR: Metric data and labels DO NOT HAVE SAME DIMENSIONS.')
        sys.exit(2)

    # parse clusters used for a priori (map method)
    clusters_all_labels = ml_clusters
    combined_labels_groups_all_IDs = combined_labels_id_groups

    # If specified, remove the label to fix its value
    if label_to_fix:
        data, labels, indiv_labels_ids, indiv_labels_names, clusters_all_labels, combined_labels_groups_all_IDs, labels_id_user, label_to_fix_name, label_to_fix_fract_vol = fix_label_value(label_to_fix, data, labels, indiv_labels_ids, indiv_labels_names, clusters_all_labels, combined_labels_groups_all_IDs, labels_id_user)

    if slices_of_interest:
        slices_list = parse_num_list(slices_of_interest)
    else:
        slices_list = np.arange(nz).tolist()

    # if perslice with slices: ['1', '2', '3', '4']
    # important: each slice number should be separated by "," not ":"
    slicegroups = [str(i) for i in slices_list]
    if not perslice and not perlevel:
        # ['1,2,3,4,5,6']
        slicegroups = [';'.join(slicegroups)]

    # if user selected vertebral levels and asked for each separate levels
    # slicegroups = ['1,2', '3,4']
    if vertebral_levels:
        list_levels = parse_num_list(vertebral_levels)
        # Re-define slices_of_interest according to the vertebral levels selected by user
        slices_of_interest = []
        for level in list_levels:
            slices_of_interest.append(get_slices_from_vertebral_levels(im_vertebral_labeling, level))
        # convert to comma-separated list for each level
        slicegroups = []
        for group in slices_of_interest:
            # for each group: [1, 2, 3, 4] --> ['1,2,3,4']
            # so that slicegroups looks like: ['1,2,3,4','5,6,7,8','9,10,11,12']
            slicegroups.append([';'.join([str(i) for i in group])][0])

        if not perlevel:
            # if user wants to concatenate all slices of interest into a single slicegroups
            slicegroups = [";".join(slicegroups)]
            if perslice:
                # if user wants to get metric per individual slice
                slicegroups = slicegroups[0].split(';')
    # loop across slicegroups
    first_pass = True
    for slicegroup in slicegroups:
        if overwrite and first_pass:
            overwrite_tmp = 1  # overwrite
        else:
            overwrite_tmp = 0
        try:
            # convert list of strings into list of int to use as index
            ind_slicegroup = [int(i) for i in slicegroup.split(';')]
            # select portion of data and labels based on slicegroup
            dataz = data[:, :, ind_slicegroup]
            labelsz = np.copy(labels)
            for i_label in range(0, nb_labels):
                labelsz[i_label] = labels[i_label][:, :, ind_slicegroup]
            # Extract metric in the labels specified by the file info_label.txt from the atlas folder given in input
            # TODO: instead of estimating everything (all labels + combined labels), only compute what is asked by the user
            # individual labels
            indiv_labels_value, indiv_labels_std, indiv_labels_fract_vol = \
                extract_metric(method, dataz, labelsz, indiv_labels_ids, clusters_all_labels, adv_param, normalizing_label,
                               normalization_method, im_weight=im_weight)
            # combined labels
            combined_labels_value = np.zeros(len(combined_labels_groups_all_IDs), dtype=float)
            combined_labels_std = np.zeros(len(combined_labels_groups_all_IDs), dtype=float)
            combined_labels_fract_vol = np.zeros(len(combined_labels_groups_all_IDs), dtype=float)
            for i_combined_labels in range(0, len(combined_labels_groups_all_IDs)):
                combined_labels_value[i_combined_labels], \
                combined_labels_std[i_combined_labels], \
                combined_labels_fract_vol[i_combined_labels] = extract_metric(method, dataz, labelsz, indiv_labels_ids,
                                                                              clusters_all_labels, adv_param,
                                                                              normalizing_label, normalization_method,
                                                                              im_weight=im_weight,
                                                                              combined_labels_id_group=combined_labels_groups_all_IDs[i_combined_labels])
            # TODO: remove that crap below at some point (check for dependencies, usage, etc.)
            if label_to_fix:
                fixed_label = [label_to_fix[0], label_to_fix_name, label_to_fix[1]]
                sct.printv('\n*' + fixed_label[0] + ', ' + fixed_label[1] + ': ' + fixed_label[2] + ' (value fixed by user)', 1, 'info')

            # deal with output display
            if vertebral_levels:
                if perlevel:
                    vert_levels = list_levels[slicegroups.index(slicegroup)]
                elif perslice:
                    vert_levels = get_vertebral_level_from_slice(im_vertebral_labeling, ind_slicegroup[0])
                else:
                    vert_levels = list_levels
                    # replace "," with ";" for easier CSV parsing
                if isinstance(vert_levels, int):
                    vert_levels = str(vert_levels)
                else:
                    vert_levels = ';'.join([str(level) for level in vert_levels])
            else:
                vert_levels = 'Unknown'

        except ValueError:
            # the slice request is out of the range of the image
            sct.printv('The slice(s) requested is out of the range of the image', type='warning')

        # write metrics into file
        save_metrics(labels_id_user, indiv_labels_ids, combined_labels_ids, indiv_labels_names, combined_labels_names,
                     slicegroup, indiv_labels_value, indiv_labels_std, indiv_labels_fract_vol,
                     combined_labels_value, combined_labels_std, combined_labels_fract_vol, fname_output, fname_data,
                     method, overwrite_tmp, fname_normalizing_label, fixed_label, vert_levels=vert_levels)
        first_pass = False  # now we can systematically overwrite

        # display results
        # TODO: simply print out the created csv file when we switch to csv output
        sct.printv('\nResults:\nID, label name [total fractional volume of the label in number of voxels]:    metric value +/- metric STDEV within label', 1)
        for i_label_user in labels_id_user:
            if i_label_user <= max(indiv_labels_ids):
                index = indiv_labels_ids.index(i_label_user)
                sct.printv(str(indiv_labels_ids[index]) + ', ' + str(indiv_labels_names[index]) + ' [' + str(np.round(indiv_labels_fract_vol[index], 2)) + ']:    ' + str(indiv_labels_value[index]) + ' +/- ' + str(indiv_labels_std[index]), 1, 'info')
            elif i_label_user > max(indiv_labels_ids):
                index = combined_labels_ids.index(i_label_user)
                sct.printv(str(combined_labels_ids[index]) + ', ' + str(combined_labels_names[index]) + ' [' + str(np.round(combined_labels_fract_vol[index], 2)) + ']:    ' + str(combined_labels_value[index]) + ' +/- ' + str(combined_labels_std[index]), 1, 'info')

    # output a metric value map
    if fname_output_metric_map:
        data_metric_map = generate_metric_value_map(fname_output_metric_map, input_im, labels, indiv_labels_value, slices_list, label_to_fix, label_to_fix_fract_vol)


def extract_metric(method, data, labels, indiv_labels_ids, clusters_labels='', adv_param='', normalizing_label=[], normalization_method='', im_weight='', combined_labels_id_group='', verbose=0):
    """Extract metric in the labels specified by the file info_label.txt in the atlas folder."""

    # Initialization to default values
    clustered_labels, matching_cluster_labels = [], []

    nb_labels_total = len(indiv_labels_ids)

    # check consistency of label input parameter (* LOI=Labels of Interest)
    list_ids_LOI = check_labels(indiv_labels_ids, combined_labels_id_group)  # If 'labels_of_interest' is empty, then label_id_user' contains the index of all labels in the file info_label.txt

    if method == 'map':
        # get clustered labels
        clustered_labels, matching_cluster_labels = get_clustered_labels(clusters_labels, labels, indiv_labels_ids, list_ids_LOI, combined_labels_id_group, verbose)

    # if user wants to get unique value across labels, then combine all labels together
    if combined_labels_id_group:
        sum_combined_labels = np.sum(labels[list_ids_LOI])  # sum the labels selected by user
        if method == 'ml' or method == 'map':  # in case the maximum likelihood and the average across different labels are wanted
            # merge labels
            labels_tmp = np.empty([nb_labels_total - len(list_ids_LOI) + 1], dtype=object)
            labels = np.delete(labels, list_ids_LOI)  # remove the labels selected by user
            labels_tmp[0] = sum_combined_labels  # put the sum of the labels selected by user in first position of the tmp variable
            for i_label in range(1, len(labels_tmp)):
                labels_tmp[i_label] = labels[i_label - 1]  # fill the temporary array with the values of the non-selected labels
            labels = labels_tmp  # replace the initial labels value by the updated ones (with the summed labels)
            del labels_tmp  # delete the temporary labels

        else:  # in other cases than the maximum likelihood, we can remove other labels (not needed for estimation)
            labels = np.empty(1, dtype=object)
            labels[0] = sum_combined_labels  # we create a new label array that includes only the summed labels

    if normalizing_label:  # if the "normalization" option is wanted
        sct.printv('\nExtract normalization values...', verbose)
        if normalization_method == 'sbs':  # case: the user wants to normalize slice-by-slice
            for z in range(0, data.shape[-1]):
                normalizing_label_slice = np.empty([1], dtype=object)  # in order to keep compatibility with the function
                # 'extract_metric_within_tract', define a new array for the slice z of the normalizing labels
                normalizing_label_slice[0] = normalizing_label[0][..., z]
                metric_normalizing_label = estimate_metric_within_tract(data[..., z], normalizing_label_slice, method, 0)
                # estimate the metric mean in the normalizing label for the slice z
                if metric_normalizing_label[0][0] != 0:
                    data[..., z] = data[..., z] / metric_normalizing_label[0][0]  # divide all the slice z by this value

        elif normalization_method == 'whole':  # case: the user wants to normalize after estimations in the whole labels
            metric_norm_label, metric_std_norm_label = estimate_metric_within_tract(data, normalizing_label, method, param_default.verbose)  # mean and std are lists

    # extract metrics within labels
    sct.printv('\nEstimate metric within labels...', verbose)
    metric_in_labels, metric_std_in_labels = estimate_metric_within_tract(data, labels, method, verbose, clustered_labels, matching_cluster_labels, adv_param, im_weight)  # mean and std are lists

    if normalizing_label and normalization_method == 'whole':  # case: user wants to normalize after estimations in the whole labels
        metric_in_labels, metric_std_in_labels = np.divide(metric_in_labels, metric_norm_label), np.divide(metric_std_in_labels, metric_std_norm_label)

    if combined_labels_id_group:
        metric_in_labels = np.asarray([metric_in_labels[0]])
        metric_std_in_labels = np.asarray([metric_std_in_labels[0]])

    # compute fractional volume for each label
    fract_vol_per_label = np.zeros(metric_in_labels.size, dtype=float)
    for i_label in range(0, metric_in_labels.size):
        fract_vol_per_label[i_label] = np.sum(labels[i_label])

    return metric_in_labels, metric_std_in_labels, fract_vol_per_label


def remove_slices(data_to_crop, slices_of_interest):
    """Crop data to only keep the slices asked by user."""
    # Parse numbers based on delimiter: ' or :
    slices_list = parse_num_list(slices_of_interest)
    # Remove slices that are not wanted (+1 is to include the last selected slice as Python "includes -1"
    data_cropped = data_to_crop[..., slices_list]
    return data_cropped, slices_list


def save_metrics(labels_id_user, indiv_labels_ids, combined_labels_ids, indiv_labels_names, combined_labels_names,
                 slices_of_interest, indiv_labels_value, indiv_labels_std, indiv_labels_fract_vol,
                 combined_labels_value, combined_labels_std, combined_labels_fract_vol, fname_output, fname_data,
                 method, overwrite, fname_normalizing_label, fixed_label=None, vert_levels='Unknown'):
    """
    Save results in the output type selected by user
    :param labels_id_user:
    :param indiv_labels_ids:
    :param combined_labels_ids:
    :param indiv_labels_names:
    :param combined_labels_names:
    :param slices_of_interest: str
    :param indiv_labels_value:
    :param indiv_labels_std:
    :param indiv_labels_fract_vol:
    :param combined_labels_value:
    :param combined_labels_std:
    :param combined_labels_fract_vol:
    :param fname_output:
    :param fname_data:
    :param method:
    :param overwrite:
    :param fname_normalizing_label:
    :param fixed_label:
    :param vert_levels: str
    :return:
    """

    sct.printv('\nSaving results in: ' + fname_output + ' ...')

    # Note: Because of the pressing issue #1963 and the current refactoring of metric_saving (see PR #1931), a quick-
    # -and-dirty workaround here is to always save as xsl file, and if user asked for a .txt file, then the .xls will
    # be converted to a txt.
    output_path, output_file, output_type = sct.extract_fname(fname_output)
    fname_output_xls = os.path.join(output_path, output_file + '.xls')

    # if the user asked for no overwriting but the specified output file does not exist yet
    if (not overwrite) and (not os.path.isfile(fname_output_xls)):
        sct.printv('WARNING: You asked to edit the pre-existing file \"' + fname_output + '\" but this file does not exist. It will be created.', type='warning')
        overwrite = 1

    if not overwrite:
        from xlrd import open_workbook
        from xlutils.copy import copy

        existing_book = open_workbook(fname_output_xls)

        # get index of the first empty row and leave one empty row between the two subjects
        row_index = existing_book.sheet_by_index(0).nrows

        book = copy(existing_book)
        sh = book.get_sheet(0)

    elif overwrite:
        from xlwt import Workbook

        book = Workbook()
        sh = book.add_sheet('Results', cell_overwrite_ok=True)

        # write header line
        sh.write(0, 0, 'Date - Time')
        sh.write(0, 1, 'Metric file')
        sh.write(0, 2, 'Extraction method')
        sh.write(0, 3, 'Vertebral levels')
        sh.write(0, 4, 'Slices (z)')
        sh.write(0, 5, 'ID')
        sh.write(0, 6, 'Label name')
        sh.write(0, 7, 'Total fractional volume of the label (in number of voxels)')
        sh.write(0, 8, 'Metric value')
        sh.write(0, 9, 'Metric STDEV within label')
        if fname_normalizing_label:
            sh.write(0, 10, 'Label used to normalize the metric estimation slice-by-slice')

        row_index = 1

    # iterate on user's labels
    # TODO: this should be done outside of this function
    for i_label_user in labels_id_user:
        try:
            sh.write(row_index, 0, time.strftime('%Y/%m/%d - %H:%M:%S'))
            sh.write(row_index, 1, os.path.abspath(fname_data))
            sh.write(row_index, 2, method)
            sh.write(row_index, 3, vert_levels)
            sh.write(row_index, 4, slices_of_interest)
            if fname_normalizing_label:
                sh.write(row_index, 10, fname_normalizing_label)

            # display result for this label
            if i_label_user <= max(indiv_labels_ids):
                index = indiv_labels_ids.index(i_label_user)
                sh.write(row_index, 5, indiv_labels_ids[index])
                sh.write(row_index, 6, indiv_labels_names[index])
                sh.write(row_index, 7, indiv_labels_fract_vol[index])
                sh.write(row_index, 8, indiv_labels_value[index])
                sh.write(row_index, 9, indiv_labels_std[index])
            elif i_label_user > max(indiv_labels_ids):
                index = combined_labels_ids.index(i_label_user)
                sh.write(row_index, 5, combined_labels_ids[index])
                sh.write(row_index, 6, combined_labels_names[index])
                sh.write(row_index, 7, combined_labels_fract_vol[index])
                sh.write(row_index, 8, combined_labels_value[index])
                sh.write(row_index, 9, combined_labels_std[index])
        except TypeError:
            # out of range. Ignore
            break

        row_index += 1

    if fixed_label:
        sh.write(row_index, 0, time.strftime('%Y/%m/%d - %H:%M:%S'))
        sh.write(row_index, 1, os.path.abspath(fname_data))
        sh.write(row_index, 2, method)
        sh.write(row_index, 3, vert_levels)
        sh.write(row_index, 4, slices_of_interest)
        if fname_normalizing_label:
            sh.write(row_index, 10, fname_normalizing_label)

        sh.write(row_index, 5, int(fixed_label[0]))
        sh.write(row_index, 6, fixed_label[1])
        sh.write(row_index, 7, 'nan')
        sh.write(row_index, 8, '*' + fixed_label[2] + ' (value fixed by user)')
        sh.write(row_index, 9, 'nan')

    book.save(fname_output_xls)

    # if the user chose to output results under a .txt file
    if output_type == '.txt':
        # simply convert the XLS into TXT (see comment above)
        import pandas as pd
        data_xls = pd.read_excel(fname_output_xls, index_col=None)
        # add "#" to first column element because this is going to be the header
        columns = data_xls.columns.tolist()
        columns[0] = "#" + columns[0]
        data_xls.columns = columns
        data_xls.to_csv(fname_output, encoding='utf-8', index=False)
        # # CSV format, header lines start with "#"
        #
        # # Write mode of file
        # fid_metric = open(fname_output, 'w')
        #
        # # WRITE HEADER:
        # # Write date and time
        # fid_metric.write('# Date - Time: ' + time.strftime('%Y/%m/%d - %H:%M:%S'))
        # # Write metric data file path
        # fid_metric.write('\n' + '# Metric file: ' + os.path.abspath(fname_data))
        # # If it's the case, write the label used to normalize the metric estimation:
        # if fname_normalizing_label:
        #     fid_metric.write('\n' + '# Label used to normalize the metric estimation slice-by-slice: ' + fname_normalizing_label)
        # # Write method used for the metric estimation
        # fid_metric.write('\n' + '# Extraction method: ' + method)
        #
        # # Write selected vertebral levels
        # fid_metric.write('\n# Vertebral levels: ' + vert_levels)
        #
        # # Write selected slices
        # fid_metric.write('\n' + '# Slices (z): ' + slices_of_interest)
        #
        # # label headers
        # fid_metric.write('%s' % ('\n' + '# ID, label name, total fractional volume of the label (in number of voxels), metric value, metric stdev within label\n\n'))
        #
        # # WRITE RESULTS
        # labels_id_user.sort()
        # section = ''
        # if labels_id_user[0] <= max(indiv_labels_ids):
        #     section = '\n# White matter atlas\n'
        # elif labels_id_user[0] > max(indiv_labels_ids):
        #     section = '\n# Combined labels\n'
        #     fid_metric.write(section)
        # for i_label_user in labels_id_user:
        #     # change section if not individual label anymore
        #     if i_label_user > max(indiv_labels_ids) and section == '\n# White matter atlas\n':
        #         section = '\n# Combined labels\n'
        #         fid_metric.write(section)
        #     # display result for this label
        #     if section == '\n# White matter atlas\n':
        #         index = indiv_labels_ids.index(i_label_user)
        #         fid_metric.write('%i, %s, %f, %f, %f\n' % (indiv_labels_ids[index], indiv_labels_names[index], indiv_labels_fract_vol[index], indiv_labels_value[index], indiv_labels_std[index]))
        #     elif section == '\n# Combined labels\n':
        #         index = combined_labels_ids.index(i_label_user)
        #         fid_metric.write('%i, %s, %f, %f, %f\n' % (combined_labels_ids[index], combined_labels_names[index], combined_labels_fract_vol[index], combined_labels_value[index], combined_labels_std[index]))
        #
        # if fixed_label:
        #     fid_metric.write('\n*' + fixed_label[0] + ', ' + fixed_label[1] + ': ' + fixed_label[2] + ' (value fixed by user)')
        #
        # # Close file .txt
        # fid_metric.close()


    # if user chose to output results under a pickle file (variables that can be loaded in a python environment)
    elif output_type == '.pickle':

        # write results in a dictionary
        metric_extraction_results = {}

        metric_extraction_results['Date - Time'] = time.strftime('%Y/%m/%d - %H:%M:%S')
        metric_extraction_results['Metric file'] = os.path.abspath(fname_data)
        metric_extraction_results['Extraction method'] = method
        metric_extraction_results['Vertebral levels'] = vert_levels
        metric_extraction_results['Slices (z)'] = slices_of_interest
        if fname_normalizing_label:
            metric_extraction_results['Label used to normalize the metric estimation slice-by-slice'] = fname_normalizing_label

        # keep only the labels selected by user (flag -l)
        ID_field = []
        Label_names_field = []
        Fract_vol_field = []
        Metric_value_field = []
        Metric_std_field = []
        # iterate on user's labels
        for i_label_user in labels_id_user:
            # display result for this label
            if i_label_user <= max(indiv_labels_ids):
                index = indiv_labels_ids.index(i_label_user)
                ID_field.append(indiv_labels_ids[index])
                Label_names_field.append(indiv_labels_names[index])
                Fract_vol_field.append(indiv_labels_fract_vol[index])
                Metric_value_field.append(indiv_labels_value[index])
                Metric_std_field.append(indiv_labels_std[index])
            elif i_label_user > max(indiv_labels_ids):
                index = combined_labels_ids.index(i_label_user)
                ID_field.append(combined_labels_ids[index])
                Label_names_field.append(combined_labels_names[index])
                Fract_vol_field.append(combined_labels_fract_vol[index])
                Metric_value_field.append(combined_labels_value[index])
                Metric_std_field.append(combined_labels_std[index])

        metric_extraction_results['ID'] = np.array(ID_field)
        metric_extraction_results['Label name'] = np.array(Label_names_field)
        metric_extraction_results['Total fractional volume of the label (in number of voxels)'] = np.array(Fract_vol_field)
        metric_extraction_results['Metric value'] = np.array(Metric_value_field)
        metric_extraction_results['Metric STDEV within label'] = np.array(Metric_std_field)
        if fixed_label:
            metric_extraction_results['Fixed label'] = 'Label ID = ' + fixed_label[0] + ', Label name = ' + fixed_label[1] + ', Value (set by user) = ' + fixed_label[2]

        # save results into a pickle file
        import pickle
        output_file = open(fname_output, 'wb')
        pickle.dump(metric_extraction_results, output_file)
        output_file.close()

    else:
        sct.printv('WARNING: The file extension for the output result file that was specified was not recognized. No result file will be created.', type='warning')

    sct.printv('\tDone.')


def check_method(method, fname_normalizing_label, normalization_method):
    """Check the consistency of the methods asked by the user."""

    # THIS BELOW IS ALREADY CHECKED BY THE PARSER SO I COMMENTED IT. jcohenadad 2016-10-23
    # if (method != 'wa') & (method != 'ml') & (method != 'bin') & (method != 'wath') & (method != 'map'):
    #     sct.printv(parser.usage.generate(error='ERROR: Method "' + method + '" is not correct. See help. Exit program.\n'))

    if normalization_method and not fname_normalizing_label:
        sct.printv(parser.usage.generate(error='ERROR: You selected a normalization method (' + str(normalization_method) + ') but you didn\'t selected any label to be used for the normalization.'))

    if fname_normalizing_label and normalization_method != 'sbs' and normalization_method != 'whole':
        sct.printv(parser.usage.generate(error='\nERROR: The normalization method you selected is incorrect:' + str(normalization_method)))


def check_labels(indiv_labels_ids, selected_labels):
    """Check the consistency of the labels asked by the user."""

    # TODO: allow selection of combined labels as "36, Ventral, 7:14,22:19"

    # convert strings to int
    list_ids_of_labels_of_interest = list(map(int, indiv_labels_ids))

    # if selected_labels:
    #     # Check if label chosen is in the right format
    #     for char in selected_labels:
    #         if not char in '0123456789,:':
    #             sct.printv(parser.usage.generate(error='\nERROR: ' + selected_labels + ' is not the correct format to select combined labels.\n Exit program.\n'))
    #
    #     if ':' in selected_labels:
    #         label_ids_range = [int(x) for x in selected_labels.split(':')]
    #         if len(label_ids_range) > 2:
    #             sct.printv(parser.usage.generate(error='\nERROR: Combined labels ID selection must be in format X:Y, with X and Y between 0 and 31.\nExit program.\n\n'))
    #         else:
    #             label_ids_range.sort()
    #             list_ids_of_labels_of_interest = [int(x) for x in range(label_ids_range[0], label_ids_range[1]+1)]
    #
    #     else:
    #         list_ids_of_labels_of_interest = [int(x) for x in selected_labels.split(',')]

    if selected_labels:
        # Remove redundant values
        list_ids_of_labels_of_interest = [i_label for n, i_label in enumerate(selected_labels) if i_label not in selected_labels[:n]]

        # Check if the selected labels are in the available labels ids
        if not set(list_ids_of_labels_of_interest).issubset(set(indiv_labels_ids)):
            sct.printv('\nERROR: At least one of the selected labels (' + str(list_ids_of_labels_of_interest) + ') is not available according to the label list from the text file in the atlas folder. Exit program.\n\n', type='error')

    return list_ids_of_labels_of_interest


def estimate_metric_within_tract(data, labels, method, verbose, clustered_labels=[], matching_cluster_labels=[], adv_param=[], im_weight=None):
    """Extract metric within labels.
    :data: (nx,ny,nz) numpy array
    :labels: nlabel tuple of (nx,ny,nz) array
    """

    nb_labels = len(labels)  # number of labels

    # if user asks for binary regions, binarize atlas
    if method == 'bin' or method == 'max':
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

    if method == 'map' or 'ml':
        # if specified (flag -mask-weighted), define a matrix to weight voxels. If not, this matrix is set to identity.
        if im_weight:
            data_weight_1d = im_weight.data[ind_positive]
        else:
            data_weight_1d = np.ones(nb_vox)
        W = np.diag(data_weight_1d)  # weight matrix

    # Display number of non-zero values
    sct.printv('  Number of non-null voxels: ' + str(nb_vox), verbose=verbose)

    # initialization
    metric_mean = np.empty([nb_labels], dtype=object)
    metric_std = np.empty([nb_labels], dtype=object)

    # Estimation with maximum a posteriori (map)
    if method == 'map':

        # ML estimation in the defined clusters to get a priori
        # -----------------------------------------------------

        sct.printv('Maximum likelihood estimation within the selected clusters to get a priori for the MAP estimation...', verbose=verbose)

        nb_clusters = len(clustered_labels)

        #  Select non-zero values in the union of the clustered labels
        clustered_labels_sum = np.sum(clustered_labels)
        ind_positive_clustered_labels = clustered_labels_sum > ALMOST_ZERO  # labels_sum > ALMOST_ZERO

        # define the problem to apply the maximum likelihood to clustered labels
        y_apriori = data[ind_positive_clustered_labels]  # [nb_vox x 1]

        # create matrix X to use ML and estimate beta_0
        x_apriori = np.zeros([len(y_apriori), nb_clusters])
        for i_cluster in range(nb_clusters):
            x_apriori[:, i_cluster] = clustered_labels[i_cluster][ind_positive_clustered_labels]

        # remove unused voxels from the weighting matrix W
        if im_weight:
            data_weight_1d_apriori = im_weight.data[ind_positive_clustered_labels]
        else:
            data_weight_1d_apriori = np.ones(np.sum(ind_positive_clustered_labels))
        W_apriori = np.diag(data_weight_1d_apriori)  # weight matrix

        # apply the weighting matrix
        y_apriori = np.dot(W_apriori, y_apriori)
        x_apriori = np.dot(W_apriori, x_apriori)

        # estimate values using ML for each cluster
        beta = np.dot(np.linalg.pinv(np.dot(x_apriori.T, x_apriori)), np.dot(x_apriori.T, y_apriori))  # beta = (Xt . X)-1 . Xt . y
        # display results
        sct.printv('  Estimated beta0 per cluster: ' + str(beta), verbose=verbose)

        # MAP estimations within the selected labels
        # ------------------------------------------

        # perc_var_label = int(adv_param[0])^2  # variance within label, in percentage of the mean (mean is estimated using cluster-based ML)
        var_label = int(adv_param[0]) ^ 2  # variance within label
        var_noise = int(adv_param[1]) ^ 2  # variance of the noise (assumed Gaussian)

        # define the problem: y is the measurements vector (to which weights are applied, to each voxel) and x is the linear relation between the measurements y and the true metric value to be estimated beta
        y = np.dot(W, data1d)  # [nb_vox x 1]
        x = np.dot(W, labels2d.T)  # [nb_vox x nb_labels]
        # construct beta0
        beta0 = np.zeros(nb_labels)
        for i_cluster in range(nb_clusters):
            beta0[np.where(np.asarray(matching_cluster_labels) == i_cluster)[0]] = beta[i_cluster]
        # construct covariance matrix (variance between tracts). For simplicity, we set it to be the identity.
        Rlabel = np.diag(np.ones(nb_labels))
        A = np.linalg.pinv(np.dot(x.T, x) + np.linalg.pinv(Rlabel) * var_noise / var_label)
        B = x.T
        C = y - np.dot(x, beta0)
        beta = beta0 + np.dot(A, np.dot(B, C))
        for i_label in range(0, nb_labels):
            metric_mean[i_label] = beta[i_label]
            metric_std[i_label] = 0  # need to assign a value for writing output file

    # clear memory
    del data, labels

    # Estimation with maximum likelihood
    if method == 'ml':
        # define the problem: y is the measurements vector (to which weights are applied, to each voxel) and x is the linear relation between the measurements y and the true metric value to be estimated beta
        y = np.dot(W, data1d)  # [nb_vox x 1]
        x = np.dot(W, labels2d.T)  # [nb_vox x nb_labels]
        beta = np.dot(np.linalg.pinv(np.dot(x.T, x)), np.dot(x.T, y))  # beta = (Xt . X)-1 . Xt . y
        #beta, residuals, rank, singular_value = np.linalg.lstsq(np.dot(x.T, x), np.dot(x.T, y), rcond=-1)
        #beta, residuals, rank, singular_value = np.linalg.lstsq(x, y)
        # sct.printv(beta, residuals, rank, singular_value)
        for i_label in range(0, nb_labels):
            metric_mean[i_label] = beta[i_label]
            metric_std[i_label] = 0  # need to assign a value for writing output file

    # Estimation with weighted average (also works for binary)
    if method == 'wa' or method == 'bin' or method == 'wath' or method == 'max':
        for i_label in range(0, nb_labels):
            # check if all labels are equal to zero
            if sum(labels2d[i_label, :]) == 0:
                sct.printv('WARNING: labels #' + str(i_label) + ' contains only null voxels. Mean and std are set to 0.')
                metric_mean[i_label] = 0
                metric_std[i_label] = 0
            else:
                if method == 'max':
                    # just take the max within the mask
                    metric_mean[i_label] = max(data1d * labels2d[i_label, :])
                    metric_std[i_label] = 0  # set to 0, although this value is irrelevant here
                else:
                    # estimate the weighted average
                    metric_mean[i_label] = sum(data1d * labels2d[i_label, :]) / sum(labels2d[i_label, :])
                    # estimate the biased weighted standard deviation
                    metric_std[i_label] = np.sqrt(
                        sum(labels2d[i_label, :] * (data1d - metric_mean[i_label]) ** 2) / sum(labels2d[i_label, :]))

    return metric_mean, metric_std


def get_clustered_labels(clusters_all_labels, labels, indiv_labels_ids, labels_user, averaging_flag, verbose):
    """
    Cluster labels according to selected options (labels and averaging).
    :ml_clusters: clusters in form: '0:29,30,31'
    :labels: all labels data
    :labels_user: label IDs selected by the user
    :averaging_flag: flag -a (0 or 1)
    :return: clustered_labels: labels summed by clustered
    """

    nb_clusters = len(clusters_all_labels)

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
    sct.printv('  Number of clusters: ' + str(nb_clusters), verbose=verbose)

    # sum labels within each cluster
    clustered_labels = np.empty([nb_clusters], dtype=object)  # labels(nb_labels_total, x, y, z)
    for i_cluster in range(0, nb_clusters):
        indexes_labels_cluster_i = [indiv_labels_ids.index(label_ID) for label_ID in clusters_all_labels[i_cluster]]
        clustered_labels[i_cluster] = np.sum(labels[indexes_labels_cluster_i])

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


def fix_label_value(label_to_fix, data, labels, indiv_labels_ids, indiv_labels_names, ml_clusters, combined_labels_id_groups, labels_id_user):
    """
    This function updates the data and list of labels as explained in:
    https://github.com/neuropoly/spinalcordtoolbox/issues/958
    :param label_to_fix:
    :param data:
    :param labels:
    :param indiv_labels_ids:
    :param indiv_labels_names:
    :param ml_clusters:
    :param combined_labels_id_groups:
    :param labels_id_user:
    :return:
    """

    label_to_fix_ID = int(label_to_fix[0])
    label_to_fix_value = float(label_to_fix[1])

    # remove the value from the data
    label_to_fix_index = indiv_labels_ids.index(label_to_fix_ID)
    label_to_fix_fract_vol = labels[label_to_fix_index]
    data = data - label_to_fix_fract_vol * label_to_fix_value

    # remove the label to fix from the labels lists
    labels = np.delete(labels, label_to_fix_index, 0)
    del indiv_labels_ids[label_to_fix_index]
    label_to_fix_name = indiv_labels_names[label_to_fix_index]
    del indiv_labels_names[label_to_fix_index]

    # remove the label to fix from the label list specified by user
    if label_to_fix_ID in labels_id_user:
        labels_id_user.remove(label_to_fix_ID)

    # redefine the clusters
    ml_clusters = remove_label_from_group(ml_clusters, label_to_fix_ID)

    # redefine the combined labels groups
    combined_labels_id_groups = remove_label_from_group(combined_labels_id_groups, label_to_fix_ID)

    return data, labels, indiv_labels_ids, indiv_labels_names, ml_clusters, combined_labels_id_groups, labels_id_user, label_to_fix_name, label_to_fix_fract_vol


def remove_label_from_group(list_label_groups, label_ID):
    """Redefine groups of labels after removing one specific label."""

    for i_group in range(len(list_label_groups)):
        if label_ID in list_label_groups[i_group]:
            list_label_groups[i_group].remove(label_ID)

    list_label_groups = list(filter(None, list_label_groups))

    return list_label_groups


def generate_metric_value_map(fname_output_metric_map, input_im, labels, indiv_labels_value, slices_list, label_to_fix, label_to_fix_fract_vol):
    """Produces a map where each label is assigned the metric value estimated previously based on their fractional volumes."""

    sct.printv('\nGenerate metric value map based on each label fractional volumes: ' + fname_output_metric_map + '...')

    # initialize metric value map with zeros
    metric_map = input_im
    metric_map.data = np.zeros(input_im.data.shape)

    # assign to each label the corresponding estimated metric value
    for i_label in range(len(labels)):
        metric_map.data[:, :, slices_list] = metric_map.data[:, :, slices_list] + labels[i_label] * indiv_labels_value[i_label]

    if label_to_fix:
        metric_map.data[:, :, slices_list] = metric_map.data[:, :, slices_list] + label_to_fix_fract_vol * float(label_to_fix[1])

    # save metric value map
    metric_map.save(fname_output_metric_map)

    sct.printv('\tDone.')
    return metric_map


# =======================================================================================================================
# Start program
# =======================================================================================================================
if __name__ == "__main__":
    sct.init_sct()

    param_default = Param()

    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])

    overwrite = 0
    fname_data = arguments['-i']
    path_label = arguments['-f']
    method = arguments['-method']
    fname_output = arguments['-o']
    if '-l' in arguments:
        labels_user = arguments['-l']
    else:
        labels_user = ''
    if '-param' in arguments:
        adv_param_user = arguments['-param']
    else:
        adv_param_user = ''
    if '-z' in arguments:
        slices_of_interest = arguments['-z']
    else:
        slices_of_interest = ''
    if '-vert' in arguments:
        vertebral_levels = arguments['-vert']
    else:
        vertebral_levels = ''
    if '-vertfile' in arguments:
        fname_vertebral_labeling = arguments['-vertfile']
    else:
        fname_vertebral_labeling = ""
    if '-perslice' in arguments:
        perslice = arguments['-perslice']
    else:
        perslice = 0
    if '-perlevel' in arguments:
        perlevel = arguments['-perlevel']
    else:
        perlevel = 0
    if '-overwrite' in arguments:
        overwrite = arguments['-overwrite']
    fname_normalizing_label = ''
    if '-norm-file' in arguments:
        fname_normalizing_label = arguments['-norm-file']
    normalization_method = ''
    if '-norm-method' in arguments:
        normalization_method = arguments['-norm-method']
    if '-fix-label' in arguments:
        label_to_fix = arguments['-fix-label']
    else:
        label_to_fix = ''
    if '-output-map' in arguments:
        fname_output_metric_map = arguments['-output-map']
    else:
        fname_output_metric_map = ''
    if '-mask-weighted' in arguments:
        fname_mask_weight = arguments['-mask-weighted']
    else:
        fname_mask_weight = ''
    # if 'discard_negative_values' in arguments:
    discard_negative_values = int(arguments['-discard-neg-val'])

    # call main function
    main(fname_data, path_label, method, slices_of_interest, vertebral_levels, fname_output, labels_user, overwrite,
         fname_normalizing_label, normalization_method, label_to_fix, adv_param_user, fname_output_metric_map,
         fname_mask_weight, fname_vertebral_labeling=fname_vertebral_labeling, perslice=perslice, perlevel=perlevel,
         discard_negative_values=discard_negative_values)
