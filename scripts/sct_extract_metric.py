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

# TODO: fetch vert level in atlas by default-- would be nice to output in csv
# TODO: use argparse
# TODO (not urgent): vertebral levels selection should only consider voxels of the selected levels in slices where
#  two different vertebral levels coexist (and not the whole slice)

from __future__ import division, absolute_import

import sys, os

import numpy as np

from spinalcordtoolbox.metadata import read_label_file
from spinalcordtoolbox.utils import parse_num_list
from spinalcordtoolbox.aggregate_slicewise import check_labels, extract_metric, save_as_csv, Metric, LabelStruc
import sct_utils as sct
from spinalcordtoolbox.image import Image
from msct_parser import Parser

# get path of the script and the toolbox
# TODO: is that useful??
path_script = os.path.dirname(__file__)
path_sct = os.path.dirname(path_script)


class Param:
    def __init__(self):
        self.method = 'wa'
        self.path_label = os.path.join(path_sct, "data", "PAM50", "atlas")
        self.verbose = 1
        self.vertebral_levels = ''
        self.slices_of_interest = ''  # 2-element list corresponding to zmin:zmax. example: '5:8'. For all slices, leave empty.
        self.average_all_labels = 0  # average all labels together after concatenation
        self.fname_output = 'extract_metric.csv'
        self.file_info_label = 'info_label.txt'
        self.perslice = None
        self.perlevel = None


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
bin: binarize mask (threshold=0.5)
max: for each z-slice of the input data, extract the max value for each slice of the input data.""",
                      example=['ml', 'map', 'wa', 'bin', 'max'],
                      mandatory=False,
                      default_value=param_default.method)
    parser.add_option(name='-append',
                      type_value='int',
                      description='Append results as a new line in the output csv file instead of overwriting it.',
                      mandatory=False,
                      default_value=0)
    parser.add_option(name='-combine',
                      type_value='int',
                      description='Combine multiple labels into a single estimation.',
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
                      default_value=param_default.perslice)
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
                                  'output metric. This flag needs to be used with flag -vert.',
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


def main(fname_data, path_label, method, slices, levels, fname_output, labels_user, append_csv,
         fname_vertebral_labeling="", perslice=1, perlevel=1, verbose=1, combine_labels=True):
    """
    Extract metrics from MRI data based on mask (could be single file of folder to atlas)
    :param fname_data: data to extract metric from
    :param path_label: mask: could be single file or folder to atlas (which contains info_label.txt)
    :param method {'wa', 'bin', 'ml', 'map'}
    :param slices. Slices of interest. Accepted format:
           "0,1,2,3": slices 0,1,2,3
           "0:3": slices 0,1,2,3
    :param levels: Vertebral levels to extract metrics from. Should be associated with a template
           (e.g. PAM50/template/) or a specified file: fname_vertebral_labeling. Same format as slices_of_interest.
    :param fname_output:
    :param labels_user:
    :param append_csv: Append to csv file
    :param fname_normalizing_label:
    :param fname_vertebral_labeling: vertebral labeling to be used with vertebral_levels
    :param perslice: if user selected several slices, then the function outputs a metric within each slice
           instead of a single average output.
    :param perlevel: if user selected several levels, then the function outputs a metric within each vertebral level
           instead of a single average output.
    :param verbose
    :param combine_labels: bool: Combine labels into a single value
    :return:
    """

    # check if path_label is a file (e.g., single binary mask) instead of a folder (e.g., SCT atlas structure which
    # contains info_label.txt file)
    if os.path.isfile(path_label):
        # Label is a single file
        indiv_labels_ids = [0]
        indiv_labels_files = [path_label]
        combined_labels_ids = []
        label_struc = {0: LabelStruc(id=0,
                                     name=sct.extract_fname(path_label)[1],
                                     filename=path_label)}
        # set path_label to empty string, because indiv_labels_files will replace it from now on
        path_label = ''
    elif os.path.isdir(path_label):
        # Labels is an SCT atlas folder structure
        # Parse labels according to the file info_label.txt
        # Note: the "combined_labels_*" is a list of single labels that are defined in the section defined by the keyword
        # "# Keyword=CombinedLabels" in info_label.txt.
        # TODO: redirect to appropriate Sphinx documentation
        # TODO: output Class instead of multiple variables.
        #   Example 1:
        #     label_struc[2].id = (2)
        #     label_struc[2].name = "left fasciculus cuneatus"
        #     label_struc[2].filename = "PAM50_atlas_02.nii.gz"
        #   Example 2:
        #     label_struc[51].id = (1, 2, 3, ..., 29)
        #     label_struc[51].name = "White Matter"
        #     label_struc[51].filename = ""  # no name because it is combined
        indiv_labels_ids, indiv_labels_names, indiv_labels_files, \
        combined_labels_ids, combined_labels_names, combined_labels_id_groups, map_clusters \
            = read_label_file(path_label, param_default.file_info_label)

        label_struc = {}
        # fill IDs for indiv labels
        for i_label in range(len(indiv_labels_ids)):
            label_struc[indiv_labels_ids[i_label]] = LabelStruc(id=indiv_labels_ids[i_label],
                                                                name=indiv_labels_names[i_label],
                                                                filename=indiv_labels_files[i_label],
                                                                map_cluster=[indiv_labels_ids[i_label] in map_cluster for
                                                                             map_cluster in map_clusters].index(True))
        # fill IDs for combined labels
        # TODO: problem for defining map_cluster: if labels overlap two regions, e.g. WM and GM (e.g. id=50),
        #  map_cluster will take value 0, which is wrong.
        for i_label in range(len(combined_labels_ids)):
            label_struc[combined_labels_ids[i_label]] = LabelStruc(id=combined_labels_id_groups[i_label],
                                                                   name=combined_labels_names[i_label],
                                                                   map_cluster=[indiv_labels_ids[i_label] in map_cluster for
                                                                                map_cluster in map_clusters].index(True))
    else:
        raise RuntimeError(path_label + ' does not exist')

    # check syntax of labels asked by user
    labels_id_user = check_labels(indiv_labels_ids + combined_labels_ids, parse_num_list(labels_user))
    nb_labels = len(indiv_labels_files)

    # Load data and systematically reorient to RPI because we need the 3rd dimension to be z
    sct.printv('\nLoad metric image...', verbose)
    input_im = Image(fname_data).change_orientation("RPI")

    data = Metric(data=input_im.data, label='')
    # Load labels
    labels_tmp = np.empty([nb_labels], dtype=object)
    for i_label in range(nb_labels):
        im_label = Image(os.path.join(path_label, indiv_labels_files[i_label])).change_orientation("RPI")
        labels_tmp[i_label] = np.expand_dims(im_label.data, 3)  # TODO: generalize to 2D input label
    labels = np.concatenate(labels_tmp[:], 3)  # labels: (x,y,z,label)
    # Load vertebral levels
    if vertebral_levels:
        im_vertebral_labeling = Image(fname_vertebral_labeling).change_orientation("RPI")
    else:
        im_vertebral_labeling = None

    # Get dimensions of data and labels
    nx, ny, nz = data.data.shape
    nx_atlas, ny_atlas, nz_atlas, nt_atlas = labels.shape

    # Check dimensions consistency between atlas and data
    if (nx, ny, nz) != (nx_atlas, ny_atlas, nz_atlas):
        sct.printv('\nERROR: Metric data and labels DO NOT HAVE SAME DIMENSIONS.', 1, type='error')

    # Combine individual labels for estimation
    if combine_labels:
        # Add entry with internal ID value (99) which corresponds to combined labels
        label_struc[99] = LabelStruc(id=labels_id_user, name=','.join([str(i) for i in labels_id_user]),
                                     map_cluster=None)
        labels_id_user = [99]

    for id_label in labels_id_user:
        sct.printv('Estimation for label: '+label_struc[id_label].name, verbose)
        agg_metric = extract_metric(data, labels=labels, slices=slices, levels=levels, perslice=perslice,
                                    perlevel=perlevel, vert_level=im_vertebral_labeling, method=method,
                                    label_struc=label_struc, id_label=id_label, indiv_labels_ids=indiv_labels_ids)

        save_as_csv(agg_metric, fname_output, fname_in=fname_data, append=append_csv)
        append_csv = True  # when looping across labels, need to append results in the same file
    sct.display_open(fname_output)


if __name__ == "__main__":

    sct.init_sct()

    param_default = Param()

    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])

    overwrite = 0
    fname_data = sct.get_absolute_path(arguments['-i'])
    path_label = arguments['-f']
    method = arguments['-method']
    fname_output = arguments['-o']
    if '-append' in arguments:
        append_csv = int(arguments['-append'])
    else:
        append_csv = 0
    if '-combine' in arguments:
        combine_labels = arguments['-combine']
    else:
        combine_labels = 0
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
        perslice = param_default.perslice
    if '-perlevel' in arguments:
        perlevel = arguments['-perlevel']
    else:
        perlevel = 0
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
    verbose = int(arguments.get('-v'))
    sct.init_sct(log_level=verbose, update=True)  # Update log level

    # call main function
    main(fname_data=fname_data, path_label=path_label, method=method, slices=parse_num_list(slices_of_interest),
         levels=parse_num_list(vertebral_levels), fname_output=fname_output, labels_user=labels_user,
         append_csv=append_csv, fname_vertebral_labeling=fname_vertebral_labeling, perslice=perslice,
         perlevel=perlevel, verbose=verbose, combine_labels=combine_labels)
