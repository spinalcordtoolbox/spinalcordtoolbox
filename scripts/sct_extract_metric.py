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

import sys
import os
import argparse

import numpy as np

from spinalcordtoolbox.metadata import read_label_file
from spinalcordtoolbox.utils import parse_num_list
from spinalcordtoolbox.aggregate_slicewise import check_labels, extract_metric, save_as_csv, Metric, LabelStruc
import sct_utils as sct
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils import Metavar, SmartFormatter

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

    # Read .txt files referencing the labels (for extended usage description)
    file_label = os.path.join(param_default.path_label,
                              param_default.file_info_label)
    sct.check_file_exist(file_label, 0)
    default_info_label = open(file_label, 'r')
    label_references = default_info_label.read()
    default_info_label.close()

    description = (f"This program extracts metrics (e.g., DTI or MTR) within labels. Labels could be a single file or "
                   f"a folder generated with 'sct_warp_template' and containing multiple label files and a label "
                   f"description file (info_label.txt). The labels should be in the same space coordinates as the "
                   f"input image.\n"
                   f"\n"
                   f"To list white matter atlas labels: {os.path.basename(__file__)} -f "
                   f"{os.path.join(path_sct, 'data', 'atlas')}\n"
                   f"\n"
                   f"To compute FA within labels 0, 2 and 3 within vertebral levels C2 to C7 using binary method: "
                   f"{os.path.basename(__file__)} -i dti_FA.nii.gz -f label/atlas -l 0,2,3 -v 2:7 -m bin\n")

    if label_references != '':
        description += (f"\nTo compute average MTR in a region defined by a single label file (could be binary or 0-1 "
                        f"weighted mask) between slices 1 and 4: {os.path.basename(__file__)} -i mtr.nii.gz -f "
                        f"my_mask.nii.gz -z 1:4 -m wa\n"
                        f"List of labels in {file_label}:\n"
                        f"--------------------------------------------------------------------------------------\n"
                        f"{label_references}\n"
                        f"--------------------------------------------------------------------------------------\n")

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=SmartFormatter,
        add_help=None,
        prog=os.path.basename(__file__).strip(".py")
    )
    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        '-i',
        metavar=Metavar.file,
        required=True,
        help="Image file to extract metrics from. Example: FA.nii.gz"
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit."
    )
    optional.add_argument(
        '-f',
        metavar=Metavar.folder,
        default=os.path.join("label", "atlas"),
        help=(f"Single label file, or folder that contains WM tract labels."
              f"Example: {os.path.join(path_sct, 'data', 'atlas')}")
    )
    optional.add_argument(
        '-l',
        metavar=Metavar.str,
        default='',
        help=("Label IDs to extract the metric from. Default = all labels. Separate labels with ','. To select a group " 
              "of consecutive labels use ':'. Example: 1:3 is equivalent to 1,2,3. Maximum Likelihood (or MAP) is "
              "computed using all tracts, but only values of the selected tracts are reported.")
    )
    optional.add_argument(
        '-method',
        metavar=Metavar.str,
        choices=['ml', 'map', 'wa', 'bin', 'max'],
        default=param_default.method,
        help=("R|Method to extract metrics.\n"
              "    - ml: maximum likelihood (only use with well-defined regions and low noise)\n"
              "          N.B. ONLY USE THIS METHOD WITH THE WHITE MATTER ATLAS! The sum of all tracts should be 1 in " 
              "          all voxels (the algorithm doesn't normalize the atlas).\n"
              "    - map: maximum a posteriori. Mean priors are estimated by maximum likelihood within three clusters "
              "           (white matter, gray matter and CSF). Tract and  noise variance are set with flag -p.\n"
              "           N.B. ONLY USE THIS METHOD WITH THE WHITE MATTER ATLAS! The sum of all tracts should be 1 in "
              "           all voxels (the algorithm doesn't normalize the atlas).\n"
              "    - wa: weighted average\n"
              "    - bin: binarize mask (threshold=0.5)\n"
              "    - max: for each z-slice of the input data, extract the max value for each slice of the input data.")
    )
    # TODO: Change binary int options (-append, -combine -perslice, -perlevel) to flags using action=store_true
    optional.add_argument(
        '-append',
        type=int,
        metavar=Metavar.int,
        choices=(0, 1),
        default=0,
        help="Whether to append results as a new line in the output csv file instead of overwriting it. 0 = no, 1 = yes"
    )
    optional.add_argument(
        '-combine',
        type=int,
        metavar=Metavar.int,
        choices=(0, 1),
        default=0,
        help="Whether to combine multiple labels into a single estimation. 0 = no, 1 = yes"
    )
    optional.add_argument(
        '-o',
        metavar=Metavar.file,
        default=param_default.fname_output,
        help=("File name (including the file extension) of the output result file collecting the metric estimation "
              "results.\n"
              "Three file types are available: a CSV text file (extension .txt), a MS Excel file (extension .xls) and "
              "a pickle file (extension .pickle).")
    )
    optional.add_argument(
        '-output_map',
        metavar=Metavar.file,
        default='',
        help=("File name for an image consisting of the atlas labels multiplied by the estimated metric values "
              "yielding the metric value map, useful to assess the metric estimation and especially partial volume "
              "effects.")
    )
    optional.add_argument(
        '-z',
        metavar=Metavar.str,
        default=param_default.slices_of_interest,
        help=("Slice range to estimate the metric from. First slice is 0. Example: 5:23\n"
              "You can also select specific slices using commas. Example: 0,2,3,5,12'")
    )
    optional.add_argument(
        '-perslice',
        type=int,
        metavar=Metavar.int,
        choices=(0, 1),
        default=param_default.perslice,
        help=("Whether to output one metric per slice instead of a single output metric. 0 = no, 1 = yes.\n"
              "Please note that when methods ml or map are used, outputting a single metric per slice and then "
              "averaging them all is not the same as outputting a single metric at once across all slices.")
    )
    optional.add_argument(
        '-vert',
        metavar=Metavar.str,
        default=param_default.vertebral_levels,
        help="Vertebral levels to estimate the metric across. Example: 2:9 (for C2 to T2)"
    )
    optional.add_argument(
        '-vertfile',
        metavar=Metavar.file,
        default="./label/template/PAM50_levels.nii.gz",
        help="Vertebral labeling file. Only use with flag -vert. Example: PAM50_levels.nii.gz"
    )
    optional.add_argument(
        '-perlevel',
        type=int,
        metavar=Metavar.int,
        default=0,
        help=("Whether to output one metric per vertebral level instead of a single output metric. 0 = no, 1 = yes.\n"
              "Please note that this flag needs to be used with the -vert option.")
    )
    optional.add_argument(
        '-v',
        metavar=Metavar.str,
        choices=("0", "1"),
        default="1",
        help="Verbose. 0 = nothing, 1 = expanded"
    )

    # TODO: Ask about how parameter deprecations are to be handled (e.g. -param/-p, -normfile/-n)
    advanced = parser.add_argument_group("\nFOR ADVANCED USERS")
    advanced.add_argument(
        '-param',
        metavar=Metavar.str,
        default='',
        help=("Advanced parameters for the 'map' method. Separate with comma. All items must be listed (separated "
              "with comma).\n"
              "    -#1: standard deviation of metrics across labels\n"
              "    -#2: standard deviation of the noise (assumed Gaussian)")
    )
    advanced.add_argument(
        '-fix_label',
        metavar=Metavar.str,
        default='',
        help=("When using ML or MAP estimations, if you do not want to estimate the metric in one label and fix its "
              "value to avoid effects on other labels, specify <label_ID>,<metric_value. Example: -fix-label 36,0 "
              "(Fix the CSF value)")
    )
    advanced.add_argument(
        '-norm_file',
        metavar=Metavar.file,
        default='',
        help='Filename of the label by which the user wants to normalize.'
    )
    advanced.add_argument(
        '-norm_method',
        metavar=Metavar.str,
        choices=['sbs', 'whole'],
        default='',
        help=("Method to use for normalization:\n"
              "    -sbs: normalization slice-by-slice\n"
              "    -whole: normalization by the metric value in the whole label for all slices.")
    )
    advanced.add_argument(
        '-mask_weighted',
        metavar=Metavar.file,
        default='',
        help="Nifti mask to weight each voxel during ML or MAP estimation. Example: PAM50_wm.nii.gz"
    )
    advanced.add_argument(
        '-discard_neg_val',
        metavar=Metavar.str,
        choices=('0', '1'),
        default='0',
        help='Whether to discard voxels with negative value when computing metrics statistics. 0 = no, 1 = yes'
    )

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
    arguments = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    overwrite = 0
    fname_data = sct.get_absolute_path(arguments.i)
    path_label = arguments.f
    method = arguments.method
    fname_output = arguments.o
    append_csv = arguments.append
    combine_labels = arguments.combine
    labels_user = arguments.l
    adv_param_user = arguments.param
    slices_of_interest = arguments.z
    vertebral_levels = arguments.vert
    fname_vertebral_labeling = arguments.vertfile
    perslice = arguments.perslice
    perlevel = arguments.perlevel
    fname_normalizing_label = arguments.norm_file
    normalization_method = arguments.norm_method
    label_to_fix = arguments.fix_label
    fname_output_metric_map = arguments.output_map
    fname_mask_weight = arguments.mask_weighted
    discard_negative_values = int(arguments.discard_neg_val)
    verbose = int(arguments.v)
    sct.init_sct(log_level=verbose, update=True)  # Update log level

    # call main function
    main(fname_data=fname_data, path_label=path_label, method=method, slices=parse_num_list(slices_of_interest),
         levels=parse_num_list(vertebral_levels), fname_output=fname_output, labels_user=labels_user,
         append_csv=append_csv, fname_vertebral_labeling=fname_vertebral_labeling, perslice=perslice,
         perlevel=perlevel, verbose=verbose, combine_labels=combine_labels)
