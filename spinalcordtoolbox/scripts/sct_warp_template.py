#!/usr/bin/env python
#########################################################################################
#
# Warp template and atlas to a given volume (DTI, MT, etc.).
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad
# Modified: 2014-07-20
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sys
import os

import spinalcordtoolbox.metadata
from spinalcordtoolbox.reports.qc import generate_qc
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, ActionCreateFolder, display_viewer_syntax
from spinalcordtoolbox.utils.sys import init_sct, printv, __data_dir__, set_loglevel
from spinalcordtoolbox.utils.fs import copy
from spinalcordtoolbox.scripts import sct_apply_transfo


# DEFAULT PARAMETERS
class Param:
    # The constructor
    def __init__(self):
        self.debug = 0
        self.folder_out = 'label'  # name of output folder
        self.path_template = os.path.join(__data_dir__, "PAM50")
        self.folder_template = 'template'
        self.folder_atlas = 'atlas'
        self.folder_spinal_levels = 'spinal_levels'
        self.file_info_label = 'info_label.txt'
        # self.warp_template = 1
        self.warp_atlas = 1
        self.warp_spinal_levels = 0
        self.list_labels_nn = ['_level.nii.gz', '_levels.nii.gz', '_csf.nii.gz', '_CSF.nii.gz', '_cord.nii.gz']  # list of files for which nn interpolation should be used. Default = linear.
        self.verbose = 1  # verbose
        self.path_qc = None
        self.warp_histo = 0
        self.folder_histo = 'histology'
        self.list_labels_nn_histo = ['_200um_Naxons.nii.gz', '_200um_MVF.nii.gz', '_200um_AVF.nii.gz',
                                     '_200um_EquivDiameter.nii.gz', '_200um_EquivDiameter14.nii.gz',
                                     '_200um_EquivDiameter48.nii.gz', '_200um_Eccentricity.nii.gz']


class WarpTemplate:
    def __init__(self, fname_src, fname_transfo, warp_atlas, warp_spinal_levels, folder_out, path_template,
                 folder_template, folder_atlas, folder_spinal_levels, file_info_label, list_labels_nn,
                 verbose, warp_histo, folder_histo):

        # Initialization
        self.fname_src = fname_src
        self.fname_transfo = fname_transfo
        self.warp_atlas = warp_atlas
        self.warp_spinal_levels = warp_spinal_levels
        self.folder_out = folder_out
        self.path_template = path_template
        self.folder_template = folder_template
        self.folder_atlas = folder_atlas
        self.folder_spinal_levels = folder_spinal_levels
        self.file_info_label = file_info_label
        self.list_labels_nn = list_labels_nn
        self.verbose = verbose
        self.warp_histo = warp_histo
        self.folder_histo = folder_histo

        # printv(arguments)
        # printv(arguments)
        printv('\nCheck parameters:')
        printv('  Working directory ........ ' + os.getcwd())
        printv('  Destination image ........ ' + self.fname_src)
        printv('  Warping field ............ ' + self.fname_transfo)
        printv('  Path template ............ ' + self.path_template)
        printv('  Output folder ............ ' + self.folder_out + "\n")

        # create output folder
        if not os.path.exists(self.folder_out):
            os.makedirs(self.folder_out)

        # Warp template objects
        printv('\nWARP TEMPLATE:', self.verbose)
        warp_label(self.path_template, self.folder_template, self.file_info_label, self.fname_src,
                   self.fname_transfo, self.folder_out, self.list_labels_nn, self.verbose)

        # Warp atlas
        if self.warp_atlas == 1:
            printv('\nWARP ATLAS OF WHITE MATTER TRACTS:', self.verbose)
            warp_label(self.path_template, self.folder_atlas, self.file_info_label, self.fname_src,
                       self.fname_transfo, self.folder_out, self.list_labels_nn, self.verbose)

        # Warp spinal levels
        if self.warp_spinal_levels == 1:
            printv('\nWARP SPINAL LEVELS:', self.verbose)
            warp_label(self.path_template, self.folder_spinal_levels, self.file_info_label, self.fname_src,
                       self.fname_transfo, self.folder_out, self.list_labels_nn, self.verbose)

        # Warp histology atlas
        if self.warp_histo == 1:
            printv('\nWARP HISTOLOGY ATLAS:', self.verbose)
            warp_label(self.path_template, self.folder_histo, self.file_info_label, self.fname_src,
                       self.fname_transfo, self.folder_out, self.list_labels_nn, self.verbose)


def warp_label(path_label, folder_label, file_label, fname_src, fname_transfo, path_out, list_labels_nn, verbose):
    """
    Warp label files according to info_label.txt file
    :param path_label:
    :param folder_label:
    :param file_label:
    :param fname_src:
    :param fname_transfo:
    :param path_out:
    :param list_labels_nn:
    :param verbose:
    :return:
    """
    try:
        # Read label file
        template_label_ids, template_label_names, template_label_file, combined_labels_ids, combined_labels_names, \
            combined_labels_id_groups, clusters_apriori = \
            spinalcordtoolbox.metadata.read_label_file(os.path.join(path_label, folder_label), file_label)
    except Exception as error:
        printv('\nWARNING: Cannot warp label ' + folder_label + ': ' + str(error), 1, 'warning')
        raise
    else:
        # create output folder
        if not os.path.exists(os.path.join(path_out, folder_label)):
            os.makedirs(os.path.join(path_out, folder_label))
        # Warp label
        for i in range(0, len(template_label_file)):
            fname_label = os.path.join(path_label, folder_label, template_label_file[i])
            # apply transfo
            sct_apply_transfo.main(['-i', fname_label,
                                    '-d', fname_src,
                                    '-w', fname_transfo,
                                    '-o', os.path.join(path_out, folder_label, template_label_file[i]),
                                    '-x', get_interp(template_label_file[i], list_labels_nn),
                                    '-v', '0'])
        # Copy list.txt
        copy(os.path.join(path_label, folder_label, file_label), os.path.join(path_out, folder_label))


# Get interpolation method
# ==========================================================================================
def get_interp(file_label, list_labels_nn):
    # default interp
    interp = 'linear'
    # Nearest Neighbours interp
    if any(substring in file_label for substring in list_labels_nn):
        interp = 'nn'
    # output
    return interp


# PARSER
# ==========================================================================================
def get_parser():
    # Initialize default parameters
    param_default = Param()
    parser = SCTArgumentParser(
        description="This function warps the template and all atlases to a destination image."
    )

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        '-d',
        metavar=Metavar.file,
        required=True,
        help="Destination image the template will be warped to. Example: dwi_mean.nii.gz"
    )
    mandatory.add_argument(
        '-w',
        metavar=Metavar.file,
        required=True,
        help="Warping field. Example: warp_template2dmri.nii.gz"
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit."
    )
    optional.add_argument(
        '-a',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=param_default.warp_atlas,
        help="Warp atlas of white matter."
    )
    optional.add_argument(
        '-s',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=param_default.warp_spinal_levels,
        help="Warp spinal levels."
    )
    optional.add_argument(
        '-ofolder',
        metavar=Metavar.folder,
        action=ActionCreateFolder,
        default=param_default.folder_out,
        help="Name of output folder."
    )
    optional.add_argument(
        '-t',
        metavar=Metavar.folder,
        default=str(param_default.path_template),
        help="Path to template."
    )
    optional.add_argument(
        '-qc',
        metavar=Metavar.folder,
        action=ActionCreateFolder,
        default=param_default.path_qc,
        help="The path where the quality control generated content will be saved."
    )
    optional.add_argument(
        '-qc-dataset',
        metavar=Metavar.str,
        help="If provided, this string will be mentioned in the QC report as the dataset the process was run on."
    )
    optional.add_argument(
        '-qc-subject',
        metavar=Metavar.str,
        help="If provided, this string will be mentioned in the QC report as the subject the process was run on."
    )
    optional.add_argument(
        '-v',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode"
    )
    optional.add_argument(
        '-histo',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=param_default.warp_histo,
        help="Warp histology atlas from Duval et al. Neuroimage 2019 (https://pubmed.ncbi.nlm.nih.gov/30326296/)."
    )

    return parser


def main(argv=None):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)

    param = Param()

    fname_src = arguments.d
    fname_transfo = arguments.w
    warp_atlas = arguments.a
    warp_spinal_levels = arguments.s
    folder_out = arguments.ofolder
    path_template = arguments.t
    path_qc = arguments.qc
    qc_dataset = arguments.qc_dataset
    qc_subject = arguments.qc_subject
    folder_template = param.folder_template
    folder_atlas = param.folder_atlas
    folder_spinal_levels = param.folder_spinal_levels
    file_info_label = param.file_info_label
    list_labels_nn = param.list_labels_nn
    warp_histo = arguments.histo
    folder_histo = param.folder_histo

    # call main function
    w = WarpTemplate(fname_src, fname_transfo, warp_atlas, warp_spinal_levels, folder_out, path_template,
                     folder_template, folder_atlas, folder_spinal_levels, file_info_label, list_labels_nn,
                     verbose, warp_histo, folder_histo)

    path_template = os.path.join(w.folder_out, w.folder_template)

    # Deal with QC report
    if path_qc is not None:
        try:
            fname_wm = os.path.join(
                w.folder_out, w.folder_template, spinalcordtoolbox.metadata.get_file_label(path_template, id_label=4))  # label = 'white matter mask (probabilistic)'
            generate_qc(
                fname_src, fname_seg=fname_wm, args=sys.argv[1:], path_qc=os.path.abspath(path_qc), dataset=qc_dataset,
                subject=qc_subject, process='sct_warp_template')
        # If label is missing, get_file_label() throws a RuntimeError
        except RuntimeError:
            printv("QC not generated since expected labels are missing from template", type="warning")

    # Deal with verbose
    try:
        display_viewer_syntax(
            [fname_src,
             spinalcordtoolbox.metadata.get_file_label(path_template, id_label=1, output="filewithpath"),  # label = 'T2-weighted template'
             spinalcordtoolbox.metadata.get_file_label(path_template, id_label=5, output="filewithpath"),  # label = 'gray matter mask (probabilistic)'
             spinalcordtoolbox.metadata.get_file_label(path_template, id_label=4, output="filewithpath")],  # label = 'white matter mask (probabilistic)'
            colormaps=['gray', 'gray', 'red-yellow', 'blue-lightblue'],
            opacities=['1', '1', '0.5', '0.5'],
            minmax=['', '0,4000', '0.4,1', '0.4,1'],
            verbose=verbose)
    # If label is missing, continue silently
    except RuntimeError:
        pass


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])

