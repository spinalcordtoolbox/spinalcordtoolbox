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

from __future__ import absolute_import

import sys, os

import spinalcordtoolbox.metadata
from spinalcordtoolbox.reports.qc import generate_qc
from msct_parser import Parser
import sct_utils as sct


# DEFAULT PARAMETERS
class Param:
    # The constructor
    def __init__(self):
        self.debug = 0
        self.folder_out = 'label'  # name of output folder
        self.path_template = os.path.join(sct.__data_dir__, "PAM50")
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


class WarpTemplate:
    def __init__(self, fname_src, fname_transfo, warp_atlas, warp_spinal_levels, folder_out, path_template, verbose):

        # Initialization
        self.fname_src = fname_src
        self.fname_transfo = fname_transfo
        self.warp_atlas = warp_atlas
        self.warp_spinal_levels = warp_spinal_levels
        self.folder_out = folder_out
        self.path_template = path_template
        self.folder_template = param.folder_template
        self.folder_atlas = param.folder_atlas
        self.folder_spinal_levels = param.folder_spinal_levels
        self.verbose = verbose

        # sct.printv(arguments)
        sct.printv('\nCheck parameters:')
        sct.printv('  Working directory ........ ' + os.getcwd())
        sct.printv('  Destination image ........ ' + self.fname_src)
        sct.printv('  Warping field ............ ' + self.fname_transfo)
        sct.printv('  Path template ............ ' + self.path_template)
        sct.printv('  Output folder ............ ' + self.folder_out + "\n")

        # create output folder
        if not os.path.exists(self.folder_out):
            os.makedirs(self.folder_out)

        # Warp template objects
        sct.printv('\nWARP TEMPLATE:', self.verbose)
        warp_label(self.path_template, self.folder_template, param.file_info_label, self.fname_src, self.fname_transfo, self.folder_out)

        # Warp atlas
        if self.warp_atlas == 1:
            sct.printv('\nWARP ATLAS OF WHITE MATTER TRACTS:', self.verbose)
            warp_label(self.path_template, self.folder_atlas, param.file_info_label, self.fname_src, self.fname_transfo, self.folder_out)

        # Warp spinal levels
        if self.warp_spinal_levels == 1:
            sct.printv('\nWARP SPINAL LEVELS:', self.verbose)
            warp_label(self.path_template, self.folder_spinal_levels, param.file_info_label, self.fname_src, self.fname_transfo, self.folder_out)


def warp_label(path_label, folder_label, file_label, fname_src, fname_transfo, path_out):
    """
    Warp label files according to info_label.txt file
    :param path_label:
    :param folder_label:
    :param file_label:
    :param fname_src:
    :param fname_transfo:
    :param path_out:
    :return:
    """
    try:
        # Read label file
        template_label_ids, template_label_names, template_label_file, combined_labels_ids, combined_labels_names, \
        combined_labels_id_groups, clusters_apriori = \
            spinalcordtoolbox.metadata.read_label_file(os.path.join(path_label, folder_label), file_label)
    except Exception as error:
        sct.printv('\nWARNING: Cannot warp label ' + folder_label + ': ' + str(error), 1, 'warning')
        raise
    else:
        # create output folder
        if not os.path.exists(os.path.join(path_out, folder_label)):
            os.makedirs(os.path.join(path_out, folder_label))
        # Warp label
        for i in range(0, len(template_label_file)):
            fname_label = os.path.join(path_label, folder_label, template_label_file[i])
            # apply transfo
            sct.run('isct_antsApplyTransforms -d 3 -i %s -r %s -t %s -o %s -n %s' %
                    (fname_label,
                     fname_src,
                     fname_transfo,
                     os.path.join(path_out, folder_label, template_label_file[i]),
                     get_interp(template_label_file[i])),
                    is_sct_binary=True,
                    verbose=param.verbose)
        # Copy list.txt
        sct.copy(os.path.join(path_label, folder_label, param.file_info_label), os.path.join(path_out, folder_label))


# Get interpolation method
# ==========================================================================================
def get_interp(file_label):
    # default interp
    interp = 'Linear'
    # NN interp
    if any(substring in file_label for substring in param.list_labels_nn):
        interp = 'NearestNeighbor'
    # output
    return interp


# PARSER
# ==========================================================================================
def get_parser():
    # Initialize default parameters
    param_default = Param()
    # Initialize parser
    parser = Parser(__file__)
    parser.usage.set_description('This function warps the template and all atlases to a destination image.')
    parser.add_option(name="-d",
                      type_value="file",
                      description="destination image the template will be warped to",
                      mandatory=True,
                      example="dwi_mean.nii.gz")
    parser.add_option(name="-w",
                      type_value="file",
                      description="warping field",
                      mandatory=True,
                      example="warp_template2dmri.nii.gz")
    parser.add_option(name="-a",
                      type_value="multiple_choice",
                      description="warp atlas of white matter.",
                      mandatory=False,
                      default_value=str(param_default.warp_atlas),
                      example=['0', '1'])
    parser.add_option(name="-s",
                      type_value="multiple_choice",
                      description="warp spinal levels.",
                      mandatory=False,
                      default_value=str(param_default.warp_spinal_levels),
                      example=['0', '1'])
    parser.add_option(name="-ofolder",
                      type_value="folder_creation",
                      description="name of output folder.",
                      mandatory=False,
                      default_value=param_default.folder_out,
                      example="label")
    parser.add_option(name="-t",
                      type_value="folder",
                      description="Path to template.",
                      mandatory=False,
                      default_value=str(param_default.path_template))
    parser.add_option(name='-qc',
                      type_value='folder_creation',
                      description='The path where the quality control generated content will be saved',
                      default_value=param_default.path_qc)
    parser.add_option(name='-qc-dataset',
                      type_value='str',
                      description='If provided, this string will be mentioned in the QC report as the dataset the process was run on',
                      )
    parser.add_option(name='-qc-subject',
                      type_value='str',
                      description='If provided, this string will be mentioned in the QC report as the subject the process was run on',
                      )
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="""Verbose.""",
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])
    return parser


def main(args=None):

    parser = get_parser()

    arguments = parser.parse(sys.argv[1:])

    fname_src = arguments["-d"]
    fname_transfo = arguments["-w"]
    warp_atlas = int(arguments["-a"])
    warp_spinal_levels = int(arguments["-s"])
    folder_out = arguments['-ofolder']
    path_template = arguments['-t']
    verbose = int(arguments.get('-v'))
    sct.init_sct(log_level=verbose, update=True)  # Update log level
    path_qc = arguments.get("-qc", None)
    qc_dataset = arguments.get("-qc-dataset", None)
    qc_subject = arguments.get("-qc-subject", None)

    # call main function
    w = WarpTemplate(fname_src, fname_transfo, warp_atlas, warp_spinal_levels, folder_out, path_template, verbose)

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
            sct.printv("QC not generated since expected labels are missing from template", type="warning")

    # Deal with verbose
    try:
        sct.display_viewer_syntax(
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
    sct.init_sct()
    param = Param()
    main()
