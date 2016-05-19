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


#import re
import sys
import commands
import getopt
import os
import time

from msct_parser import Parser
import sct_utils as sct
from sct_extract_metric import read_label_file


# get path of the script and the toolbox
path_script = os.path.dirname(__file__)
path_sct = os.path.dirname(path_script)


# DEFAULT PARAMETERS
class Param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.folder_out = 'label'  # name of output folder
        self.path_template = path_sct+'/data'
        self.folder_template = 'template'
        self.folder_atlas = 'atlas'
        self.folder_spinal_levels = 'spinal_levels'
        self.file_info_label = 'info_label.txt'
        # self.warp_template = 1
        self.warp_atlas = 1
        self.warp_spinal_levels = 0
        self.list_labels_nn = ['MNI-Poly-AMU_level.nii.gz', 'MNI-Poly-AMU_CSF.nii.gz', 'MNI-Poly-AMU_cord.nii.gz']  # list of files for which nn interpolation should be used. Default = linear.
        self.list_labels_spline = ['']  # list of files for which spline interpolation should be used. Default = linear.
        self.verbose = 1  # verbose
        self.qc = 1
param = Param()


# MAIN
# ==========================================================================================
class WarpTemplate:
    def __init__(self, fname_src, fname_transfo, warp_atlas, warp_spinal_levels, folder_out, path_template, verbose, qc):

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
        self.qc = qc
        start_time = time.time()

        # Check file existence
        sct.printv('\nCheck file existence...', self.verbose)
        sct.check_file_exist(self.fname_src)
        sct.check_file_exist(self.fname_transfo)

        # add slash at the end of folder name (in case there is no slash)
        self.path_template = sct.slash_at_the_end(self.path_template, 1)
        self.folder_out = sct.slash_at_the_end(self.folder_out, 1)
        self.folder_template = sct.slash_at_the_end(self.folder_template, 1)
        self.folder_atlas = sct.slash_at_the_end(self.folder_atlas, 1)
        self.folder_spinal_levels = sct.slash_at_the_end(self.folder_spinal_levels, 1)

        # print arguments
        print '\nCheck parameters:'
        print '  Destination image ........ '+self.fname_src
        print '  Warping field ............ '+self.fname_transfo
        print '  Path template ............ '+self.path_template
        print '  Output folder ............ '+self.folder_out+'\n'

        # Extract path, file and extension
        path_src, file_src, ext_src = sct.extract_fname(self.fname_src)

        # create output folder
        if os.path.exists(self.folder_out):
            sct.printv('WARNING: Output folder already exists. Deleting it...', self.verbose, 'warning')
            sct.run('rm -rf '+self.folder_out)
        sct.run('mkdir '+self.folder_out)

        # Warp template objects
        sct.printv('\nWarp template objects...', self.verbose)
        warp_label(self.path_template, self.folder_template, param.file_info_label, self.fname_src, self.fname_transfo, self.folder_out)

        # Warp atlas
        if self.warp_atlas == 1:
            sct.printv('\nWarp atlas of white matter tracts...', self.verbose)
            warp_label(self.path_template, self.folder_atlas, param.file_info_label, self.fname_src, self.fname_transfo, self.folder_out)

        # Warp spinal levels
        if self.warp_spinal_levels == 1:
            sct.printv('\nWarp spinal levels...', self.verbose)
            warp_label(self.path_template, self.folder_spinal_levels, param.file_info_label, self.fname_src, self.fname_transfo, self.folder_out)

        # to view results
        sct.printv('\nDone! To view results, type:', self.verbose)
        sct.printv('fslview '+self.fname_src+' '+self.folder_out+self.folder_template+'MNI-Poly-AMU_T2.nii.gz -b 0,4000 '+self.folder_out+self.folder_template+'MNI-Poly-AMU_level.nii.gz -l MGH-Cortical -t 0.5 '+self.folder_out+self.folder_template+'MNI-Poly-AMU_GM.nii.gz -l Red-Yellow -b 0.5,1 '+self.folder_out+self.folder_template+'MNI-Poly-AMU_WM.nii.gz -l Blue-Lightblue -b 0.5,1 &\n', self.verbose, 'info')

        if self.qc:
            from msct_image import Image
            # output QC image
            im = Image(self.fname_src)
            im_wm = Image(self.folder_out+self.folder_template+'MNI-Poly-AMU_WM.nii.gz')
            im.save_quality_control(plane='axial', n_slices=4, seg=im_wm, thr=0.5, cmap_col='blue-cyan', path_output=self.folder_out)


# Warp labels
# ==========================================================================================
def warp_label(path_label, folder_label, file_label, fname_src, fname_transfo, path_out):
    # read label file and check if file exists
    sct.printv('\nRead label file...', param.verbose)
    template_label_ids, template_label_names, template_label_file, combined_labels_ids, combined_labels_names, combined_labels_id_groups = read_label_file(path_label+folder_label, file_label)
    # create output folder
    sct.run('mkdir '+path_out+folder_label, param.verbose)
    # Warp label
    for i in xrange(0, len(template_label_file)):
        fname_label = path_label+folder_label+template_label_file[i]
        # check if file exists
        # sct.check_file_exist(fname_label)
        # apply transfo
        sct.run('sct_apply_transfo -i '+fname_label+' -o '+path_out+folder_label+template_label_file[i] +' -d '+fname_src+' -w '+fname_transfo+' -x '+get_interp(template_label_file[i]), param.verbose)
    # Copy list.txt
    sct.run('cp '+path_label+folder_label+param.file_info_label+' '+path_out+folder_label, 0)



# Get interpolation method
# ==========================================================================================
def get_interp(file_label):
    # default interp
    interp = 'linear'
    # NN interp
    if file_label in param.list_labels_nn:
        interp = 'nn'
    # spline interp
    if file_label in param.list_labels_spline:
        interp = 'spline'
    # output
    return interp


# START PROGRAM
# ==========================================================================================
def main(args=None):

    if args is None:
        args = sys.argv[1:]
    # initialize parameters
    param_default = Param()

    # Initialize parser
    parser = Parser(__file__)

    # Mandatory arguments
    parser.usage.set_description('This function warps the template and all atlases to a given image (e.g. fMRI, DTI, MTR, etc.).')
    parser.add_option(name="-d",
                      type_value="file",
                      description="destination image the template will be warped into",
                      mandatory=True,
                      example="dwi_mean.nii.gz")
    parser.add_option(name="-w",
                      type_value="file",
                      description="warping field",
                      mandatory=True,
                      example="warp_template2dmri.nii.gz")
    parser.add_option(name="-a",
                      type_value="multiple_choice",
                      description="warp atlas of white matter",
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
    parser.add_option(name="-o",
                      type_value=None,
                      description="name of output folder.",
                      mandatory=False,
                      deprecated_by='-ofolder')
    parser.add_option(name="-t",
                      type_value="folder",
                      description="Specify path to template data.",
                      mandatory=False,
                      default_value=str(param_default.path_template))
    parser.add_option(name='-qc',
                      type_value='multiple_choice',
                      description='Output images for quality control.',
                      mandatory=False,
                      example=['0', '1'],
                      default_value='1')
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="""Verbose.""",
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])

    arguments = parser.parse(args)

    fname_src = arguments["-d"]
    fname_transfo = arguments["-w"]
    warp_atlas = int(arguments["-a"])
    warp_spinal_levels = int(arguments["-s"])
    folder_out = arguments["-ofolder"]
    path_template = arguments["-t"]
    verbose = int(arguments["-v"])
    qc = int(arguments["-qc"])

    # call main function
    WarpTemplate(fname_src, fname_transfo, warp_atlas, warp_spinal_levels, folder_out, path_template, verbose, qc)


if __name__ == "__main__":
    main()
