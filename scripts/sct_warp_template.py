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

import sct_utils as sct
from sct_extract_metric import read_label_file




# get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')


# DEFAULT PARAMETERS
class Param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.folder_out = 'label'  # name of output folder
        self.path_template = path_sct+'/data/'
        self.folder_template = 'template'
        self.folder_atlas = 'atlas'
        self.folder_spinal_levels = 'spinal_levels'
        self.file_info_label = 'info_label.txt'
        self.warp_template = 1
        self.warp_atlas = 1
        self.warp_spinal_levels = 0
        self.list_labels_nn = ['MNI-Poly-AMU_level.nii.gz', 'MNI-Poly-AMU_CSF.nii.gz', 'MNI-Poly-AMU_cord.nii.gz']  # list of files for which nn interpolation should be used. Default = linear.
        self.list_labels_spline = ['']  # list of files for which spline interpolation should be used. Default = linear.
        self.verbose = 1  # verbose


# MAIN
# ==========================================================================================
def main():

    # Initialization
    fname_src = ''
    fname_transfo = ''
    folder_out = param.folder_out
    path_template = param.path_template
    folder_template = param.folder_template
    folder_atlas = param.folder_atlas
    folder_spinal_levels = param.folder_spinal_levels
    file_info_label = param.file_info_label
    warp_template = param.warp_template
    warp_atlas = param.warp_atlas
    warp_spinal_levels = param.warp_spinal_levels
    verbose = param.verbose
    start_time = time.time()


    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        fname_src = path_sct+'/testing/sct_testing_data/data/mt/mtr.nii.gz'
        fname_transfo = path_sct+'/testing/sct_testing_data/data/mt/warp_template2mt.nii.gz'
        warp_atlas = 1
        warp_spinal_levels = 1
        verbose = 1
    else:
        # Check input parameters
        try:
            opts, args = getopt.getopt(sys.argv[1:], 'ha:d:w:o:s:t:v:')
        except getopt.GetoptError:
            usage()
        for opt, arg in opts:
            if opt == '-h':
                usage()
            elif opt in ("-a"):
                warp_atlas = int(arg)
            elif opt in ("-d"):
                fname_src = arg
            elif opt in ("-o"):
                folder_out = arg
            elif opt in ("-t"):
                path_template = arg
            elif opt in ("-s"):
                warp_spinal_levels = int(arg)
            elif opt in ('-v'):
                verbose = int(arg)
            elif opt in ("-w"):
                fname_transfo = arg

    # display usage if a mandatory argument is not provided
    if fname_src == '' or fname_transfo == '':
        usage()

    # Check file existence
    sct.printv('\nCheck file existence...', verbose)
    sct.check_file_exist(fname_src)
    sct.check_file_exist(fname_transfo)

    # add slash at the end of folder name (in case there is no slash)
    path_template = sct.slash_at_the_end(path_template, 1)
    folder_out = sct.slash_at_the_end(folder_out, 1)
    folder_template = sct.slash_at_the_end(folder_template, 1)
    folder_atlas = sct.slash_at_the_end(folder_atlas, 1)
    folder_spinal_levels = sct.slash_at_the_end(folder_spinal_levels, 1)

    # print arguments
    print '\nCheck parameters:'
    print '  Destination image ........ '+fname_src
    print '  Warping field ............ '+fname_transfo
    print '  Path template ............ '+path_template
    print '  Output folder ............ '+folder_out+'\n'

    # Extract path, file and extension
    path_src, file_src, ext_src = sct.extract_fname(fname_src)

    # create output folder
    if os.path.exists(folder_out):
        sct.printv('WARNING: Output folder already exists. Deleting it...', verbose)
        sct.run('rm -rf '+folder_out)
    sct.run('mkdir '+folder_out)

    # Warp template objects
    if warp_template == 1:
        sct.printv('\nWarp template objects...', verbose)
        warp_label(path_template, folder_template, param.file_info_label, fname_src, fname_transfo, folder_out)

    # Warp atlas
    if warp_atlas == 1:
        sct.printv('\nWarp atlas of white matter tracts...', verbose)
        warp_label(path_template, folder_atlas, param.file_info_label, fname_src, fname_transfo, folder_out)

    # Warp spinal levels
    if warp_spinal_levels == 1:
        sct.printv('\nWarp spinal levels...', verbose)
        warp_label(path_template, folder_spinal_levels, param.file_info_label, fname_src, fname_transfo, folder_out)

    # to view results
    sct.printv('\nDone! To view results, type:', verbose)
    sct.printv('fslview '+fname_src+' '+folder_out+folder_template+'MNI-Poly-AMU_T2.nii.gz -b 0,4000 '+folder_out+folder_template+'MNI-Poly-AMU_level.nii.gz -l MGH-Cortical -t 0.5 '+folder_out+folder_template+'MNI-Poly-AMU_GM.nii.gz -l Red-Yellow -b 0.5,1 '+folder_out+folder_template+'MNI-Poly-AMU_WM.nii.gz -l Blue-Lightblue -b 0.5,1 &\n', verbose, 'info')



# Warp labels
# ==========================================================================================
def warp_label(path_label, folder_label, file_label, fname_src, fname_transfo, path_out):
    # read label file and check if file exists
    sct.printv('\nRead label file...', param.verbose)
    template_label_ids, template_label_names, template_label_file = read_label_file(path_label+folder_label, file_label)
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



# Print usage
# ==========================================================================================
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>
DESCRIPTION
  This function warps the template and all atlases to a given image (e.g. fMRI, DTI, MTR, etc.).
USAGE
  """+os.path.basename(__file__)+""" -d <dest> -w <warp>
MANDATORY ARGUMENTS
  -d <dest>             destination image the template will be warped into
  -w <warp>             warping field
OPTIONAL ARGUMENTS
  -a {0,1}              warp atlas of white matter. Default="""+str(param_default.warp_atlas)+"""
  -s {0,1}              warp spinal levels. Default="""+str(param_default.warp_spinal_levels)+"""
  -o <folder_out>       name of output folder. Default="""+param_default.folder_out+"""
  -t <path_template>    Specify path to template data. Default="""+str(param_default.path_template)+"""
  -v {0,1}              verbose. Default="""+str(param_default.verbose)+"""
  -h                    help. Show this message
EXAMPLE
  """+os.path.basename(__file__)+""" -d dwi_mean.nii.gz -w warp_template2dmri.nii.gz -o label\n"""

    # exit program
    sys.exit(2)


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    param_default = Param()
    # call main function
    main()
