#!/usr/bin/env python
#########################################################################################
#
# Warp spinal cord atlas to metric (DTI, MT, etc.).
#
# See Usage() below for more information.
#
#
# DEPENDENCIES
# ---------------------------------------------------------------------------------------
# EXTERNAL PYTHON PACKAGES
# none
#
# EXTERNAL SOFTWARE
# - itksnap/c3d <http://www.itksnap.org/pmwiki/pmwiki.php?n=Main.HomePage>
# - ants <http://stnava.github.io/ANTs/>
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
# Modified: 2014-05-03
#
# About the license: see the file LICENSE.TXT
#########################################################################################


# DEFAULT PARAMETERS
class param:
    ## The constructor
    def __init__(self):
        self.debug              = 0
        self.verbose            = 0 # verbose

import re
import sys
import commands
import getopt
import os
import time
import sct_utils as sct

# MAIN
# ==========================================================================================
def main():

    # Initialization
    fname_src = ''
    fname_transfo = ''
    path_out = 'atlas'
    verbose = param.verbose
    start_time = time.time()

    # get path of the toolbox
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
    print path_sct

    # Parameters for debug mode
    if param.debug:
        fname_src = os.path.expanduser("~")+'/code/spinalcordtoolbox_dev/testing/data/errsm_23/mt/mtr.nii.gz'
        fname_transfo = os.path.expanduser("~")+'/code/spinalcordtoolbox_dev/testing/data/errsm_23/template/warp_template2mt.nii.gz'
        verbose = 1

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],'hd:w:o:v:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ("-d"):
            fname_src = arg
        elif opt in ("-o"):
            path_out = arg
        elif opt in ("-w"):
            fname_transfo = arg
        elif opt in ('-v'):
            verbose = int(arg)

    # display usage if a mandatory argument is not provided
    if fname_src == '' or fname_transfo == '':
        usage()

    # check existence of input files
    sct.check_file_exist(fname_src)
    sct.check_file_exist(fname_transfo)

    # print arguments
    print '\nCheck parameters:'
    print '.. Metric image:         '+fname_src
    print '.. Transformation:       '+fname_transfo
    print '.. Output folder:        '+path_out

    # Extract path, file and extension
    path_src, file_src, ext_src = sct.extract_fname(fname_src)

    # create output folder
    if os.path.exists(path_out):
        sct.run('rm -rf '+path_out)
    sct.run('mkdir '+path_out)

    # get atlas files
    status, output = sct.run('ls '+path_sct+'/data/atlas/vol*.nii.gz')
    file_atlas_list = output.split()
    # Warp atlas
    for i in xrange (0,len(file_atlas_list)):
        path_atlas, file_atlas, ext_atlas = sct.extract_fname(file_atlas_list[i])
        sct.run('WarpImageMultiTransform 3 '+file_atlas_list[i]+' '+path_out+'/'+file_atlas+ext_atlas+' -R '+fname_src+' '+fname_transfo)
    # Copy list.txt
    sct.run('cp '+path_sct+'/data/atlas/list.txt '+path_out+'/')
    # Warp other template objects
    sct.run('WarpImageMultiTransform 3 '+path_sct+'/data/template/MNI-Poly-AMU_GM.nii.gz '+path_out+'/../gray_matter.nii.gz -R '+fname_src+' '+fname_transfo)
    sct.run('WarpImageMultiTransform 3 '+path_sct+'/data/template/MNI-Poly-AMU_WM.nii.gz '+path_out+'/../white_matter.nii.gz -R '+fname_src+' '+fname_transfo)
    sct.run('WarpImageMultiTransform 3 '+path_sct+'/data/template/MNI-Poly-AMU_level.nii.gz '+path_out+'/../vertebral_labeling.nii.gz -R '+fname_src+' --use-NN '+fname_transfo)
    sct.run('WarpImageMultiTransform 3 '+path_sct+'/data/template/MNI-Poly-AMU_CSF.nii.gz '+path_out+'/../csf.nii.gz -R '+fname_src+' --use-NN '+fname_transfo)


# Print usage
# ==========================================================================================
def usage():
    print '\n' \
        ''+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        '  This script warps all the spinal cord tracts of the atlas according to the warping field given as input.\n' \
        '\n' \
        'USAGE\n' \
        '  '+os.path.basename(__file__)+' -d <source> -w <dest>\n' \
        '\n' \
        'MANDATORY ARGUMENTS\n' \
        '  -d <dest>                  destination image\n' \
        '  -w <warping_field>         warping field\n' \
        '\n' \
        'OPTIONAL ARGUMENTS\n' \
        '  -o <output_folder>           output folder path (default=./atlas)\n' \
        '  -v <0,1>                     verbose. Default='+str(param.verbose)+' (not functional yet)\n'


    # exit program
    sys.exit(2)


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = param()
    # call main function
    main()
