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


# DEFAULT PARAMETERS
class param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.folder_out = 'template/'  # name of output folder
        self.warp_atlas = 1
        self.warp_spinal_levels = 0
        self.verbose = 1  # verbose


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
    folder_out = ''
    warp_atlas = param.warp_atlas
    warp_spinal_levels = param.warp_spinal_levels
    verbose = param.verbose
    start_time = time.time()

    # get path of the toolbox
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')

    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        fname_src = path_sct+'/testing/data/errsm_23/mt/mtr.nii.gz'
        fname_transfo = path_sct+'/testing/data/errsm_23/template/warp_template2mt.nii.gz'
        warp_atlas = 0
        warp_spinal_levels = 1
        verbose = 1

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'ha:d:w:o:s:v:')
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

    # Get output folder
    if folder_out == '':
        folder_out = param.folder_out  # folder created in user's current directory

    # add slash at the end of folder name (in case there is no slash)
    folder_out = sct.slash_at_the_end(folder_out, 1)

    # print arguments
    print '\nCheck parameters:'
    print '  Destination image ........ '+fname_src
    print '  Warping field ............ '+fname_transfo
    print '  Output folder ............ '+folder_out+'\n'

    # Extract path, file and extension
    path_src, file_src, ext_src = sct.extract_fname(fname_src)

    # create output folder
    if os.path.exists(folder_out):
        sct.printv('WARNING: Output folder already exists. Deleting it...', verbose)
        sct.run('rm -rf '+folder_out)
    sct.run('mkdir '+folder_out)


    # Warp template objects
    sct.printv('\nWarp template objects...', verbose)
    sct.run('WarpImageMultiTransform 3 '+path_sct+'/data/template/MNI-Poly-AMU_T2.nii.gz '+folder_out+'t2.nii.gz -R '+fname_src+' '+fname_transfo)
    sct.run('WarpImageMultiTransform 3 '+path_sct+'/data/template/MNI-Poly-AMU_GM.nii.gz '+folder_out+'gray_matter.nii.gz -R '+fname_src+' '+fname_transfo)
    sct.run('WarpImageMultiTransform 3 '+path_sct+'/data/template/MNI-Poly-AMU_WM.nii.gz '+folder_out+'white_matter.nii.gz -R '+fname_src+' '+fname_transfo)
    sct.run('WarpImageMultiTransform 3 '+path_sct+'/data/template/MNI-Poly-AMU_level.nii.gz '+folder_out+'vertebral_labeling.nii.gz -R '+fname_src+' --use-NN '+fname_transfo)
    sct.run('WarpImageMultiTransform 3 '+path_sct+'/data/template/MNI-Poly-AMU_CSF.nii.gz '+folder_out+'csf.nii.gz -R '+fname_src+' --use-NN '+fname_transfo)

    # Warp atlas
    if warp_atlas == 1:
        sct.printv('\nWarp atlas of white matter tracts...', verbose)
        # create output folder
        sct.run('mkdir '+folder_out+'atlas/')
        # get atlas files
        status, output = sct.run('ls '+path_sct+'/data/atlas/*.nii.gz', verbose)
        fname_list = output.split()
        # Warp atlas
        for i in xrange(0, len(fname_list)):
            path_list, file_list, ext_list = sct.extract_fname(fname_list[i])
            sct.run('WarpImageMultiTransform 3 '+fname_list[i]+' '+folder_out+'atlas/'+file_list+ext_list+' -R '+fname_src+' '+fname_transfo)
        # Copy list.txt
        sct.run('cp '+path_sct+'/data/atlas/list.txt '+folder_out+'atlas/')

    # Warp spinal levels
    if warp_spinal_levels == 1:
        sct.printv('\nWarp spinal levels...', verbose)
        # create output folder
        sct.run('mkdir '+folder_out+'spinal_levels/', verbose)
        # get spinal level files
        status, output = sct.run('ls '+path_sct+'/data/spinal_level/*.nii.gz', verbose)
        fname_list = output.split()
        # Warp levels
        for i in xrange(0, len(fname_list)):
            path_list, file_list, ext_list = sct.extract_fname(fname_list[i])
            sct.run('WarpImageMultiTransform 3 '+fname_list[i]+' '+folder_out+'spinal_levels/'+file_list+ext_list+' -R '+fname_src+' '+fname_transfo)

    # to view results
    print '\nDone! To view results, type:'
    print 'fslview '+fname_src+' '+folder_out+'t2.nii.gz '+folder_out+'gray_matter.nii.gz '+folder_out+'white_matter.nii.gz '+folder_out+'vertebral_labeling.nii.gz '+folder_out+'csf.nii.gz '+' &'
    print ''


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
  -a {0,1}              warp atlas of white matter. Default="""+str(param.warp_atlas)+"""
  -s {0,1}              warp spinal levels. Default="""+str(param.warp_spinal_levels)+"""
  -o <folder_out>       name of output folder. Default="""+param.folder_out+"""
  -v {0,1}              verbose. Default="""+str(param.verbose)+"""
  -h                    help. Show this message

EXAMPLE
  """+os.path.basename(__file__)+""" -d dwi_mean.nii.gz -w warp_template2dmri.nii.gz -o template -s 1\n"""

    # exit program
    sys.exit(2)


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = param()
    # call main function
    main()
