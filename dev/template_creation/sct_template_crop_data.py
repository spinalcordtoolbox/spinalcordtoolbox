#!/usr/bin/env python
#########################################################################################
#
# crop template images
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
# Modified: 2014-11-14
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sys, io, os, getopt, glob, shutil

path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(path_sct, 'scripts'))

import sct_utils as sct

class Param:
    ## The constructor
    def __init__(self):
        self.debug = 0



#=======================================================================================================================
# main
#=======================================================================================================================
def main():
    # Initialization
    path_data = ''
    xmin = '50'
    xsize = '100'
    ymin = '0'
    ysize = '-1'
    zmin = '0'
    zsize = '-1'
    fsloutput = 'export FSLOUTPUTTYPE=NIFTI; ' # for faster processing, all outputs are in NIFTI

    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        path_data = '/Volumes/folder_shared/template/t2'
        path_out = '/Volumes/folder_shared/template/t2_crop'
    else:
        # Check input parameters
        try:
            opts, args = getopt.getopt(sys.argv[1:], 'hi:o:')
        except getopt.GetoptError:
            usage()
        if not opts:
            usage()
        for opt, arg in opts:
            if opt == '-h':
                usage()
            elif opt in ("-i"):
                path_data = arg
            elif opt in ("-o"):
                path_out = arg

    # check input folder
    sct.check_folder_exist(path_data)

    # create output folder
    if os.path.exists(path_out):
        sct.printv('WARNING: Output folder exists. Deleting it.', 1, 'warning')
        # remove dir
        shutil.rmtree(path_out)
    # create dir
    os.makedirs(path_out)

    # list all files in folder
    files = [f for f in glob.glob(os.path.join(path_data, '*.nii.gz'))]
    # for files in glob.glob(os.path.join(path_data, '*.nii.gz')):
    #     print files

    # crop files one by one (to inform user)
    for f in files:
        path_f, file_f, ext_f = sct.extract_fname(f)
        sct.run('fslroi '+f+' '+os.path.join(path_out, file_f)+' '+xmin+' '+xsize+' '+ymin+' '+ysize+' '+zmin+' '+zsize)

    # to view results
    print '\nDone!'


#=======================================================================================================================
# usage
#=======================================================================================================================
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  Crop template data.

USAGE
  """+os.path.basename(__file__)+"""

MANDATORY ARGUMENTS
  -i <path_in>          source path
  -o <path_out>         dest path

OPTIONAL ARGUMENTS

EXAMPLE
  """+os.path.basename(__file__)+"""\n"""

    #Exit Program
    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    param = Param()
    param_default = Param()
    # call main function
    main()
