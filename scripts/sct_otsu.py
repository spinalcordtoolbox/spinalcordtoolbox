#!/usr/bin/env python
#########################################################################################
#
# Segment images using OTSU algorithm.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad, Brian Avants
# Modified: 2014-10-08
#
# About the license: see the file LICENSE.TXT
#########################################################################################


import sys
import os
import getopt
import commands
import sct_utils as sct
import time
from sct_convert import convert


# DEFAULT PARAMETERS
class Param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.fname_data = ''
        self.threshold = '3'  # threshold value
        self.file_suffix = '_seg'  # output suffix
        self.verbose = 1
        self.remove_tmp_files = 1


# main
#=======================================================================================================================
def main():

    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        # get path of the testing data
        status, path_sct_data = commands.getstatusoutput('echo $SCT_TESTING_DATA_DIR')
        param.fname_data = path_sct_data+'/fmri/fmri.nii.gz'
        param.threshold = '3'
        param.remove_tmp_files = 0
        param.verbose = 1
    else:
        # Check input parameters
        try:
            opts, args = getopt.getopt(sys.argv[1:], 'hi:r:t:v:')
        except getopt.GetoptError:
            usage()
        if not opts:
            usage()
        for opt, arg in opts:
            if opt == '-h':
                usage()
            elif opt in '-i':
                param.fname_data = arg
            elif opt in '-r':
                param.remove_tmp_files = int(arg)
            elif opt in '-t':
                param.threshold = arg
            elif opt in '-v':
                param.verbose = int(arg)

    # run main program
    otsu()


# otsu
#=======================================================================================================================
def otsu():

    dim = 4  # by default, will be adjusted later

    # display usage if a mandatory argument is not provided
    if param.fname_data == '':
        sct.printv('ERROR: All mandatory arguments are not provided. See usage (add -h).', 1, 'error')

    # check existence of input files
    sct.printv('\ncheck existence of input files...', param.verbose)
    sct.check_file_exist(param.fname_data, param.verbose)

    # display input parameters
    sct.printv('\nInput parameters:', param.verbose)
    sct.printv('  data ..................'+param.fname_data, param.verbose)
    sct.printv('  threshold .............'+str(param.threshold), param.verbose)

    # Extract path/file/extension
    path_data, file_data, ext_data = sct.extract_fname(param.fname_data)
    path_out, file_out, ext_out = '', file_data, ext_data

    # create temporary folder
    sct.printv('\nCreate temporary folder...', param.verbose)
    path_tmp = sct.slash_at_the_end('tmp.'+time.strftime("%y%m%d%H%M%S"), 1)
    sct.run('mkdir '+path_tmp, param.verbose)

    # Copying input data to tmp folder and convert to nii
    # NB: cannot use c3d here because c3d cannot convert 4D data.
    sct.printv('\nCopying input data to tmp folder and convert to nii...', param.verbose)
    sct.run('cp '+param.fname_data+' '+path_tmp+'data'+ext_data, param.verbose)

    # go to tmp folder
    os.chdir(path_tmp)

    # convert to nii format
    convert('data'+ext_data, 'data.nii')
    # sct.run('fslchfiletype NIFTI data', param.verbose)

    # Get dimensions of data
    sct.printv('\nGet dimensions of data...', param.verbose)
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension('data.nii')
    sct.printv('  ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz)+ ' x ' + str(nt), param.verbose)
    if nt == 1:
        dim = 3
    if nz == 1:
        dim = 2

    # threshold images
    sct.run('sct_ThresholdImage '+str(dim)+' data.nii data_otsu.nii Otsu '+str(param.threshold))
    # binarize
    sct.run('sct_ThresholdImage '+str(dim)+' data_otsu.nii data_otsu_thr.nii '+str(param.threshold)+' '+str(param.threshold))
    # get largest component of binary mask
    sct.run('isct_ImageMath '+str(dim)+' data_otsu_thr.nii GetLargestComponent data_otsu_thr.nii')
    # Morphological Dilation
    sct.run('isct_ImageMath '+str(dim)+' data_otsu_thr.nii MD data_otsu_thr.nii')
    # Morphological Erosion
    sct.run('isct_ImageMath '+str(dim)+' data_otsu_thr.nii ME data_otsu_thr.nii')
    # Fill holes
    sct.run('isct_ImageMath '+str(dim)+' data_otsu_thr.nii FillHoles data_otsu_thr.nii')

    # come back to parent folder
    os.chdir('..')

    # Generate output files
    sct.printv('\nGenerate output files...', param.verbose)
    sct.generate_output_file(path_tmp+'data_otsu_thr.nii', path_out+file_out+param.file_suffix+ext_out)

    # Remove temporary files
    if param.remove_tmp_files == 1:
        print('\nRemove temporary files...')
        sct.run('rm -rf '+path_tmp, param.verbose)

    # to view results
    sct.printv('\nDone! To view results, type:', param.verbose)
    sct.printv('fslview '+path_out+file_out+param.file_suffix+ext_out+' &', param.verbose, 'code')
    print


# Print usage
# ==========================================================================================
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  Segment data using OTSU algorithm.

USAGE
  """+os.path.basename(__file__)+""" -i <data>

MANDATORY ARGUMENTS
  -i <data>        image to segment. Can be 2D, 3D or 4D.

OPTIONAL ARGUMENTS
  -t <int>         threshold value for images. Default="""+str(param.threshold)+"""
  -r {0,1}         remove temporary files. Default="""+str(param.remove_tmp_files)+"""
  -v {0,1}         verbose. Default="""+str(param.verbose)+"""
  -h               help. Show this message

EXAMPLE
  """+os.path.basename(__file__)+""" -i dwi.nii.gz -t 5\n"""

    # exit program
    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    param_default = Param()
    # call main function
    main()
