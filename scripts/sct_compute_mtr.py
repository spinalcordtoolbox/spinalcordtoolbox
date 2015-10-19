#!/usr/bin/env python
#########################################################################################
#
# Compute magnetization transfer ratio (MTR).
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad
# Modified: 2014-09-21
#
# About the license: see the file LICENSE.TXT
#########################################################################################


import sys
import os
import getopt
import commands
import sct_utils as sct
import time


# DEFAULT PARAMETERS
class Param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        # self.register = 1
        self.verbose = 1
        self.file_out = 'mtr'
        self.remove_tmp_files = 1


# main
#=======================================================================================================================
def main():

    # Initialization
    fname_mt0 = ''
    fname_mt1 = ''
    file_out = param.file_out
    # register = param.register
    fsloutput = 'export FSLOUTPUTTYPE=NIFTI; ' # for faster processing, all outputs are in NIFTI
    remove_tmp_files = param.remove_tmp_files
    verbose = param.verbose

    # get path of the toolbox
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')

    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        fname_mt0 = path_sct+'/testing/data/errsm_23/mt/mt0.nii.gz'
        fname_mt1 = path_sct+'/testing/data/errsm_23/mt/mt1.nii.gz'
        register = 1
        verbose = 1
    else:
        # Check input parameters
        try:
            opts, args = getopt.getopt(sys.argv[1:], 'hi:j:r:v:')
        except getopt.GetoptError:
            usage()
        if not opts:
            usage()
        for opt, arg in opts:
            if opt == '-h':
                usage()
            elif opt in '-i':
                fname_mt0 = arg
            elif opt in '-j':
                fname_mt1 = arg
            elif opt in '-r':
                remove_tmp_files = int(arg)
            elif opt in '-v':
                verbose = int(arg)
            # elif opt in '-x':
            #     register = int(arg)

    # display usage if a mandatory argument is not provided
    if fname_mt0 == '' or fname_mt1 == '':
        sct.printv('ERROR: All mandatory arguments are not provided. See usage (add -h).', 1, 'error')

    # display input parameters
    sct.printv('\nInput parameters:', verbose)
    sct.printv('  mt0 ...................'+fname_mt0, verbose)
    sct.printv('  mt1 ...................'+fname_mt1, verbose)

    # check existence of input files
    sct.printv('\ncheck existence of input files...', verbose)
    sct.check_file_exist(fname_mt0, verbose)
    sct.check_file_exist(fname_mt1, verbose)

    # Extract path/file/extension
    path_mt0, file_mt0, ext_mt0 = sct.extract_fname(fname_mt0)
    path_out, file_out, ext_out = '', file_out, ext_mt0

    # create temporary folder
    sct.printv('\nCreate temporary folder...', verbose)
    path_tmp = sct.slash_at_the_end('tmp.'+time.strftime("%y%m%d%H%M%S"), 1)
    sct.run('mkdir '+path_tmp, verbose)

    # Copying input data to tmp folder and convert to nii
    sct.printv('\nCopying input data to tmp folder and convert to nii...', verbose)
    from sct_convert import convert
    convert(fname_mt0, path_tmp+'mt0.nii', type='float32')
    convert(fname_mt1, path_tmp+'mt1.nii', type='float32')

    # go to tmp folder
    os.chdir(path_tmp)

    # compute MTR
    sct.printv('\nCompute MTR...', verbose)
    from msct_image import Image
    nii_mt1 = Image('mt1.nii')
    data_mt1 = nii_mt1.data
    data_mt0 = Image('mt0.nii').data
    data_mtr = 100 * (data_mt0 - data_mt1) / data_mt0
    # save MTR file
    nii_mtr = nii_mt1
    nii_mtr.data = data_mtr
    nii_mtr.setFileName('mtr.nii')
    nii_mtr.save()
    # sct.run(fsloutput+'fslmaths -dt double mt0.nii -sub mt1.nii -mul 100 -div mt0.nii -thr 0 -uthr 100 mtr.nii', verbose)

    # come back to parent folder
    os.chdir('..')

    # Generate output files
    sct.printv('\nGenerate output files...', verbose)
    sct.generate_output_file(path_tmp+'mtr.nii', path_out+file_out+ext_out)

    # Remove temporary files
    if remove_tmp_files == 1:
        print('\nRemove temporary files...')
        sct.run('rm -rf '+path_tmp)

    # to view results
    sct.printv('\nDone! To view results, type:', verbose)
    sct.printv('fslview '+fname_mt0+' '+fname_mt1+' '+file_out+' &\n', verbose, 'info')


# Print usage
# ==========================================================================================
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  Compute magnetization transfer ratio (MTR). Output is given in percentage.

USAGE
  """+os.path.basename(__file__)+""" -i <mt0> -j <mt1>

MANDATORY ARGUMENTS
  -i <mt0>         image without MT pulse
  -j <mt1>         image with MT pulse

OPTIONAL ARGUMENTS
  -r {0,1}         remove temporary files. Default="""+str(param_default.remove_tmp_files)+"""
  -v {0,1}         verbose. Default="""+str(param_default.verbose)+"""
  -h               help. Show this message

EXAMPLE
  """+os.path.basename(__file__)+""" -i mt0.nii.gz -j mt1.nii.gz\n"""

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
