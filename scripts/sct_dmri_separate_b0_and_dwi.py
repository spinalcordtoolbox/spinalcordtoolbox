#!/usr/bin/env python
#########################################################################################
#
# Separate b=0 and DW images from diffusion dataset.
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
# Modified: 2014-05-22
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
import getopt
import os
import math
import time
import sct_utils as sct



# MAIN
# ==========================================================================================
def main():

    # Initialization
    path_script = os.path.dirname(__file__)
    fsloutput = 'export FSLOUTPUTTYPE=NIFTI; ' # for faster processing, all outputs are in NIFTI
    # THIS DOES NOT WORK IN MY LAPTOP: path_sct = os.environ['SCT_DIR'] # path to spinal cord toolbox
    path_sct = path_script[:-8] # TODO: make it cleaner!
    fname_data = ''
    fname_bvecs = ''
    verbose = param.verbose
    start_time = time.time()

    # Parameters for debug mode
    if param.debug:
        fname_data = os.path.expanduser("~")+'/code/spinalcordtoolbox_dev/testing/data/errsm_22/dmri/dmri.nii.gz'
        fname_bvecs = os.path.expanduser("~")+'/code/spinalcordtoolbox_dev/testing/data/errsm_22/dmri/bvecs.txt'
        verbose = 1

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],'hb:i:v:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ("-b"):
            fname_bvecs = arg
        elif opt in ("-i"):
            fname_data = arg
        elif opt in ('-v'):
            verbose = int(arg)

    # display usage if a mandatory argument is not provided
    if fname_data == '' or fname_bvecs == '':
        usage()

    # check existence of input files
    sct.check_file_exist(fname_data)
    sct.check_file_exist(fname_bvecs)

    # print arguments
    print '\nCheck parameters:'
    print '.. DWI data:             '+fname_data
    print '.. bvecs file:           '+fname_bvecs

    # Extract path, file and extension
    path_data, file_data, ext_data = sct.extract_fname(fname_data)

    # create temporary folder
    path_tmp = 'tmp.'+time.strftime("%y%m%d%H%M%S")
    sct.run('mkdir '+path_tmp)

    # copy files into tmp folder
    sct.run('cp '+fname_data+' '+path_tmp)
    sct.run('cp '+fname_bvecs+' '+path_tmp)

    # go to tmp folder
    os.chdir(path_tmp)

    # Get size of data
    print '\nGet dimensions data...'
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_data)
    print '.. '+str(nx)+' x '+str(ny)+' x '+str(nz)+' x '+str(nt)

    # Open bvecs file
    bvecs = []
    with open(fname_bvecs) as f:
        for line in f:
            bvecs_new = map(float, line.split())
            bvecs.append(bvecs_new)

    # Check if bvecs file is nx3
    if not len(bvecs[0][:]) == 3:
        print 'WARNING: bvecs file is 3xn instead of nx3. Consider using sct_dmri_transpose_bvecs'
        # transpose bvecs
        bvecs = zip(*bvecs)

    # Identify b=0 and DW images
    print '\nIdentify b=0 and DW images...'
    index_b0 = []
    index_dwi = []
    for it in xrange(0,nt):
        if math.sqrt(math.fsum([i**2 for i in bvecs[it]])) < 0.01:
            index_b0.append(it)
        else:
            index_dwi.append(it)
    nb_b0 = len(index_b0)
    nb_dwi = len(index_dwi)
    print '.. Number of b=0: '+str(nb_b0)+' '+str(index_b0)
    print '.. Number of DWI: '+str(nb_dwi)+' '+str(index_dwi)

    #TODO: check if number of bvecs and nt match

    # Split into T dimension
    print '\nSplit along T dimension...'
    sct.run(fsloutput+' fslsplit '+fname_data+' data_splitT')

    # retrieve output names
    status, output = sct.run('ls data_splitT*.*')
    file_data_split = output.split()
    # Remove .nii extension
    file_data_split = [file_data_split[i].replace('.nii','') for i in xrange (0,len(file_data_split))]

    # Merge b=0 images
    print '\nMerge b=0...'
    cmd = fsloutput+'fslmerge -t b0'
    for it in xrange(0,nb_b0):
        cmd += ' '+file_data_split[index_b0[it]]
    sct.run(cmd)

    # Merge DWI images
    print '\nMerge DWI...'
    cmd = fsloutput+'fslmerge -t dwi'
    for it in xrange(0,nb_dwi):
        cmd += ' '+file_data_split[index_dwi[it]]
    sct.run(cmd)

    # come back to parent folder
    os.chdir('..')

    # Generate output files
    print('\nGenerate output files...')
    sct.generate_output_file(path_tmp+'/b0.nii',path_data,'b0',ext_data)
    sct.generate_output_file(path_tmp+'/dwi.nii',path_data,'dwi',ext_data)

    # Remove temporary files
    print('\nRemove temporary files...')
    sct.run('rm -rf '+path_tmp)

    # display elapsed time
    elapsed_time = time.time() - start_time
    print '\nFinished! Elapsed time: '+str(int(round(elapsed_time)))+'s'

    # to view results
    print '\nTo view results, type:'
    print 'fslview b0 dwi &\n'

    # End of Main


# Print usage
# ==========================================================================================
def usage():
    print '\n' \
        ''+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        '  Separate b=0 and DW images from diffusion dataset.\n' \
        '\n' \
        'USAGE\n' \
        '  '+os.path.basename(__file__)+' -i <dmri> -b <bvecs>\n' \
        '\n' \
        'MANDATORY ARGUMENTS\n' \
        '  -i <dmri>                  diffusion data\n' \
        '  -b <bvecs>                 bvecs file\n' \
        '\n' \
        'OPTIONAL ARGUMENTS\n' \
        '  -v <0,1>                   verbose. Default='+str(param.verbose)+'.\n'

    # exit program
    sys.exit(2)


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = param()
    # call main function
    main()