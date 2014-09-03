#!/usr/bin/env python
#########################################################################################
#
# Separate b=0 and DW images from diffusion dataset.
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
# Modified: 2014-08-14
#
# About the license: see the file LICENSE.TXT
#########################################################################################


import sys
import getopt
import os
import math
import time
import commands
import sct_utils as sct

class param:
    def __init__(self):
        self.debug = 0
        self.average = 0
        self.remove_tmp_files = 1
        self.verbose = 1


# MAIN
# ==========================================================================================
def main():

    # Initialization
    fsloutput = 'export FSLOUTPUTTYPE=NIFTI; ' # for faster processing, all outputs are in NIFTI
    fname_data = ''
    fname_bvecs = ''
    path_out = ''
    average = param.average
    verbose = param.verbose
    remove_tmp_files = param.remove_tmp_files
    start_time = time.time()

    # get path of the toolbox
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')

    # Parameters for debug mode
    if param.debug:
        fname_data = path_sct+'/testing/data/errsm_23/dmri/dmri.nii.gz'
        fname_bvecs = path_sct+'/testing/data/errsm_23/dmri/bvecs.txt'
        average = 1
        verbose = 1

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],'ha:b:i:o:r:v:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ("-a"):
            average = int(arg)
        elif opt in ("-b"):
            fname_bvecs = arg
        elif opt in ("-i"):
            fname_data = arg
        elif opt in ("-o"):
            path_out = arg
        elif opt in ("-r"):
            remove_temp_file = int(arg)
        elif opt in ('-v'):
            verbose = int(arg)

    # display usage if a mandatory argument is not provided
    if fname_data == '' or fname_bvecs == '':
        usage()

    # check existence of input files
    sct.check_file_exist(fname_data, verbose)
    sct.check_file_exist(fname_bvecs, verbose)

    # print arguments
    sct.printv('\nInput parameters:', verbose)
    sct.printv('  input file ............'+fname_data, verbose)
    sct.printv('  bvecs file ............'+fname_bvecs, verbose)
    sct.printv('  average ...............'+str(average), verbose)

    # Get full path
    fname_data = os.path.abspath(fname_data)
    fname_bvecs = os.path.abspath(fname_bvecs)

    # Extract path, file and extension
    path_data, file_data, ext_data = sct.extract_fname(fname_data)

    # # get output folder
    # if path_out == '':
    #     path_out = ''

    # create temporary folder
    sct.printv('\nCreate temporary folder...', verbose)
    path_tmp = sct.slash_at_the_end('tmp.'+time.strftime("%y%m%d%H%M%S"), 1)
    sct.run('mkdir '+path_tmp, verbose)

    # copy files into tmp folder
    sct.printv('\nCopy files into temporary folder...', verbose)
    sct.run('cp '+fname_data+' '+path_tmp+'dmri'+ext_data, verbose)
    sct.run('cp '+fname_bvecs+' '+path_tmp+'bvecs', verbose)

    # go to tmp folder
    os.chdir(path_tmp)

    # Get size of data
    sct.printv('\nGet dimensions data...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension('dmri'+ext_data)
    sct.printv('.. '+str(nx)+' x '+str(ny)+' x '+str(nz)+' x '+str(nt), verbose)

    # Open bvecs file
    sct.printv('\nOpen bvecs file...', verbose)
    bvecs = []
    with open('bvecs') as f:
        for line in f:
            bvecs_new = map(float, line.split())
            bvecs.append(bvecs_new)

    # Check if bvecs file is nx3
    if not len(bvecs[0][:]) == 3:
        sct.printv('  WARNING: bvecs file is 3xn instead of nx3. Consider using sct_dmri_transpose_bvecs.', verbose, 'warning')
        sct.printv('  Transpose bvecs...', verbose)
        # transpose bvecs
        bvecs = zip(*bvecs)

    # Identify b=0 and DWI images
    sct.printv('\nIdentify b=0 and DWI images...', verbose)
    index_b0 = []
    index_dwi = []
    for it in xrange(0, nt):
        if math.sqrt(math.fsum([i**2 for i in bvecs[it]])) < 0.01:
            index_b0.append(it)
        else:
            index_dwi.append(it)
    nb_b0 = len(index_b0)
    nb_dwi = len(index_dwi)
    sct.printv('  Number of b=0: '+str(nb_b0)+' '+str(index_b0), verbose)
    sct.printv('  Number of DWI: '+str(nb_dwi)+' '+str(index_dwi), verbose)

    # Split into T dimension
    sct.printv('\nSplit along T dimension...', verbose)
    sct.run(fsloutput+' fslsplit dmri dmri_T', verbose)

    # Merge b=0 images
    sct.printv('\nMerge b=0...', verbose)
    cmd = fsloutput + 'fslmerge -t b0'
    for iT in range(nb_b0):
        cmd = cmd + ' dmri_T' + str(index_b0[iT]).zfill(4)
    sct.run(cmd, verbose)

    # Average b=0 images
    if average:
        sct.printv('\nAverage b=0...', verbose)
        sct.run(fsloutput + 'fslmaths b0 -Tmean b0_mean', verbose)

    # Merge DWI
    sct.printv('\nMerge DWI...', verbose)
    cmd = fsloutput + 'fslmerge -t dwi'
    for iT in range(nb_dwi):
        cmd = cmd + ' dmri_T' + str(index_dwi[iT]).zfill(4)
    sct.run(cmd, verbose)

    # Average DWI images
    if average:
        sct.printv('\nAverage DWI...', verbose)
        sct.run(fsloutput + 'fslmaths dwi -Tmean dwi_mean', verbose)

    # come back to parent folder
    os.chdir('..')

    # Generate output files
    sct.printv('\nGenerate output files...', verbose)
    sct.generate_output_file(path_tmp+'b0.nii', path_out, 'b0', ext_data, verbose)
    sct.generate_output_file(path_tmp+'dwi.nii', path_out, 'dwi', ext_data, verbose)
    if average:
        sct.generate_output_file(path_tmp+'b0_mean.nii', path_out, 'b0_mean', ext_data, verbose)
        sct.generate_output_file(path_tmp+'dwi_mean.nii', path_out, 'dwi_mean', ext_data, verbose)

    # Remove temporary files
    if remove_tmp_files == 1:
        sct.printv('\nRemove temporary files...', verbose)
        sct.run('rm -rf '+path_tmp, verbose)

    # display elapsed time
    elapsed_time = time.time() - start_time
    sct.printv('\nFinished! Elapsed time: '+str(int(round(elapsed_time)))+'s', verbose)

    # to view results
    sct.printv('\nTo view results, type: ', verbose)
    if average:
        sct.printv('fslview b0 b0_mean dwi dwi_mean &\n', verbose)
    else:
        sct.printv('fslview b0 dwi &\n', verbose)


# Print usage
# ==========================================================================================
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  Separate b=0 and DW images from diffusion dataset.

USAGE
  """+os.path.basename(__file__)+""" -i <dmri> -b <bvecs>

MANDATORY ARGUMENTS
  -i <dmri>        diffusion data
  -b <bvecs>       bvecs file

OPTIONAL ARGUMENTS
  -a {0,1}         average b=0 and DWI data. Default="""+str(param.average)+"""
  -o <output>      output folder. Default = local folder.
  -v {0,1}         verbose. Default="""+str(param.verbose)+"""
  -r {0,1}         remove temporary files. Default="""+str(param.remove_tmp_files)+"""
  -h               help. Show this message

EXAMPLE
  """+os.path.basename(__file__)+""" -i dmri.nii.gz -b bvecs.txt -a 1\n"""

    #Exit Program
    sys.exit(2)


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = param()
    # call main function
    main()