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
import math
import time

import os
import commands
import sct_utils as sct
from msct_image import Image
from sct_image import split_data
from msct_parser import Parser


class Param:
    def __init__(self):
        self.debug = 0
        self.average = 0
        self.remove_tmp_files = 1
        self.verbose = 1
        self.bval_min = 100  # in case user does not have min bvalues at 0, set threshold.


# MAIN
# ==========================================================================================
def main(fname_data, fname_bvecs, fname_bvals, path_out, average, verbose, remove_tmp_files):

    # Initialization
    start_time = time.time()

    # sct.printv(arguments)
    sct.printv('\nInput parameters:', verbose)
    sct.printv('  input file ............' + fname_data, verbose)
    sct.printv('  bvecs file ............' + fname_bvecs, verbose)
    sct.printv('  bvals file ............' + fname_bvals, verbose)
    sct.printv('  average ...............' + str(average), verbose)

    # Get full path
    fname_data = os.path.abspath(fname_data)
    fname_bvecs = os.path.abspath(fname_bvecs)
    if fname_bvals:
        fname_bvals = os.path.abspath(fname_bvals)

    # Extract path, file and extension
    path_data, file_data, ext_data = sct.extract_fname(fname_data)

    # # get output folder
    # if path_out == '':
    #     path_out = ''

    # create temporary folder
    sct.printv('\nCreate temporary folder...', verbose)
    path_tmp = sct.slash_at_the_end('tmp.' + time.strftime("%y%m%d%H%M%S"), 1)
    sct.run('mkdir ' + path_tmp, verbose)

    # copy files into tmp folder and convert to nifti
    sct.printv('\nCopy files into temporary folder...', verbose)
    ext = '.nii'
    dmri_name = 'dmri'
    b0_name = 'b0'
    b0_mean_name = b0_name + '_mean'
    dwi_name = 'dwi'
    dwi_mean_name = dwi_name + '_mean'

    from sct_convert import convert
    if not convert(fname_data, path_tmp + dmri_name + ext):
        sct.printv('ERROR in convert.', 1, 'error')
    sct.run('cp ' + fname_bvecs + ' ' + path_tmp + 'bvecs', verbose)

    # go to tmp folder
    os.chdir(path_tmp)

    # Get size of data
    im_dmri = Image(dmri_name + ext)
    sct.printv('\nGet dimensions data...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = im_dmri.dim
    sct.printv('.. ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz) + ' x ' + str(nt), verbose)

    # Identify b=0 and DWI images
    sct.printv(fname_bvals)
    index_b0, index_dwi, nb_b0, nb_dwi = identify_b0(fname_bvecs, fname_bvals, param.bval_min, verbose)

    # Split into T dimension
    sct.printv('\nSplit along T dimension...', verbose)
    im_dmri_split_list = split_data(im_dmri, 3)
    for im_d in im_dmri_split_list:
        im_d.save()

    # Merge b=0 images
    sct.printv('\nMerge b=0...', verbose)
    cmd = 'sct_image -concat t -o ' + b0_name + ext + ' -i '
    for it in range(nb_b0):
        cmd = cmd + dmri_name + '_T' + str(index_b0[it]).zfill(4) + ext + ','
    cmd = cmd[:-1]  # remove ',' at the end of the string
    # WARNING: calling concat_data in python instead of in command line causes a non understood issue
    status, output = sct.run(cmd, param.verbose)

    # Average b=0 images
    if average:
        sct.printv('\nAverage b=0...', verbose)
        sct.run('sct_maths -i ' + b0_name + ext + ' -o ' + b0_mean_name + ext + ' -mean t', verbose)

    # Merge DWI
    cmd = 'sct_image -concat t -o ' + dwi_name + ext + ' -i '
    for it in range(nb_dwi):
        cmd = cmd + dmri_name + '_T' + str(index_dwi[it]).zfill(4) + ext + ','
    cmd = cmd[:-1]  # remove ',' at the end of the string
    # WARNING: calling concat_data in python instead of in command line causes a non understood issue
    status, output = sct.run(cmd, param.verbose)

    # Average DWI images
    if average:
        sct.printv('\nAverage DWI...', verbose)
        sct.run('sct_maths -i ' + dwi_name + ext + ' -o ' + dwi_mean_name + ext + ' -mean t', verbose)
        # if not average_data_across_dimension('dwi.nii', 'dwi_mean.nii', 3):
        #     sct.printv('ERROR in average_data_across_dimension', 1, 'error')
        # sct.run(fsloutput + 'fslmaths dwi -Tmean dwi_mean', verbose)

    # come back to parent folder
    os.chdir('..')

    # Generate output files
    sct.printv('\nGenerate output files...', verbose)
    sct.generate_output_file(path_tmp + b0_name + ext, path_out + b0_name + ext_data, verbose)
    sct.generate_output_file(path_tmp + dwi_name + ext, path_out + dwi_name + ext_data, verbose)
    if average:
        sct.generate_output_file(path_tmp + b0_mean_name + ext, path_out + b0_mean_name + ext_data, verbose)
        sct.generate_output_file(path_tmp + dwi_mean_name + ext, path_out + dwi_mean_name + ext_data, verbose)

    # Remove temporary files
    if remove_tmp_files == 1:
        sct.printv('\nRemove temporary files...', verbose)
        sct.run('rm -rf ' + path_tmp, verbose)

    # display elapsed time
    elapsed_time = time.time() - start_time
    sct.printv('\nFinished! Elapsed time: ' + str(int(round(elapsed_time))) + 's', verbose)

    # to view results
    sct.printv('\nTo view results, type: ', verbose)
    if average:
        sct.printv('fslview b0 b0_mean dwi dwi_mean &\n', verbose)
    else:
        sct.printv('fslview b0 dwi &\n', verbose)


# ==========================================================================================
# identify b=0 and DW images
# ==========================================================================================
def identify_b0(fname_bvecs, fname_bvals, bval_min, verbose):

    # Identify b=0 and DWI images
    sct.printv('\nIdentify b=0 and DWI images...', verbose)
    index_b0 = []
    index_dwi = []

    # if bval is not provided
    if not fname_bvals:
        # Open bvecs file
        #sct.printv('\nOpen bvecs file...', verbose)
        bvecs = []
        with open(fname_bvecs) as f:
            for line in f:
                bvecs_new = map(float, line.split())
                bvecs.append(bvecs_new)

        # Check if bvecs file is nx3
        if not len(bvecs[0][:]) == 3:
            sct.printv('  WARNING: bvecs file is 3xn instead of nx3. Consider using sct_dmri_transpose_bvecs.', verbose, 'warning')
            sct.printv('  Transpose bvecs...', verbose)
            # transpose bvecs
            bvecs = zip(*bvecs)

        # get number of lines
        nt = len(bvecs)

        # identify b=0 and dwi
        for it in xrange(0, nt):
            if math.sqrt(math.fsum([i**2 for i in bvecs[it]])) < 0.01:
                index_b0.append(it)
            else:
                index_dwi.append(it)

    # if bval is provided
    else:

        # Open bvals file
        from dipy.io import read_bvals_bvecs
        bvals, bvecs = read_bvals_bvecs(fname_bvals, fname_bvecs)

        # get number of lines
        nt = len(bvals)

        # Identify b=0 and DWI images
        sct.printv('\nIdentify b=0 and DWI images...', verbose)
        for it in xrange(0, nt):
            if bvals[it] < bval_min:
                index_b0.append(it)
            else:
                index_dwi.append(it)

    # check if no b=0 images were detected
    if index_b0 == []:
        sct.printv('ERROR: no b=0 images detected. Maybe you are using non-null low bvals? in that case use flag -bvalmin. Exit program.', 1, 'error')
        sys.exit(2)

    # display stuff
    nb_b0 = len(index_b0)
    nb_dwi = len(index_dwi)
    sct.printv('  Number of b=0: ' + str(nb_b0) + ' ' + str(index_b0), verbose)
    sct.printv('  Number of DWI: ' + str(nb_dwi) + ' ' + str(index_dwi), verbose)

    # return
    return index_b0, index_dwi, nb_b0, nb_dwi


# sct.printv(usage)
# ==========================================================================================
def usage():
    print("""
{0}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  Separate b=0 and DW images from diffusion dataset.

USAGE
  {0} -i <dmri> -b <bvecs>

MANDATORY ARGUMENTS
  -i <dmri>        diffusion data
  -b <bvecs>       bvecs file

OPTIONAL ARGUMENTS
  -a {0,1}         average b=0 and DWI data. Default={1}
  -m <bvals>       bvals file. Used to identify low b-values (in case different from 0).
  -o <output>      output folder. Default = local folder.
  -v {0,1}         verbose. Default={2}
  -r {0,1}         remove temporary files. Default={3}
  -h               help. Show this message

EXAMPLE
  {0} -i dmri.nii.gz -b bvecs.txt -a 1\n""".format(
        os.path.basename(__file__)), param_default.average, param_default.verbose, param_default.remove_tmp_files)

    # Exit Program
    sys.exit(2)


def get_parser():
    # Initialize parser
    param_default = Param()
    parser = Parser(__file__)

    # Mandatory arguments
    parser.usage.set_description("Separate b=0 and DW images from diffusion dataset.")
    parser.add_option(name='-i',
                      type_value='image_nifti',
                      description='Diffusion data',
                      mandatory=True,
                      example='dmri.nii.gz')
    parser.add_option(name='-b',
                      type_value='file',
                      description='bvecs file',
                      mandatory=False,
                      example='bvecs.txt',
                      deprecated_by='-bvec')
    parser.add_option(name='-bvec',
                      type_value='file',
                      description='bvecs file',
                      mandatory=True,
                      example='bvecs.txt')

    # Optional arguments
    parser.add_option(name='-a',
                      type_value='multiple_choice',
                      description='average b=0 and DWI data.',
                      mandatory=False,
                      example=['0', '1'],
                      default_value=str(param_default.average))
    parser.add_option(name='-m',
                      type_value='file',
                      description='bvals file. Used to identify low b-values (in case different from 0).',
                      mandatory=False,
                      deprecated_by='-bval')
    parser.add_option(name='-bval',
                      type_value='file',
                      description='bvals file. Used to identify low b-values (in case different from 0).',
                      mandatory=False)
    parser.add_option(name='-bvalmin',
                      type_value='float',
                      description='B-value threshold (in s/mm2) below which data is considered as b=0.',
                      mandatory=False,
                      example='50')
    parser.add_option(name='-o',
                      type_value='folder_creation',
                      description='Output folder.',
                      mandatory=False,
                      default_value='./',
                      deprecated_by='-ofolder')
    parser.add_option(name='-ofolder',
                      type_value='folder_creation',
                      description='Output folder.',
                      mandatory=False,
                      default_value='./')
    parser.add_option(name='-v',
                      type_value='multiple_choice',
                      description='Verbose.',
                      mandatory=False,
                      example=['0', '1'],
                      default_value=str(param_default.verbose))
    parser.add_option(name='-r',
                      type_value='multiple_choice',
                      description='remove temporary files.',
                      mandatory=False,
                      example=['0', '1'],
                      default_value=str(param_default.remove_tmp_files))

    return parser

# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.start_stream_logger()
    # initialize parameters
    param = Param()
    param_default = Param()
    # call main function
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])

    fname_data = arguments['-i']
    fname_bvecs = arguments['-bvec']

    fname_bvals = ''
    path_out = ''
    average = param.average
    verbose = param.verbose
    remove_tmp_files = param.remove_tmp_files

    if '-bval' in arguments:
        fname_bvals = arguments['-bval']
    if '-bvalmin' in arguments:
        param.bval_min = arguments['-bvalmin']
    if '-a' in arguments:
        average = arguments['-a']
    if '-ofolder' in arguments:
        path_out = arguments['-ofolder']
    if '-v' in arguments:
        verbose = int(arguments['-v'])
    if '-r' in arguments:
        remove_tmp_files = int(arguments['-r'])

    main(fname_data, fname_bvecs, fname_bvals, path_out, average, verbose, remove_tmp_files)
