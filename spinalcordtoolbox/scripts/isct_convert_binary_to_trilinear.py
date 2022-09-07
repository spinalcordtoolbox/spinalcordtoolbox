#!/usr/bin/env python
#########################################################################################
#
# Convert binary spinal cord segmentation to trilinear-interpolated segmentation. Instead of simply re-interpolating
# the image, this function oversample the binary mask, then smooth along centerline (to remove step-effects), then
# downsample back to native resolution.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad
# Modified: 2014-07-06
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sys
import os
import getopt
import time

import numpy as np

from spinalcordtoolbox.image import Image, generate_output_file, convert
from spinalcordtoolbox.utils.sys import init_sct, __data_dir__, printv
from spinalcordtoolbox.utils.fs import tmp_create, check_file_exist, rmtree, extract_fname

from spinalcordtoolbox.scripts import sct_resample, sct_smooth_spinalcord


class Param:
    def __init__(self):
        self.debug = 0
        self.smoothing_sigma = 5
        self.interp_factor = 1  # interpolation factor. Works fine with 1 (i.e., no interpolation required).
        self.suffix = '_trilin'  # output suffix
        self.remove_temp_files = 1
        self.verbose = 1


# main
# =======================================================================================================================
def main():

    # Initialization
    fname_data = ''
    interp_factor = param.interp_factor
    remove_temp_files = param.remove_temp_files
    verbose = param.verbose
    suffix = param.suffix
    smoothing_sigma = param.smoothing_sigma

    # start timer
    start_time = time.time()

    # Parameters for debug mode
    if param.debug:
        fname_data = os.path.join(__data_dir__, 'sct_testing_data', 't2', 't2_seg.nii.gz')
        remove_temp_files = 0
        param.mask_size = 10
    else:
        # Check input parameters
        try:
            opts, args = getopt.getopt(sys.argv[1:], 'hi:v:r:s:')
        except getopt.GetoptError:
            usage()
            raise SystemExit(2)
        if not opts:
            usage()
            raise SystemExit(2)
        for opt, arg in opts:
            if opt == '-h':
                usage()
                return
            elif opt in ('-i'):
                fname_data = arg
            elif opt in ('-r'):
                remove_temp_files = int(arg)
            elif opt in ('-s'):
                smoothing_sigma = arg
            elif opt in ('-v'):
                verbose = int(arg)

    # display usage if a mandatory argument is not provided
    if fname_data == '':
        usage()
        raise SystemExit(2)

    # printv(arguments)
    printv('\nCheck parameters:')
    printv('  segmentation ........... ' + fname_data)
    printv('  interp factor .......... ' + str(interp_factor))
    printv('  smoothing sigma ........ ' + str(smoothing_sigma))

    # check existence of input files
    printv('\nCheck existence of input files...')
    check_file_exist(fname_data, verbose)

    # Extract path, file and extension
    path_data, file_data, ext_data = extract_fname(fname_data)

    path_tmp = tmp_create(basename="binary_to_trilinear")

    printv('\nCopying input data to tmp folder and convert to nii...', param.verbose)
    im_input = convert(Image(fname_data))
    im_input.save(os.path.join(path_tmp, "data.nii"), mutable=True, verbose=param.verbose)

    # go to tmp folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    # Get dimensions of data
    printv('\nGet dimensions of data...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = Image('data.nii').dim
    printv('.. ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz), verbose)

    # upsample data
    printv('\nUpsample data...', verbose)
    sct_resample.main([
        "-i", "data.nii",
        "-x", "linear",
        "-vox", str(nx * interp_factor) + 'x' + str(ny * interp_factor) + 'x' + str(nz * interp_factor),
        "-o", "data_up.nii",
    ])

    # Smooth along centerline
    printv('\nSmooth along centerline...', verbose)
    sct_smooth_spinalcord.main(["-i", "data_up.nii",
                                "-s", "data_up.nii",
                                "-smooth", str(smoothing_sigma),
                                "-r", str(remove_temp_files),
                                "-v", str(verbose), ])

    # downsample data
    printv('\nDownsample data...', verbose)
    sct_resample.main([
        "-i", "data_up_smooth.nii",
        "-x", "linear",
        "-vox", str(nx) + 'x' + str(ny) + 'x' + str(nz),
        "-o", "data_up_smooth_down.nii",
    ])

    # come back
    os.chdir(curdir)

    # Generate output files
    printv('\nGenerate output files...')
    fname_out = generate_output_file(os.path.join(path_tmp, "data_up_smooth_down.nii"), '' + file_data + suffix + ext_data)

    # Delete temporary files
    if remove_temp_files == 1:
        printv('\nRemove temporary files...')
        rmtree(path_tmp)

    # display elapsed time
    elapsed_time = time.time() - start_time
    printv('\nFinished! Elapsed time: ' + str(int(np.round(elapsed_time))) + 's')

    # to view results
    printv('\nTo view results, type:')
    printv('fslview ' + file_data + ' ' + file_data + suffix + ' &\n')


# printv(usage)
# ==========================================================================================
def usage():
    print('\n'
          '' + os.path.basename(__file__) + '\n'
          '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
          'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n'
          '\n'
          'DESCRIPTION\n'
          '  Convert binary spinal cord segmentation to trilinear-interpolated segmentation. Instead of simply\n'
          '  re-interpolating the image, this function oversamples the binary mask, smoothes along centerline\n'
          '  (to remove step-effects), then downsamples back to native resolution.\n'
          '\n'
          'USAGE\n'
          '  ' + os.path.basename(__file__) + ' -i <bin_seg>\n'
          '\n'
          'MANDATORY ARGUMENTS\n'
          '  -i <bin_seg>      binary segmentation of spinal cord\n'
          '\n'
          'OPTIONAL ARGUMENTS\n'
          '  -s                sigma of the smoothing Gaussian kernel (in voxel). Default=' + str(param_default.smoothing_sigma) + '\n'
          '  -r {0,1}          remove temporary files. Default=' + str(param_default.remove_temp_files) + '\n'
          '  -v {0,1}          verbose. Default=' + str(param_default.verbose) + '\n'
          '  -h                help. Show this message\n'
          '\n'
          'EXAMPLE\n'
          '  ' + os.path.basename(__file__) + ' -i segmentation.nii \n')


# =======================================================================================================================
# Start program
# =======================================================================================================================
if __name__ == "__main__":
    init_sct()
    # initialize parameters
    param = Param()
    param_default = Param()
    # call main function
    main()
