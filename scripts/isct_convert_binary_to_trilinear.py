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
import sct_utils as sct
from msct_image import Image


class Param:
    def __init__(self):
        self.debug = 0
        self.smoothing_sigma = 5
        self.interp_factor = 1  # interpolation factor. Works fine with 1 (i.e., no interpolation required).
        self.suffix = '_trilin'  # output suffix
        self.remove_temp_files = 1
        self.verbose = 1


# main
#=======================================================================================================================
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

    # get path of the toolbox
    path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))

    # Parameters for debug mode
    if param.debug:
        fname_data = os.path.join(path_sct, 'testing', 'data', 'errsm_23', 't2', 't2_manual_segmentation.nii.gz')
        remove_temp_files = 0
        param.mask_size = 10
    else:
        # Check input parameters
        try:
            opts, args = getopt.getopt(sys.argv[1:], 'hi:v:r:s:')
        except getopt.GetoptError:
            usage()
        if not opts:
            usage()
        for opt, arg in opts:
            if opt == '-h':
                usage()
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

    # sct.printv(arguments)
    sct.printv('\nCheck parameters:')
    sct.printv('  segmentation ........... ' + fname_data)
    sct.printv('  interp factor .......... ' + str(interp_factor))
    sct.printv('  smoothing sigma ........ ' + str(smoothing_sigma))

    # check existence of input files
    sct.printv('\nCheck existence of input files...')
    sct.check_file_exist(fname_data, verbose)

    # Extract path, file and extension
    path_data, file_data, ext_data = sct.extract_fname(fname_data)

    path_tmp = sct.tmp_create(basename="binary_to_trilinear", verbose=verbose)

    from sct_convert import convert
    sct.printv('\nCopying input data to tmp folder and convert to nii...', param.verbose)
    convert(fname_data, os.path.join(path_tmp, "data.nii"))

    # go to tmp folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    # Get dimensions of data
    sct.printv('\nGet dimensions of data...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = Image('data.nii').dim
    sct.printv('.. ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz), verbose)

    # upsample data
    sct.printv('\nUpsample data...', verbose)
    sct.run('sct_resample -i data.nii -x linear -vox ' + str(nx * interp_factor) + 'x' + str(ny * interp_factor) + 'x' + str(nz * interp_factor) + ' -o data_up.nii', verbose)

    # Smooth along centerline
    sct.printv('\nSmooth along centerline...', verbose)
    sct.run('sct_smooth_spinalcord -i data_up.nii -s data_up.nii' + ' -smooth ' + str(smoothing_sigma) + ' -r ' + str(remove_temp_files) + ' -v ' + str(verbose), verbose)

    # downsample data
    sct.printv('\nDownsample data...', verbose)
    sct.run('sct_resample -i data_up_smooth.nii -x linear -vox ' + str(nx) + 'x' + str(ny) + 'x' + str(nz) + ' -o data_up_smooth_down.nii', verbose)

    # come back
    os.chdir(curdir)

    # Generate output files
    sct.printv('\nGenerate output files...')
    fname_out = sct.generate_output_file(os.path.join(path_tmp, "data_up_smooth_down.nii"), '' + file_data + suffix + ext_data)

    # Delete temporary files
    if remove_temp_files == 1:
        sct.printv('\nRemove temporary files...')
        sct.run('rm -rf ' + path_tmp)

    # display elapsed time
    elapsed_time = time.time() - start_time
    sct.printv('\nFinished! Elapsed time: ' + str(int(round(elapsed_time))) + 's')

    # to view results
    sct.printv('\nTo view results, type:')
    sct.printv('fslview ' + file_data + ' ' + file_data + suffix + ' &\n')


# sct.printv(usage)
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

    # exit program
    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    sct.start_stream_logger()
    # initialize parameters
    param = Param()
    param_default = Param()
    # call main function
    main()
