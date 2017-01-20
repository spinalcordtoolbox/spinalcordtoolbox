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

import getopt
import os
import shutil
import sys
import time

import msct_image
import sct_convert
import sct_resample
import sct_smooth_spinalcord
import sct_utils as sct


class Param(object):
    def __init__(self):
        self.debug = 0
        self.smoothing_sigma = 5
        # interpolation factor. Works fine with 1 (i.e., no interpolation required).
        self.interp_factor = 1
        self.suffix = '_trilin'
        self.remove_temp_files = 1
        self.verbose = 1


def main(args=None):

    param = Param()

    if not args:
        args = sys.argv[1:]
    else:
        script_name =os.path.splitext(os.path.basename(__file__))[0]
        sct.printv('{0} {1}'.format(script_name, " ".join(args)))

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
    path_sct = os.environ.get('SCT_DIR')

    # Parameters for debug mode
    if param.debug:
        fname_data = path_sct + '/testing/data/errsm_23/t2/t2_manual_segmentation.nii.gz'
        remove_temp_files = 0
        param.mask_size = 10
    else:
        # Check input parameters
        try:
            opts, args = getopt.getopt(args, 'hi:v:r:s:')
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

    # print arguments
    sct.printv('\nCheck parameters:')
    sct.printv('  segmentation ........... ' + fname_data)
    sct.printv('  interp factor .......... ' + str(interp_factor))
    sct.printv('  smoothing sigma ........ ' + str(smoothing_sigma))

    # check existence of input files
    sct.printv('\nCheck existence of input files...')
    sct.check_file_exist(fname_data, verbose)

    # Extract path, file and extension
    path_data, file_data, ext_data = sct.extract_fname(fname_data)

    # create temporary folder
    sct.printv('\nCreate temporary folder...')
    path_tmp = 'tmp.' + time.strftime("%y%m%d%H%M%S")
    if os.path.exists(path_tmp):
        os.makedirs(path_tmp)

    sct.printv('\nCopying input data to tmp folder and convert to nii...', param.verbose)
    sct_convert.convert(fname_data, path_tmp + '/data.nii')

    # go to tmp folder
    os.chdir(path_tmp)

    # Get dimensions of data
    sct.printv('\nGet dimensions of data...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = msct_image.Image('data.nii').dim
    sct.printv('.. ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz), verbose)

    # upsample data
    sct.printv('\nUpsample data...', verbose)
    sct_resample.main(args=[
        '-i', 'data.nii', '-x', 'linear'
        '-vox', '{0}x{1}x{2}'.format(nx * interp_factor, ny * interp_factor, nz * interp_factor), '-o', 'data_up.nii'
    ])

    # Smooth along centerline
    sct.printv('\nSmooth along centerline...', verbose)
    sct_smooth_spinalcord.main(args=[
        '-i', 'data_up.nii', '-s', 'data_up.nii', '-smooth', '{0}'.format(smoothing_sigma), '-r',
        '{0}'.format(remove_temp_files), '-v', '{0}'.format(verbose)
    ])

    # downsample data
    sct.printv('\nDownsample data...', verbose)
    sct_resample.main(args=[
        '-i', 'data_up_smooth.nii', '-x', 'linear', '-vox', '{0}x{1}x{2}'.format(nx, ny, nz), '-o',
        'data_up_smooth_down.nii'
    ])

    # come back to parent folder
    os.chdir('..')

    # Generate output files
    sct.printv('\nGenerate output files...')
    sct.generate_output_file(path_tmp + '/data_up_smooth_down.nii', '' + file_data + suffix + ext_data)

    # Delete temporary files
    if remove_temp_files == 1:
        print '\nRemove temporary files...'
        shutil.rmtree(path_tmp, ignore_errors=True)

    # display elapsed time
    elapsed_time = time.time() - start_time
    sct.printv('\nFinished! Elapsed time: ' + str(int(round(elapsed_time))) + 's')

    # to view results
    sct.printv('\nTo view results, type:')
    sct.printv('fslview ' + file_data + ' ' + file_data + suffix + ' &\n')


def usage():
    print '\n' \
        ''+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        '  Convert binary spinal cord segmentation to trilinear-interpolated segmentation. Instead of simply\n' \
        '  re-interpolating the image, this function oversamples the binary mask, smoothes along centerline\n' \
        '  (to remove step-effects), then downsamples back to native resolution.\n' \
        '\n' \
        'USAGE\n' \
        '  '+os.path.basename(__file__)+' -i <bin_seg>\n' \
        '\n' \
        'MANDATORY ARGUMENTS\n' \
        '  -i <bin_seg>      binary segmentation of spinal cord\n' \
        '\n' \
        'OPTIONAL ARGUMENTS\n' \
        '  -s                sigma of the smoothing Gaussian kernel (in voxel). Default='+str(param_default.smoothing_sigma)+'\n' \
        '  -r {0,1}          remove temporary files. Default='+str(param_default.remove_temp_files)+'\n' \
        '  -v {0,1}          verbose. Default='+str(param_default.verbose)+'\n' \
        '  -h                help. Show this message\n' \
        '\n'\
        'EXAMPLE\n' \
        '  '+os.path.basename(__file__)+' -i segmentation.nii \n'

    # exit program
    sys.exit(2)


if __name__ == "__main__":
    # call main function
    main()
