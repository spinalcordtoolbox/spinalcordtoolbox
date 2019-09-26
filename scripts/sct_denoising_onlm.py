#!/usr/bin/env python

from __future__ import absolute_import, division

import sys, io, os, argparse

import numpy as np
from time import time
import nibabel as nib

import sct_utils as sct
from spinalcordtoolbox.utils import Metavar, SmartFormatter


# DEFAULT PARAMETERS
class Param:
    # The constructor
    def __init__(self):
        self.debug = 0
        self.verbose = 1  # verbose
        self.remove_temp_files = 1
        self.parameter = "Rician"
        self.file_to_denoise = ''
        self.output_file_name = ''

def get_parser():
    # Initialize the parser

    parser = argparse.ArgumentParser(
        description='Utility function to denoise images. (Return the denoised image and also the difference '
                    'between the input and the output.)',
        add_help=None,
        formatter_class=SmartFormatter,
        prog=os.path.basename(__file__).strip(".py"))

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-i",
        default=None,
        required=True,
        help="Input NIFTI image to be denoised. Example: image_input.nii.gz",
        metavar=Metavar.file,
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")
    optional.add_argument(
        "-p",
        help='Type of assumed noise distribution. Default is: Rician.',
        required=False,
        choices=("Rician", "Gaussian"),
        default="Rician")
    optional.add_argument(
        "-d",
        type=int,
        help="Threshold value for what to be considered as noise. "
             "The standard deviation of the noise is calculated for values below this limit. "
             "Not relevant if -std value is precised. Default is 80.",
        metavar=Metavar.int,
        required=False,
        default="80")
    optional.add_argument(
        "-std",
        type=float,
        help="Standard deviation of the noise. "
             "If not specified, it is calculated using a background of point of values "
             "below the threshold value (parameter d).",
        metavar=Metavar.float)
    optional.add_argument(
        "-o",
        help="Name of the output NIFTI image.",
        metavar=Metavar.str,
        default=None)
    optional.add_argument(
        "-r",
        help="Remove temporary files. Specify 0 to get access to temporary files.",
        type=int,
        choices=(0, 1),
        default=1)
    optional.add_argument(
        "-v",
        help="Verbose. 0: nothing. 1: basic. 2: extended.",
        type=int,
        default=1,
        choices=(0, 1, 2))

    return parser


def main(file_to_denoise, param, output_file_name) :

    path, file, ext = sct.extract_fname(file_to_denoise)

    img = nib.load(file_to_denoise)
    hdr_0 = img.get_header()

    data = img.get_data()
    aff = img.get_affine()

    if min(data.shape) <= 5:
        sct.printv('One of the image dimensions is <= 5 : reducing the size of the block radius.')
        block_radius = min(data.shape) - 1
    else:
        block_radius = 5  # default value

    # Process for manual detecting of background
    # mask = data[:, :, :] > noise_threshold
    # data = data[:, :, :]

    from dipy.denoise.nlmeans import nlmeans

    if arguments.std is not None:
        sigma = std_noise
        # Application of NLM filter to the image
        sct.printv('Applying Non-local mean filter...')
        if param.parameter == 'Rician':
            den = nlmeans(data, sigma=sigma, mask=None, rician=True, block_radius=block_radius)
        else :
            den = nlmeans(data, sigma=sigma, mask=None, rician=False, block_radius=block_radius)
    else:
        # # Process for manual detecting of background
        mask = data > noise_threshold
        sigma = np.std(data[~mask])
        # Application of NLM filter to the image
        sct.printv('Applying Non-local mean filter...')
        if param.parameter == 'Rician':
            den = nlmeans(data, sigma=sigma, mask=mask, rician=True, block_radius=block_radius)
        else:
            den = nlmeans(data, sigma=sigma, mask=mask, rician=False, block_radius=block_radius)

    t = time()
    sct.printv("total time: %s" % (time() - t))
    sct.printv("vol size", den.shape)

    axial_middle = int(data.shape[2] / 2)

    before = data[:, :, axial_middle].T
    after = den[:, :, axial_middle].T

    diff_3d = np.absolute(den.astype('f8') - data.astype('f8'))
    difference = np.absolute(after.astype('f8') - before.astype('f8'))
    if arguments.std is None:
        difference[~mask[:, :, axial_middle].T] = 0

    if param.verbose == 2:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(before, cmap='gray', origin='lower')
        ax[0].set_title('before')
        ax[1].imshow(after, cmap='gray', origin='lower')
        ax[1].set_title('after')
        ax[2].imshow(difference, cmap='gray', origin='lower')
        ax[2].set_title('difference')
        for i in range(3):
            ax[i].set_axis_off()

        plt.show()

    # Save files
    img_denoise = nib.Nifti1Image(den, None, hdr_0)
    img_diff = nib.Nifti1Image(diff_3d, None, hdr_0)
    if output_file_name is not None:
        output_file_name = output_file_name
    else:
        output_file_name = file + '_denoised' + ext
    nib.save(img_denoise, output_file_name)
    nib.save(img_diff, file + '_difference' + ext)

    sct.printv('\nDone! To view results, type:', param.verbose)
    sct.printv('fsleyes ' + file_to_denoise + ' ' + output_file_name + ' & \n', param.verbose, 'info')


# =======================================================================================================================
# Start program
# =======================================================================================================================
if __name__ == "__main__":
    sct.init_sct()
    parser = get_parser()
    # initialize parameters
    arguments = parser.parse_args(args = None if sys.argv[1:] else ['--help'])

    parameter = arguments.p
    remove_temp_files = arguments.r
    noise_threshold = arguments.d
    verbose = arguments.v
    sct.init_sct(log_level=verbose, update=True)  # Update log level

    file_to_denoise = arguments.i
    output_file_name = arguments.o
    std_noise = arguments.std

    param = Param()
    param.verbose = verbose
    param.remove_temp_files = remove_temp_files
    param.parameter = parameter

    main(file_to_denoise, param, output_file_name)
