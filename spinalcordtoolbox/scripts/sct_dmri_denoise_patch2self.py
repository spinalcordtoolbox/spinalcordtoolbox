#!/usr/bin/env python

import os
import sys
from typing import Sequence

import numpy as np
import nibabel as nib
from dipy.denoise.patch2self import patch2self

from spinalcordtoolbox.image import add_suffix
from spinalcordtoolbox.utils import SCTArgumentParser, Metavar, init_sct, printv, set_loglevel, list_type


def get_parser():
    parser = SCTArgumentParser(
        description="Utility function to denoise diffusion MRI images. Returns the denoised image and also the difference "
                    "between the input and the output. The Patch2Self denoising algorithm is based on self-supervised denoising via statistical independence of noise, as described in the following publications:\n"
                    "\n"
                    "- Fadnavis et al. Patch2Self: Denoising Diffusion MRI with Self-supervised Learning. NeurIPS, 2020, Vol. 33. (https://arxiv.org/abs/2011.01355)\n"
                    "- Schilling et al. Patch2Self denoising of diffusion MRI in the cervical spinal cord improves intra-cord contrast, "
                    "signal modelling, repeatability, and feature conspicuity. medRxiv, 2021. (https://doi.org/10.1101/2021.10.04.21264389)\n"
                    "\n"
                    "The implementation is based on DIPY (https://dipy.org/documentation/1.5.0/examples_built/denoise_patch2self/)."
    )

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-i",
        required=True,
        help="Input NIfTI image to be denoised. Example: image_input.nii.gz",
        metavar=Metavar.file,
    )
    mandatory.add_argument(
        "-b",
        required=True,
        help="Input bvals file corresponding to the NIfTI file to be denoised."
             " Example: filename.bval",
        metavar=Metavar.file,
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action='help',
        help="Show this help message and exit.")
    optional.add_argument(
        "-model",
        help="Type of regression model used for self-supervised training within Patch2Self.",
        choices=('ols', 'ridge', 'lasso'),
        default='ols')
    optional.add_argument(
        "-radius",
        help="Patch Radius used to generate p-neighbourhoods within Patch2Self. Notes:\n"
             "- A radius of '0' will use 1x1x1 p-neighbourhoods, a radius of '1' will use "
             "3x3x3 p-neighbourhoods, and so on.\n"
             "- For anisotropic patch sizes, provide a comma-delimited list of 3 integers. "
             "(e.g. '-radius 0,1,0'). For isotropic patch sizes, provide a single int value "
             "(e.g. '-radius 0').",
        metavar=Metavar.int,
        default="0")
    optional.add_argument(
        "-o",
        help="Name of the output NIFTI image.",
        metavar=Metavar.str,
        )
    optional.add_argument(
        "-v",
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode.")

    return parser


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)

    model = arguments.model
    if "," in arguments.radius:
        patch_radius = list_type(",", int)(arguments.radius)
    else:
        patch_radius = int(arguments.radius)

    file_to_denoise = arguments.i
    if arguments.o is not None:
        output_file_name_denoised = arguments.o
        output_file_name_diff = add_suffix(arguments.o, "_difference")
    else:
        _, filename = os.path.split(file_to_denoise)
        output_file_name_denoised = add_suffix(filename, "_patch2self_denoised")
        output_file_name_diff = add_suffix(filename, "_patch2self_difference")

    nii = nib.load(file_to_denoise)
    bvals = np.loadtxt(arguments.b)
    hdr = nii.get_header()
    data = nii.get_data()

    printv("Applying Patch2Self Denoising...")
    data_denoised = patch2self(data, bvals, patch_radius=patch_radius, model=model,
                               verbose=True)

    if verbose == 2:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3)
        axial_middle = int(data.shape[2] / 2)
        middle_vol = int(data.shape[3] / 2)
        before = data[:, :, axial_middle, middle_vol].T
        ax[0].imshow(before, cmap='gray', origin='lower')
        ax[0].set_title("before")
        after = data_denoised[:, :, axial_middle, middle_vol].T
        ax[1].imshow(after, cmap='gray', origin='lower')
        ax[1].set_title("after")
        difference = np.absolute(after.astype('f8') - before.astype('f8'))
        ax[2].imshow(difference, cmap='gray', origin='lower')
        ax[2].set_title("difference")
        for i in range(3):
            ax[i].set_axis_off()
        plt.show()

    # Save files
    nii_denoised = nib.Nifti1Image(data_denoised, None, hdr)
    data_diff = np.absolute(data_denoised.astype('f8') - data.astype('f8'))
    nii_diff = nib.Nifti1Image(data_diff, None, hdr)
    nib.save(nii_denoised, output_file_name_denoised)
    nib.save(nii_diff, output_file_name_diff)

    printv("\nDone! To view results, type:", verbose)
    printv("fsleyes " + file_to_denoise + " " + output_file_name_denoised + " " + output_file_name_diff + " & \n",
           verbose, 'info')


if __name__ == '__main__':
    init_sct()
    main(sys.argv[1:])
