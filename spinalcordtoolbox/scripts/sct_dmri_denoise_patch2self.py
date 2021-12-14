#!/usr/bin/env python

import sys
import numpy as np
import nibabel as nib
from dipy.denoise.patch2self import patch2self

from spinalcordtoolbox.utils import SCTArgumentParser, Metavar, init_sct, printv, extract_fname, set_loglevel, list_type


# DEFAULT PARAMETERS
class Param:
    # The constructor
    def __init__(self):
        self.debug = 0
        self.verbose = 1  # verbose
        self.remove_temp_files = 1
        self.parameter = "ols"
        self.file_to_denoise = ''
        self.bval_file = ''
        self.output_file_name = ''


def get_parser():
    parser = SCTArgumentParser(
        description='Utility function to denoise diffusion MRI images. Return the denoised image and also the difference '
                    'between the input and the output.The Patch2Self denoising algorithm is based on self-supervised denoising via statistical independence of noise'
                    'based on method described in (Shreyas Fadnavis, Joshua Batson, Eleftherios Garyfallidis) Patch2Self: '
                    'Denoising Diffusion MRI with Self-supervised Learning, Advances in Neural Information Processing Systems '
                    '(NeurIPS), 2020, Vol. 33. The implementation is based on DIPY (https://dipy.org/documentation/1.4.1./examples_built/denoise_patch2self/#example-denoise-patch2self)'
    )

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-i",
        default=None,
        required=True,
        help="Input NIfTI image to be denoised. Example: image_input.nii.gz",
        metavar=Metavar.file,
    )
    mandatory.add_argument(
        "-b",
        default=None,
        required=True,
        help="Input bvals file corresponding to the NIfTI file to be denoised."
             " Example: filename.bval",
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
        help='Type of regression model used for self-supervised training within Patch2Self.',
        required=False,
        choices=("ols", "ridge", 'lasso'),
        default="ols")
    optional.add_argument(
        "-radius",
        help="R|Patch Radius used to generate p-neighbourhoods within Patch2Self. Notes:\n"
             "- A radius of '0' will use 1x1x1 p-neighbourhoods, a radius of '1' will use "
             "3x3x3 p-neighbourhoods, and so on.\n"
             "- For ansiotropic patch sizes, provide a comma-delimited list of 3 integers. "
             "(e.g. '-radius 0,1,0'). For isotropic patch sizes, provide a single int value "
             "(e.g. '-radius 0').",
        metavar=Metavar.int,
        required=False,
        default="0")
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
        '-v',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode")

    return parser


def main(argv=None):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)

    model = arguments.p
    remove_temp_files = arguments.r
    if "," in arguments.radius:
        patch_radius = list_type(",", int)(arguments.radius)
    else:
        patch_radius = int(arguments.radius)

    file_to_denoise = arguments.i
    bval_file = arguments.b
    output_file_name = arguments.o

    param = Param()
    param.verbose = verbose
    param.remove_temp_files = remove_temp_files
    param.parameter = model

    path, file, ext = extract_fname(file_to_denoise)

    img = nib.load(file_to_denoise)
    bvals = np.loadtxt(bval_file)
    hdr_0 = img.get_header()
    data = img.get_data()

    printv('Applying Patch2Self Denoising...')
    den = patch2self(data, bvals, patch_radius=patch_radius, model=model,
                     verbose=True)

    if param.verbose == 2:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3)
        axial_middle = int(data.shape[2] / 2)
        middle_vol = int(data.shape[3] / 2)
        before = data[:, :, axial_middle, middle_vol].T
        ax[0].imshow(before, cmap='gray', origin='lower')
        ax[0].set_title('before')
        after = den[:, :, axial_middle, middle_vol].T
        ax[1].imshow(after, cmap='gray', origin='lower')
        ax[1].set_title('after')
        difference = np.absolute(after.astype('f8') - before.astype('f8'))
        ax[2].imshow(difference, cmap='gray', origin='lower')
        ax[2].set_title('difference')
        for i in range(3):
            ax[i].set_axis_off()
        plt.show()

    # Save files
    img_denoise = nib.Nifti1Image(den, None, hdr_0)
    diff_4d = np.absolute(den.astype('f8') - data.astype('f8'))
    img_diff = nib.Nifti1Image(diff_4d, None, hdr_0)
    if output_file_name is not None:
        output_file_name = output_file_name
    else:
        output_file_name = file + '_patch2self_denoised' + ext
    nib.save(img_denoise, output_file_name)
    nib.save(img_diff, file + '_patch2self_difference' + ext)

    printv('\nDone! To view results, type:', param.verbose)
    printv('fsleyes ' + file_to_denoise + ' ' + output_file_name + ' & \n',
           param.verbose, 'info')


# =======================================================================================================================
# Start program
# =======================================================================================================================
if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
