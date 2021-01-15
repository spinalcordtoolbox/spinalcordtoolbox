#!/usr/bin/env python
#########################################################################################
#
# Compute DTI.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os
import sys

from spinalcordtoolbox.utils import SCTArgumentParser, Metavar, init_sct, printv, set_global_loglevel


def get_parser():
    parser = SCTArgumentParser(
        description='Compute Diffusion Tensor Images (DTI) using dipy.'
    )

    mandatory = parser.add_argument_group("MANDATORY ARGMENTS")
    mandatory.add_argument(
        "-i",
        required=True,
        help='Input 4d file. Example: dmri.nii.gz',
        metavar=Metavar.file,
    )
    mandatory.add_argument(
        "-bval",
        required=True,
        help='Bvals file. Example: bvals.txt',
        metavar=Metavar.file,
    )
    mandatory.add_argument(
        "-bvec",
        required=True,
        help='Bvecs file. Example: bvecs.txt',
        metavar=Metavar.file,
    )

    optional = parser.add_argument_group("OPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")
    optional.add_argument(
        '-method',
        help='R|Type of method to calculate the diffusion tensor:\n'
             ' standard: Standard equation [Basser, Biophys J 1994]\n'
             ' restore: Robust fitting with outlier detection [Chang, MRM 2005]',
        default='standard',
        choices=('standard', 'restore'))
    optional.add_argument(
        "-evecs",
        type=int,
        help='Output tensor eigenvectors and eigenvalues.',
        default=0,
        choices=(0, 1))
    optional.add_argument(
        '-m',
        metavar=Metavar.file,
        help='Mask used to compute DTI in for faster processing. Example: mask.nii.gz')
    optional.add_argument(
        '-o',
        help='Output prefix.',
        metavar=Metavar.str,
        required=False,
        default='dti_')
    optional.add_argument(
        '-v',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode")

    return parser


# MAIN
# ==========================================================================================
def main(argv=None):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_global_loglevel(verbose=verbose)

    # initialization
    file_mask = ''

    # Get parser info
    parser = get_parser()
    arguments = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    fname_in = arguments.i
    fname_bvals = arguments.bval
    fname_bvecs = arguments.bvec
    prefix = arguments.o
    method = arguments.method
    evecs = arguments.evecs
    if arguments.m is not None:
        file_mask = arguments.m

    # compute DTI
    if not compute_dti(fname_in, fname_bvals, fname_bvecs, prefix, method, evecs, file_mask, verbose):
        printv('ERROR in compute_dti()', 1, 'error')


# compute_dti
# ==========================================================================================
def compute_dti(fname_in, fname_bvals, fname_bvecs, prefix, method, evecs, file_mask, verbose):
    """
    Compute DTI.
    :param fname_in: input 4d file.
    :param bvals: bvals txt file
    :param bvecs: bvecs txt file
    :param prefix: output prefix. Example: "dti_"
    :param method: algo for computing dti
    :param evecs: bool: output diffusion tensor eigenvectors and eigenvalues
    :return: True/False
    """
    # Open file.
    from spinalcordtoolbox.image import Image
    nii = Image(fname_in)
    data = nii.data
    printv('data.shape (%d, %d, %d, %d)' % data.shape)

    # open bvecs/bvals
    from dipy.io import read_bvals_bvecs
    bvals, bvecs = read_bvals_bvecs(fname_bvals, fname_bvecs)
    from dipy.core.gradients import gradient_table
    gtab = gradient_table(bvals, bvecs)

    # mask and crop the data. This is a quick way to avoid calculating Tensors on the background of the image.
    if not file_mask == '':
        printv('Open mask file...', verbose)
        # open mask file
        nii_mask = Image(file_mask)
        mask = nii_mask.data

    # fit tensor model
    printv('Computing tensor using "' + method + '" method...', verbose)
    import dipy.reconst.dti as dti
    if method == 'standard':
        tenmodel = dti.TensorModel(gtab)
        if file_mask == '':
            tenfit = tenmodel.fit(data)
        else:
            tenfit = tenmodel.fit(data, mask)
    elif method == 'restore':
        import dipy.denoise.noise_estimate as ne
        sigma = ne.estimate_sigma(data)
        dti_restore = dti.TensorModel(gtab, fit_method='RESTORE', sigma=sigma)
        if file_mask == '':
            tenfit = dti_restore.fit(data)
        else:
            tenfit = dti_restore.fit(data, mask)

    # Compute metrics
    printv('Computing metrics...', verbose)
    # FA
    nii.data = tenfit.fa
    nii.save(prefix + 'FA.nii.gz', dtype='float32')
    # MD
    nii.data = tenfit.md
    nii.save(prefix + 'MD.nii.gz', dtype='float32')
    # RD
    nii.data = tenfit.rd
    nii.save(prefix + 'RD.nii.gz', dtype='float32')
    # AD
    nii.data = tenfit.ad
    nii.save(prefix + 'AD.nii.gz', dtype='float32')
    if evecs:
        data_evecs = tenfit.evecs
        data_evals = tenfit.evals
        # output 1st (V1), 2nd (V2) and 3rd (V3) eigenvectors as 4d data
        for idim in range(3):
            nii.data = data_evecs[:, :, :, :, idim]
            nii.save(prefix + 'V' + str(idim + 1) + '.nii.gz', dtype="float32")
            nii.data = data_evals[:, :, :, idim]
            nii.save(prefix + 'E' + str(idim + 1) + '.nii.gz', dtype="float32")

    return True


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])

