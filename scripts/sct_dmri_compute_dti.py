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

from __future__ import absolute_import

import sys

from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti

from msct_parser import Parser
import sct_utils as sct

class Param:
    def __init__(self):
        self.verbose = 1


# PARSER
# ==========================================================================================
def get_parser():
    param = Param()

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Compute Diffusion Tensor Images (DTI) using dipy.')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Input 4d file.",
                      mandatory=True,
                      example="dmri.nii.gz")
    parser.add_option(name="-bval",
                      type_value="file",
                      description="Bvals file.",
                      mandatory=True,
                      example="bvals.txt")
    parser.add_option(name="-bvec",
                      type_value="file",
                      description="Bvecs file.",
                      mandatory=True,
                      example="bvecs.txt")
    parser.add_option(name='-method',
                      type_value='multiple_choice',
                      description='Type of method to calculate the diffusion tensor:\nstandard: Standard equation [Basser, Biophys J 1994]\nrestore: Robust fitting with outlier detection [Chang, MRM 2005]',
                      mandatory=False,
                      default_value='standard',
                      example=['standard', 'restore'])
    parser.add_option(name="-evecs",
                      type_value="multiple_choice",
                      description="""To output tensor eigenvectors and eigenvalues, set to 1.""",
                      mandatory=False,
                      default_value='0',
                      example=['0', '1'])
    parser.add_option(name='-m',
                      type_value='file',
                      description='Mask used to compute DTI in for faster processing.',
                      mandatory=False,
                      example='mask.nii.gz')
    parser.add_option(name='-o',
                      type_value='str',
                      description='Output prefix.',
                      mandatory=False,
                      default_value='dti_')
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="""Verbose. 0: nothing. 1: basic. 2: extended.""",
                      mandatory=False,
                      default_value=str(param.verbose),
                      example=['0', '1', '2'])
    return parser


# MAIN
# ==========================================================================================
def main(args = None):

    if not args:
        args = sys.argv[1:]

    # initialization
    file_mask = ''

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    fname_in = arguments['-i']
    fname_bvals = arguments['-bval']
    fname_bvecs = arguments['-bvec']
    prefix = arguments['-o']
    method = arguments['-method']
    evecs = int(arguments['-evecs'])
    if "-m" in arguments:
        file_mask = arguments['-m']
    param.verbose = int(arguments.get('-v'))
    sct.init_sct(log_level=param.verbose, update=True)  # Update log level

    # compute DTI
    if not compute_dti(fname_in, fname_bvals, fname_bvecs, prefix, method, evecs, file_mask):
        sct.printv('ERROR in compute_dti()', 1, 'error')


# compute_dti
# ==========================================================================================
def compute_dti(fname_in, fname_bvals, fname_bvecs, prefix, method, evecs, file_mask):
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
    sct.printv('data.shape (%d, %d, %d, %d)' % data.shape)

    # open bvecs/bvals
    bvals, bvecs = read_bvals_bvecs(fname_bvals, fname_bvecs)
    gtab = gradient_table(bvals, bvecs)

    # mask and crop the data. This is a quick way to avoid calculating Tensors on the background of the image.
    if not file_mask == '':
        sct.printv('Open mask file...', param.verbose)
        # open mask file
        nii_mask = Image(file_mask)
        mask = nii_mask.data

    # fit tensor model
    sct.printv('Computing tensor using "' + method + '" method...', param.verbose)
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
    sct.printv('Computing metrics...', param.verbose)
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
            nii.save(prefix + 'V' + str(idim+1) + '.nii.gz', dtype="float32")
            nii.data = data_evals[:, :, :, idim]
            nii.save(prefix + 'E' + str(idim+1) + '.nii.gz', dtype="float32")

    return True


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.init_sct()
    # initialize parameters
    param = Param()
    # call main function
    main()
