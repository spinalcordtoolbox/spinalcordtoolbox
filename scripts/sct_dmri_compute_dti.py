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

import sys

from msct_parser import Parser
import sct_utils as sct


class Param:
    def __init__(self):
        self.verbose = 1


# PARSER
# ==========================================================================================
def get_parser():
    param = Param()

    # parser initialisation
    parser = Parser(__file__)

    # # initialize parameters
    # param = Param()
    # param_default = Param()

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
    if "-m" in arguments:
        file_mask = arguments['-m']
    param.verbose = int(arguments['-v'])

    # compute DTI
    if not compute_dti(fname_in, fname_bvals, fname_bvecs, prefix, method, file_mask):
        sct.printv('ERROR in compute_dti()', 1, 'error')


# compute_dti
# ==========================================================================================
def compute_dti(fname_in, fname_bvals, fname_bvecs, prefix, method, file_mask):
    """
    Compute DTI.
    :param fname_in: input 4d file.
    :param bvals: bvals txt file
    :param bvecs: bvecs txt file
    :param prefix: output prefix. Example: "dti_"
    :param method: algo for computing dti
    :return: True/False
    """
    # Open file.
    from msct_image import Image
    nii = Image(fname_in)
    data = nii.data
    sct.printv('data.shape (%d, %d, %d, %d)' % data.shape)

    # open bvecs/bvals
    from dipy.io import read_bvals_bvecs
    bvals, bvecs = read_bvals_bvecs(fname_bvals, fname_bvecs)
    from dipy.core.gradients import gradient_table
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
    from dipy.reconst.dti import fractional_anisotropy
    nii.data = fractional_anisotropy(tenfit.evals)
    nii.setFileName(prefix + 'FA.nii.gz')
    nii.save('float32')
    # MD
    from dipy.reconst.dti import mean_diffusivity
    nii.data = mean_diffusivity(tenfit.evals)
    nii.setFileName(prefix + 'MD.nii.gz')
    nii.save('float32')
    # RD
    from dipy.reconst.dti import radial_diffusivity
    nii.data = radial_diffusivity(tenfit.evals)
    nii.setFileName(prefix + 'RD.nii.gz')
    nii.save('float32')
    # AD
    from dipy.reconst.dti import axial_diffusivity
    nii.data = axial_diffusivity(tenfit.evals)
    nii.setFileName(prefix + 'AD.nii.gz')
    nii.save('float32')

    return True


# # Get bvecs
# # ==========================================================================================
# def get_bvecs(fname):
#     """
#     Read bvecs file and output array
#     :param fname: bvecs file
#     :return: (nx3) array
#     """
#     text_file = open(fname, 'r')
#     list_bvecs = text_file.readlines()
#     text_file.close()
#     # parse txt file and transform to array
#     from numpy import array
#     bvecs = array([[float(j.strip("\n")) for j in list_bvecs[i].split(" ")] for i in range(len(list_bvecs))])
#     # make sure one dimension is "3"
#     if not 3 in bvecs.shape:
#         sct.printv('ERROR: bvecs should be text file with 3 lines (or columns).', 1, 'error')
#     return bvecs
#


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.start_stream_logger()
    # initialize parameters
    param = Param()
    # call main function
    main()
