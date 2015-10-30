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
from sct_utils import extract_fname, printv


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
    parser.usage.set_description('Threshold image using Otsu algorithm.')
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

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    fname_in = arguments['-i']
    fname_bvals = arguments['-bval']
    fname_bvecs = arguments['-bvec']
    prefix = arguments['-o']
    param.verbose = int(arguments['-v'])

    # compute DTI
    if not compute_dti(fname_in, fname_bvals, fname_bvecs, prefix):
        printv('ERROR in compute_dti()', 1, 'error')


# compute_dti
# ==========================================================================================
def compute_dti(fname_in, fname_bvals, fname_bvecs, prefix):
    """
    Compute DTI.
    :param fname_in: input 4d file.
    :param bvals: bvals txt file
    :param bvecs: bvecs txt file
    :param prefix: output prefix. Example: "dti_"
    :return: True/False
    """
    # Open file.
    from msct_image import Image
    nii = Image(fname_in)
    data = nii.data
    print('data.shape (%d, %d, %d, %d)' % data.shape)

    # open bvecs/bvals
    from dipy.io import read_bvals_bvecs
    bvals, bvecs = read_bvals_bvecs(fname_bvals, fname_bvecs)
    from dipy.core.gradients import gradient_table
    gtab = gradient_table(bvals, bvecs)

    # # mask and crop the data. This is a quick way to avoid calculating Tensors on the background of the image.
    # from dipy.segment.mask import median_otsu
    # maskdata, mask = median_otsu(data, 3, 1, True, vol_idx=range(10, 50), dilate=2)
    # print('maskdata.shape (%d, %d, %d, %d)' % maskdata.shape)

    # fit tensor model
    import dipy.reconst.dti as dti
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(data)

    # Compute metrics
    printv('Computing metrics...', param.verbose)
    # FA
    from dipy.reconst.dti import fractional_anisotropy
    nii.data = fractional_anisotropy(tenfit.evals)
    nii.setFileName(prefix+'FA.nii.gz')
    nii.save('float32')
    # MD
    from dipy.reconst.dti import mean_diffusivity
    nii.data = mean_diffusivity(tenfit.evals)
    nii.setFileName(prefix+'MD.nii.gz')
    nii.save('float32')
    # RD
    from dipy.reconst.dti import radial_diffusivity
    nii.data = radial_diffusivity(tenfit.evals)
    nii.setFileName(prefix+'RD.nii.gz')
    nii.save('float32')
    # AD
    from dipy.reconst.dti import axial_diffusivity
    nii.data = axial_diffusivity(tenfit.evals)
    nii.setFileName(prefix+'AD.nii.gz')
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
#         printv('ERROR: bvecs should be text file with 3 lines (or columns).', 1, 'error')
#     return bvecs
#


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    # call main function
    main()
