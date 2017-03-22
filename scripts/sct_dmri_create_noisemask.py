#!/usr/bin/env python
#########################################################################################
#
# Identification and estimation of noise in the diffusion signal based on the PIESNO method
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2016 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Charley Gros
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sys
import nibabel as nib
import sct_utils as sct
from msct_parser import Parser
from dipy.denoise.noise_estimate import piesno

# PARSER
# ==========================================================================================
def get_parser():

    """
    :return: Returns the parser with the command line documentation contained in it.
    """
    #param = Param()

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('''Identification and estimation of noise in the diffusion signal, implemented by the Dipy software project (http://nipy.org/dipy/), based on the PIESNO method: Koay C.G., E. Ozarslan, C. Pierpaoli. Probabilistic Identification and Estimation of Noise (PIESNO): A self-consistent approach and its applications in MRI. JMR, 199(1):94-103, 2009.''')
    parser.add_option(name='-i',
                      type_value='file',
                      description='Input file',
                      mandatory=True,
                      example='data_highQ.nii')
    parser.add_option(name='-dof',
                      type_value='int',
                      description='Degree of freedom of the noise distribution. Corresponds to the number of antenna for an acquisition without parallel imaging with sum of squares combination. Otherwise, dof is close to 1.',
                      mandatory=False,
                      default_value='1',
                      example='1')
    parser.add_option(name="-o",
                      type_value='file_output',
                      description='Output file.',
                      mandatory=False,
                      example='noise_mask.nii.gz')
    parser.add_option(name='-h',
                      type_value=None,
                      description='Display this help.',
                      mandatory=False)
    return parser

# MAIN
# ==========================================================================================
def main(fname_in, freedom_degree,file_output):

    img = nib.load(fname_in)
    data = img.get_data()

    sigma, mask = piesno(data, N=freedom_degree, return_mask=True)

    sct.printv('\nWrite NIFTI volumes...')
    output_name = file_output
    nib.save(nib.Nifti1Image(mask, img.get_affine(), img.get_header()), output_name)
    sct.printv('\n.. The noise standard deviation is sigma = ' + str(sigma))

# START PROGRAM
# ==========================================================================================
if __name__ == '__main__':
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])

    fname_in = arguments['-i']
    freedom_degree = int(arguments['-dof'])

    if "-o" in arguments:
        file_output = arguments["-o"]
    else:
        file_output = 'noise_mask.nii.gz'

    main(fname_in, freedom_degree,file_output)
