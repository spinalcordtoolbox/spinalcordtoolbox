#!/usr/bin/env python
#########################################################################################
#
# Resample data.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad, Sara Dupont
# Modified: 2015-09-08
#
# About the license: see the file LICENSE.TXT
#########################################################################################


#TODO: pad for c3d!!!!!!


import sys
import os
import getopt
import time
import sct_utils as sct
from sct_convert import convert
from msct_image import Image
from msct_parser import Parser
import dipy.align.aniso2iso as dp_iso


# DEFAULT PARAMETERS
class Param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.fname_data = ''
        self.fname_out = ''
        self.factor = ''
        self.interpolation = 'trilinear'
        self.x_to_order = {'nn': 0, 'trilinear': 1, 'spline': 3}  # TODO: change attribution to orders (0 and 1 are OK but see for the others)
        self.mode = 'nearest'  # How to fill the points outside the boundaries of the input, possible options: constant, nearest, reflect or wrap
        self.file_suffix = '_resampled'  # output suffix
        self.verbose = 1
        self.remove_tmp_files = 1


# resample
# ======================================================================================================================
def resample():
    # extract resampling factor
    sct.printv('\nParse resampling factor...', param.verbose)
    factor_split = param.factor.split('x')
    factor = [float(factor_split[i]) for i in range(len(factor_split))]
    # check if it has three values
    if not len(factor) == 3:
        sct.printv('\nERROR: factor should have three dimensions. E.g., 2x2x1.\n', 1, 'error')
    else:
        fx, fy, fz = [float(factor_split[i]) for i in range(len(factor_split))]

    # Extract path/file/extension
    path_data, file_data, ext_data = sct.extract_fname(param.fname_data)
    path_out, file_out, ext_out = path_data, file_data, ext_data
    if param.fname_out != '':
        file_out = sct.extract_fname(param.fname_out)[1]
    else:
        file_out.append(param.file_suffix)

    input_im = Image(param.fname_data)

    # Get dimensions of data
    sct.printv('\nGet dimensions of data...', param.verbose)
    nx, ny, nz, nt, px, py, pz, pt = input_im.dim
    sct.printv('  ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz)+ ' x ' + str(nt), param.verbose)
    dim = 4  # by default, will be adjusted later
    if nt == 1:
        dim = 3
    if nz == 1:
        dim = 2
        #TODO : adapt for 2D too or change description
        sct.run('ERROR (sct_resample): Dimension of input data is different from 3 or 4. Exit program', param.verbose, 'error')

    # Calculate new dimensions
    sct.printv('\nCalculate new dimensions...', param.verbose)
    nx_new = int(round(nx*fx))
    ny_new = int(round(ny*fy))
    nz_new = int(round(nz*fz))
    px_new = px/fx
    py_new = py/fy
    pz_new = pz/fz
    sct.printv('  ' + str(nx_new) + ' x ' + str(ny_new) + ' x ' + str(nz_new)+ ' x ' + str(nt), param.verbose)


    zooms = input_im.hdr.get_zooms()[:3]
    affine = input_im.hdr.get_base_affine()
    new_zooms = (px_new, py_new, pz_new)

    if type(param.interpolation) == int:
        order = param.interpolation
    elif type(param.interpolation) == str and param.interpolation in param.x_to_order.keys():
        order = param.x_to_order[param.interpolation]
    else:
        order = 1
        sct.printv('WARNING: wrong input for the interpolation. Using default value = trilinear', param.verbose, 'warning')

    new_data, new_affine = dp_iso.reslice(input_im.data, affine, zooms, new_zooms, mode=param.mode, order=order)

    new_im = Image(param=new_data)
    new_im.absolutepath = path_out+file_out+ext_out
    new_im.path = path_out
    new_im.file_name = file_out
    new_im.ext = ext_out

    zooms_to_set = list(new_zooms)
    if dim == 4:
        zooms_to_set.append(nt)

    new_im.hdr = input_im.hdr
    new_im.hdr.set_zooms(zooms_to_set)
    new_im.save()

    # to view results
    sct.printv('\nDone! To view results, type:', param.verbose)
    sct.printv('fslview '+param.fname_out+' &', param.verbose, 'info')
    print


# ======================================================================================================================
# Start program
# ======================================================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    param_debug = Param()

    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        # get path of the testing data
        path_sct_data = os.environ.get("SCT_TESTING_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__))), "testing_data")
        param.fname_data = path_sct_data+'/fmri/fmri.nii.gz'
        param.factor = '2' #'0.5x0.5x1'
        param.remove_tmp_files = 0
        param.verbose = 1
    else:
        # Initialize the parser
        parser = Parser(__file__)
        parser.usage.set_description('Anisotropic resampling of 3D or 4D data.')
        parser.add_option(name="-i",
                          type_value="file",
                          description="Image to segment. Can be 2D, 3D or 4D.",
                          mandatory=True,
                          example='dwi.nii.gz')
        parser.add_option(name="-f",
                          type_value="str",
                          description="Resampling factor in each of the first 3 dimensions (x,y,z). Separate with \"x\"\n"
                                      "For 2x upsampling, set to 2. For 2x downsampling set to 0.5",
                          mandatory=True,
                          example='0.5x0.5x1')
        parser.add_option(name="-o",
                          type_value="file_output",
                          description="Output file name",
                          mandatory=False,
                          example='dwi_resampled.nii.gz')
        parser.add_option(name="-x",
                          type_value='multiple_choice',
                          description="Interpolation. nn (nearest neighbor), trilinear, or spline (order 3)\n"
                                      "You can also choose the order of the spline using an integer from 0 to 5",
                          mandatory=False,
                          default_value='trilinear',
                          example=['nn', 'trilinear', 'spline', '0', '1', '2', '3', '4', '5'])
        parser.add_option(name="-r",
                          type_value='multiple_choice',
                          description="Output file name",
                          mandatory=False,
                          default_value=1,
                          example=['0', '1'])
        parser.add_option(name="-v",
                          type_value='multiple_choice',
                          description="verbose: 0 = nothing, 1 = classic, 2 = expended",
                          mandatory=False,
                          default_value=1,
                          example=['0', '1', '2'])

        arguments = parser.parse(sys.argv[1:])
        param.fname_data = arguments["-i"]
        param.factor = arguments["-f"]

        if "-o" in arguments:
            param.fname_out = arguments["-o"]
        if "-x" in arguments:
            if len(arguments["-x"]) == 1:
                param.interpolation = int(arguments["-x"])
            else:
                param.interpolation = arguments["-x"]
        if "-r" in arguments:
            param.remove_tmp_files = int(arguments["-r"])
        if "-v" in arguments:
            param.verbose = int(arguments["-v"])

    # call main function
    resample()
