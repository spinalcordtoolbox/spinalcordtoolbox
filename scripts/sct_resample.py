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

import sys
# import os
# import getopt
import commands
import sct_utils as sct
# import time
# from sct_convert import convert
from msct_image import Image
from msct_parser import Parser
import dipy.align.reslice as dp_iso


# DEFAULT PARAMETERS
class Param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.fname_data = ''
        self.fname_out = ''
        self.new_size = ''
        self.new_size_type = ''
        self.interpolation = 'linear'
        self.x_to_order = {'nn': 0, 'linear': 1, 'spline': 2}
        self.mode = 'reflect'  # How to fill the points outside the boundaries of the input, possible options: constant, nearest, reflect or wrap
        # constant put the superior edges to 0, wrap does something weird with the superior edges, nearest and reflect are fine
        self.file_suffix = '_resampled'  # output suffix
        self.verbose = 1


# resample
# ======================================================================================================================
def resample():
    # extract resampling factor
    sct.printv('\nParse resampling factor...', param.verbose)
    new_size_split = param.new_size.split('x')
    new_size = [float(new_size_split[i]) for i in range(len(new_size_split))]
    # check if it has three values
    if not len(new_size) == 3:
        sct.printv('\nERROR: new size should have three dimensions. E.g., 2x2x1.\n', 1, 'error')
    else:
        ns_x, ns_y, ns_z = new_size

    # Extract path/file/extension
    path_data, file_data, ext_data = sct.extract_fname(param.fname_data)
    path_out, file_out, ext_out = '', file_data, ext_data
    if param.fname_out != '':
        path_out, file_out, ext_out = sct.extract_fname(param.fname_out)
    else:
        file_out += param.file_suffix
    param.fname_out = path_out+file_out+ext_out

    input_im = Image(param.fname_data)

    # Get dimensions of data
    sct.printv('\nGet dimensions of data...', param.verbose)
    nx, ny, nz, nt, px, py, pz, pt = input_im.dim
    sct.printv('  ' + str(px) + ' x ' + str(py) + ' x ' + str(pz)+ ' x ' + str(pt)+'mm', param.verbose)
    dim = 4  # by default, will be adjusted later
    if nt == 1:
        dim = 3
    if nz == 1:
        dim = 2
        sct.run('ERROR (sct_resample): Dimension of input data is different from 3 or 4. Exit program', param.verbose, 'error')

    # Calculate new dimensions
    sct.printv('\nCalculate new dimensions...', param.verbose)
    if param.new_size_type == 'factor':
        px_new = px/ns_x
        py_new = py/ns_y
        pz_new = pz/ns_z
    elif param.new_size_type == 'vox':
        px_new = px*nx/ns_x
        py_new = py*ny/ns_y
        pz_new = pz*nz/ns_z
    else:
        px_new = ns_x
        py_new = ns_y
        pz_new = ns_z

    sct.printv('  ' + str(px_new) + ' x ' + str(py_new) + ' x ' + str(pz_new)+ ' x ' + str(pt)+'mm', param.verbose)

    zooms = (px, py, pz)  # input_im.hdr.get_zooms()[:3]
    affine = input_im.hdr.get_qform()  # get_base_affine()
    new_zooms = (px_new, py_new, pz_new)

    if type(param.interpolation) == int:
        order = param.interpolation
    elif type(param.interpolation) == str and param.interpolation in param.x_to_order.keys():
        order = param.x_to_order[param.interpolation]
    else:
        order = 1
        sct.printv('WARNING: wrong input for the interpolation. Using default value = linear', param.verbose, 'warning')

    new_data, new_affine = dp_iso.reslice(input_im.data, affine, zooms, new_zooms, mode=param.mode, order=order)

    new_im = Image(param=new_data)
    new_im.absolutepath = param.fname_out
    new_im.path = path_out
    new_im.file_name = file_out
    new_im.ext = ext_out

    zooms_to_set = list(new_zooms)
    if dim == 4:
        zooms_to_set.append(nt)

    new_im.hdr = input_im.hdr
    new_im.hdr.set_zooms(zooms_to_set)

    # Set the new sform and qform:
    new_im.hdr.set_sform(new_affine)
    new_im.hdr.set_qform(new_affine)

    new_im.save()

    # to view results
    sct.printv('\nDone! To view results, type:', param.verbose)
    sct.printv('fslview '+param.fname_out+' &', param.verbose, 'info')


def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Anisotropic resampling of 3D or 4D data.')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Image to segment. Can be 3D or 4D. (Cannot be 2D)",
                      mandatory=True,
                      example='dwi.nii.gz')
    parser.usage.addSection('TYPE OF THE NEW SIZE INPUT : with a factor of resampling, in mm or in number of voxels\n'
                            'Please choose only one of the 3 options.')
    parser.add_option(name="-f",
                      type_value="str",
                      description="Resampling factor in each dimensions (x,y,z). Separate with \"x\"\n"
                                  "For 2x upsampling, set to 2. For 2x downsampling set to 0.5",
                      mandatory=False,
                      example='0.5x0.5x1')
    parser.add_option(name="-mm",
                      type_value="str",
                      description="Resampling size in mm in each dimensions (x,y,z). Separate with \"x\"",
                      mandatory=False)
                      # example='0.1x0.1x5')
    parser.add_option(name="-vox",
                      type_value="str",
                      description="Resampling size in number of voxels in each dimensions (x,y,z). Separate with \"x\"",
                      mandatory=False)
                      # example='50x50x20')
    parser.usage.addSection('MISC')
    parser.add_option(name="-x",
                      type_value='multiple_choice',
                      description="Interpolation. nn (nearest neighbor : spline of order 0), linear (spline of order 1), or spline (cubic spline: order 2).\n"
                                  "You can also choose the order of the spline using an integer from 3 to 5.",
                      mandatory=False,
                      default_value='linear',
                      example=['nn', 'linear', 'spline', '3', '4', '5'])

    parser.add_option(name="-o",
                      type_value="file_output",
                      description="Output file name",
                      mandatory=False,
                      example='dwi_resampled.nii.gz')
    parser.add_option(name="-v",
                      type_value='multiple_choice',
                      description="verbose: 0 = nothing, 1 = classic, 2 = expended.",
                      mandatory=False,
                      default_value=1,
                      example=['0', '1', '2'])
    return parser

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
        status, path_sct_data = commands.getstatusoutput('echo $SCT_TESTING_DATA_DIR')
        param.fname_data = path_sct_data+'/fmri/fmri.nii.gz'
        param.new_size = '2' #'0.5x0.5x1'
        param.remove_tmp_files = 0
        param.verbose = 1
    else:
        parser = get_parser()
        arguments = parser.parse(sys.argv[1:])
        param.fname_data = arguments["-i"]
        arg = 0
        if "-f" in arguments:
            param.new_size = arguments["-f"]
            param.new_size_type = 'factor'
            arg += 1
        elif "-mm" in arguments:
            param.new_size = arguments["-mm"]
            param.new_size_type = 'mm'
            arg += 1
        elif "-vox" in arguments:
            param.new_size = arguments["-vox"]
            param.new_size_type = 'vox'
            arg += 1
        else:
            sct.printv(parser.usage.generate(error='ERROR: you need to specify one of those three arguments : -f, -mm or -vox'))

        if arg > 1:
            sct.printv(parser.usage.generate(error='ERROR: you need to specify ONLY one of those three arguments : -f, -mm or -vox'))

        if "-o" in arguments:
            param.fname_out = arguments["-o"]
        if "-x" in arguments:
            if len(arguments["-x"]) == 1:
                param.interpolation = int(arguments["-x"])
            else:
                param.interpolation = arguments["-x"]
        if "-v" in arguments:
            param.verbose = int(arguments["-v"])

    # call main function
    resample()
