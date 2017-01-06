#!/usr/bin/env python
###############################################################################
#
# Resample data.
#
# -----------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad, Sara Dupont
# Modified: 2015-09-08
#
# About the license: see the file LICENSE.TXT
###############################################################################
# TODO: adjust qform of output data to account for pixel size
# TODO: test if crashes with 2d or 4d data
# TODO: raise exception if input size is not numerical

import os
import sys

import sct_utils as sct
from msct_parser import msct_parser.Parser


class Param(object):
    def __init__(self):
        self.debug = 0
        self.fname_data = ''
        self.fname_out = ''
        self.new_size = ''
        self.new_size_type = ''
        self.interpolation = 'linear'
        self.x_to_order = {'nn': 0, 'linear': 1, 'spline': 2}
        # How to fill the points outside the boundaries of the input, possible options: constant, nearest, reflect or wrap
        self.mode = 'reflect'
        # constant put the superior edges to 0, wrap does something weird with the superior edges, nearest and reflect are fine
        self.file_suffix = '_resampled'  # output suffix
        self.verbose = 1


def resample(param):
    """
    Resample data using nipy. Note: we cannot use msct_image because coordmap needs to be used.
    :return:
    """
    import nipy
    from nipy.algorithms.registration import resample
    import numpy as np

    verbose = param.verbose
    # Load data
    sct.printv('\nLoad data...', verbose)
    nii = nipy.load_image(param.fname_data)
    data = nii.get_data()
    # Get dimensions of data
    p = nii.header.get_zooms()
    n = nii.header.get_data_shape()
    sct.printv('  pixdim: ' + str(p), verbose)
    sct.printv('  shape: ' + str(n), verbose)

    # Calculate new dimensions
    sct.printv('\nCalculate new dimensions...', verbose)
    # parse input argument
    new_size = param.new_size.split('x')
    # if 4d, add resampling factor to 4th dimension
    if len(p) == 4:
        new_size.append('1')
    # compute new shape based on specific resampling method
    if param.new_size_type == 'vox':
        n_r = tuple([int(new_size[i]) for i in range(len(n))])
    elif param.new_size_type == 'factor':
        if len(new_size) == 1:
            # isotropic resampling
            new_size = tuple([new_size[0] for i in range(len(n))])
        # compute new shape as: n_r = n * f
        n_r = tuple(
            [int(round(n[i] * float(new_size[i]))) for i in range(len(n))])
    elif param.new_size_type == 'mm':
        if len(new_size) == 1:
            # isotropic resampling
            new_size = tuple([new_size[0] for i in range(len(n))])
        # compute new shape as: n_r = n * (p_r / p)
        n_r = tuple([
            int(round(n[i] * float(p[i]) / float(new_size[i])))
            for i in range(len(n))
        ])
    else:
        sct.printv('\nERROR: param.new_size_type is not recognized.', 1,
                   'error')
    sct.printv('  new shape: ' + str(n_r), verbose)

    affine = nii.coordmap.affine
    sct.printv('  affine matrix: \n' + str(affine))

    # create ref image
    arr_r = np.zeros(n_r)
    R = np.eye(len(n) + 1)
    for i in range(len(n)):
        R[i, i] = n[i] / float(n_r[i])
    affine_r = np.dot(affine, R)
    coordmap_r = nii.coordmap
    coordmap_r.affine = affine_r
    nii_r = nipy.core.api.Image(arr_r, coordmap_r)

    sct.printv('\nCalculate affine transformation...', verbose)
    # create affine transformation
    transfo = R
    # if data are 4d, delete temporal dimension
    if len(p) == 4:
        transfo = np.delete(transfo, 3, 0)
        transfo = np.delete(transfo, 3, 1)
    # translate to account for voxel size (otherwise resulting image will be shifted by half a voxel). Modify the three first rows of the last column, corresponding to the translation.
    transfo[:3, -1] = np.array(
        ((R[0, 0] - 1) / 2, (R[1, 1] - 1) / 2, (R[2, 2] - 1) / 2), dtype='f8')
    # print transfo
    sct.printv('  transfo: \n' + str(transfo), verbose)

    # set interpolation method
    if param.interpolation == 'nn':
        interp_order = 0
    elif param.interpolation == 'linear':
        interp_order = 1
    elif param.interpolation == 'spline':
        interp_order = 2

    # create 3d coordmap because resample only accepts 3d data (jcohenadad 2016-07-26)
    if len(n) == 4:
        from copy import deepcopy
        coordmap3d = deepcopy(nii.coordmap)
        from nipy.core.reference.coordinate_system import CoordinateSystem
        coordmap3d.__setattr__('function_domain', CoordinateSystem('xyz'))
        # create 3d affine transfo
        affine3d = np.delete(affine, 3, 0)
        affine3d = np.delete(affine3d, 3, 1)
        coordmap3d.affine = affine3d

    # resample data
    if len(n) == 3:
        data_r = resample(
            nii,
            transform=transfo,
            reference=nii_r,
            mov_voxel_coords=True,
            ref_voxel_coords=True,
            dtype='double',
            interp_order=interp_order,
            mode='nearest')
    elif len(n) == 4:
        data_r = np.zeros(n_r)
        for it in range(n[3]):
            # create 3d nipy-like data
            arr3d = data[:, :, :, it]
            nii3d = nipy.core.api.Image(arr3d, coordmap3d)
            arr_r3d = arr_r[:, :, :, it]
            nii_r3d = nipy.core.api.Image(arr_r3d, coordmap3d)
            # resample data
            data3d_r = resample(
                nii3d,
                transform=transfo,
                reference=nii_r3d,
                mov_voxel_coords=True,
                ref_voxel_coords=True,
                dtype='double',
                interp_order=interp_order,
                mode='nearest')
            # data_r = np.concatenate((data_r, data3d_r), axis=3)
            data_r[:, :, :, it] = data3d_r.get_data()

    # build output file name
    if param.fname_out == '':
        fname_out = sct.add_suffix(param.fname_data, '_r')
    else:
        fname_out = param.fname_out

    # save data
    nii_r = nipy.core.api.Image(data_r, coordmap_r)
    nipy.save_image(nii_r, fname_out)

    # to view results
    sct.printv('\nDone! To view results, type:', verbose)
    sct.printv('fslview ' + fname_out + ' &', verbose, 'info')


def get_parser():
    parser = msct_parser.Parser(__file__)
    parser.usage.set_description('Anisotropic resampling of 3D or 4D data.')
    parser.add_option(
        name="-i",
        type_value="file",
        description="Image to segment. Can be 3D or 4D. (Cannot be 2D)",
        mandatory=True,
        example='dwi.nii.gz')
    parser.usage.addSection(
        'TYPE OF THE NEW SIZE INPUT : with a factor of resampling, in mm or in number of voxels\n'
        'Please choose only one of the 3 options.')
    parser.add_option(
        name="-f",
        type_value="str",
        description="Resampling factor in each dimensions (x,y,z). Separate with \"x\"\n"
        "For 2x upsampling, set to 2. For 2x downsampling set to 0.5",
        mandatory=False,
        example='0.5x0.5x1')
    parser.add_option(
        name="-mm",
        type_value="str",
        description="New resolution in mm. Separate dimension with \"x\"",
        mandatory=False,
        example='0.1x0.1x5')
    parser.add_option(
        name="-vox",
        type_value="str",
        description="Resampling size in number of voxels in each dimensions (x,y,z). Separate with \"x\"",
        mandatory=False)
    # example='50x50x20')
    parser.usage.addSection('MISC')
    parser.add_option(
        name="-x",
        type_value='multiple_choice',
        description='Interpolation method.',
        mandatory=False,
        default_value='linear',
        example=['nn', 'linear', 'spline'])

    parser.add_option(
        name="-o",
        type_value="file_output",
        description="Output file name",
        mandatory=False,
        example='dwi_resampled.nii.gz')
    parser.add_option(
        name="-v",
        type_value='multiple_choice',
        description="verbose: 0 = nothing, 1 = classic, 2 = expended.",
        mandatory=False,
        default_value=1,
        example=['0', '1', '2'])
    return parser


def main(args=None):

    if args is None:
        args = sys.argv[1:]

    param = Param()
    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        # get path of the testing data
        path_sct_data = os.environ.get('SCT_TESTING_DATA_DIR')
        param.fname_data = path_sct_data + '/fmri/fmri.nii.gz'
        param.new_size = '2'  # '0.5x0.5x1'
        param.remove_tmp_files = 0
        param.verbose = 1
    else:
        parser = get_parser()
        arguments = parser.parse(args)
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
            sct.printv(
                parser.usage.generate(
                    error='ERROR: you need to specify one of those three arguments : -f, -mm or -vox'
                ))

        if arg > 1:
            sct.printv(
                parser.usage.generate(
                    error='ERROR: you need to specify ONLY one of those three arguments : -f, -mm or -vox'
                ))

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
    resample(param)


if __name__ == "__main__":
    main()
