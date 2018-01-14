#########################################################################################
#
# Resample data using nipy.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad, Sara Dupont
#
# About the license: see the file LICENSE.TXT
#########################################################################################
import copy

import nipy
import numpy as np
from nipy.algorithms.registration.resample import resample as n_resample

import sct_utils as sct


def resample_image(input_image, new_size, new_size_type,
                   interpolation='linear', verbose=1):
    """This function will resample the specified input
    image to the target size.

    :param input_image: The input image.
    :param new_size: The target size, i.e. '0.25x0.25'
    :param new_size_type: Unit of resample (mm, vox, factor)
    :param interpolation: The interpolation type
    :param verbose: Verbosity level
    :return: The resampled image.
    """

    data = input_image.get_data()
    # Get dimensions of data
    p = input_image.header.get_zooms()
    n = input_image.header.get_data_shape()
    sct.printv('  pixdim: ' + str(p), verbose)
    sct.printv('  shape: ' + str(n), verbose)

    # Calculate new dimensions
    sct.printv('\nCalculate new dimensions...', verbose)
    # parse input argument
    new_size = new_size.split('x')
    # if 4d, add resampling factor to 4th dimension
    if len(p) == 4:
        new_size.append('1')
    # compute new shape based on specific resampling method
    if new_size_type == 'vox':
        n_r = tuple([int(new_size[i]) for i in range(len(n))])
    elif new_size_type == 'factor':
        if len(new_size) == 1:
            # isotropic resampling
            new_size = tuple([new_size[0] for i in range(len(n))])
        # compute new shape as: n_r = n * f
        n_r = tuple([int(round(n[i] * float(new_size[i]))) for i in range(len(n))])
    elif new_size_type == 'mm':
        if len(new_size) == 1:
            # isotropic resampling
            new_size = tuple([new_size[0] for i in range(len(n))])
        # compute new shape as: n_r = n * (p_r / p)
        n_r = tuple([int(round(n[i] * float(p[i]) / float(new_size[i]))) for i in range(len(n))])
    else:
        sct.printv('\nERROR: new_size_type is not recognized.', 1, 'error')
    sct.printv('  new shape: ' + str(n_r), verbose)

    affine = input_image.coordmap.affine
    sct.printv('  affine matrix: \n' + str(affine))

    # create ref image
    arr_r = np.zeros(n_r)
    R = np.eye(len(n) + 1)
    for i in range(len(n)):
        R[i, i] = n[i] / float(n_r[i])
    affine_r = np.dot(affine, R)
    coordmap_r = input_image.coordmap
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
    transfo[:3, -1] = np.array(((R[0, 0] - 1) / 2, (R[1, 1] - 1) / 2, (R[2, 2] - 1) / 2), dtype='f8')
    sct.printv('  transfo: \n' + str(transfo), verbose)

    # set interpolation method
    if interpolation == 'nn':
        interp_order = 0
    elif interpolation == 'linear':
        interp_order = 1
    elif interpolation == 'spline':
        interp_order = 2

    # create 3d coordmap because resample only accepts 3d data (jcohenadad 2016-07-26)
    if len(n) == 4:
        coordmap3d = copy.deepcopy(input_image.coordmap)
        from nipy.core.reference.coordinate_system import CoordinateSystem
        coordmap3d.function_domain = CoordinateSystem('xyz')
        # create 3d affine transfo
        affine3d = np.delete(affine, 3, 0)
        affine3d = np.delete(affine3d, 3, 1)
        coordmap3d.affine = affine3d

    # resample data
    if len(n) == 3:
        data_r = n_resample(input_image, transform=transfo, reference=nii_r, mov_voxel_coords=True,
                          ref_voxel_coords=True, dtype='double', interp_order=interp_order,
                          mode='nearest')
    elif len(n) == 4:
        data_r = np.zeros(n_r)

        # loop across 4th dimension
        for it in range(n[3]):
            # create 3d nipy-like data
            arr3d = data[:, :, :, it]
            nii3d = nipy.core.api.Image(arr3d, coordmap3d)
            arr_r3d = arr_r[:, :, :, it]
            nii_r3d = nipy.core.api.Image(arr_r3d, coordmap3d)
            # resample data
            data3d_r = n_resample(nii3d, transform=transfo, reference=nii_r3d,
                                mov_voxel_coords=True, ref_voxel_coords=True,
                                dtype='double', interp_order=interp_order,
                                mode='nearest')
            data_r[:, :, :, it] = data3d_r.get_data()


    nii_r = nipy.core.api.Image(data_r, coordmap_r)

    return nii_r


def resample_file(fname_data, fname_out, new_size, new_size_type,
                  interpolation, verbose):
    """This function will resample the specified input
    image file to the target size.

    :param fname_data: The input image filename.
    :param fname_out: The output image filename.
    :param new_size: The target size, i.e. 0.25x0.25
    :param new_size_type: Unit of resample (mm, vox, factor)
    :param interpolation: The interpolation type
    :param verbose: verbosity level
    """

    # Load data
    sct.printv('\nLoad data...', verbose)
    nii = nipy.load_image(fname_data)

    nii_r = resample_image(nii, new_size, new_size_type, 
                           interpolation, verbose)

    # build output file name
    if fname_out == '':
        fname_out = sct.add_suffix(fname_data, '_r')
    else:
        fname_out = fname_out

    # save data
    nipy.save_image(nii_r, fname_out)

    # to view results
    sct.printv('\nDone! To view results, type:', verbose)
    sct.printv('fslview ' + fname_out + ' &', verbose, 'info')

    return nii_r

