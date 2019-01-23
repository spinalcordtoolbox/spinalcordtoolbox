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

# TODO: Ultimately replace resample_image (based on nipy) with Image, to avoid confusion.


from __future__ import division, absolute_import

import nipy
import numpy as np
from nipy.algorithms.registration.resample import resample as n_resample

import sct_utils as sct


def resample_nipy(img, new_size, new_size_type, interpolation='linear', verbose=1):
    """Resample a nipy image object based on a specified resampling factor.
    Can deal with 2d, 3d or 4d image objects.  TODO: make test for that
    :param img: nipy Image.
    :param factor: list of float: Resampling factor. E.g., for 2x isotropic upsampling of a 3d image: factor=[2, 2, 2]
    TODO: implement as list.
    :param interpolation: {'nn', 'linear', 'spline'}. The interpolation type
    :return: The resampled nipy Image.
    """
    # TODO: deal with 4d (and other dim) data

    # Get dimensions of data
    p = img.header.get_zooms()
    shape = img.header.get_data_shape()

    # parse input argument
    new_size = new_size.split('x')

    if img.ndim == 4:
        new_size += ['1']  # needed because the code below is general, i.e., does not assume 3d input and uses img.shape

    # assert len(shape) == len(new_size)

    # compute new shape based on specific resampling method
    if new_size_type == 'vox':
        shape_r = tuple([int(new_size[i]) for i in range(img.ndim)])
    elif new_size_type == 'factor':
        if len(new_size) == 1:
            # isotropic resampling
            new_size = tuple([new_size[0] for i in range(img.ndim)])
        # compute new shape as: shape_r = shape * f
        shape_r = tuple([int(np.round(shape[i] * float(new_size[i]))) for i in range(img.ndim)])
    elif new_size_type == 'mm':
        if len(new_size) == 1:
            # isotropic resampling
            new_size = tuple([new_size[0] for i in range(img.ndim)])
        # compute new shape as: shape_r = shape * (p_r / p)
        shape_r = tuple([int(np.round(shape[i] * float(p[i]) / float(new_size[i]))) for i in range(img.ndim)])
    else:
        sct.log.error('new_size_type is not recognized.')

    # Generate 3d affine transformation: R
    affine = img.affine[:4, :4]
    affine[3, :] = np.array([0, 0, 0, 1])  # satisfy to nifti convention. Otherwise it grabs the temporal
    sct.log.debug('Affine matrix: \n' + str(affine), verbose)
    R = np.eye(4)
    for i in range(3):
        R[i, i] = img.shape[i] / float(shape_r[i])
    affine_r = np.dot(affine, R)

    # set interpolation method
    # TODO: make a dict
    if interpolation == 'nn':
        interp_order = 0
    elif interpolation == 'linear':
        interp_order = 1
    elif interpolation == 'spline':
        interp_order = 2

    if img.ndim == 3:
        img_r = n_resample(img, transform=R, reference=(shape_r, affine_r), mov_voxel_coords=True,
                           ref_voxel_coords=True, dtype='double', interp_order=interp_order, mode='nearest')
    elif img.ndim == 4:
        # Import here instead of top of the file because this is an isolated case and nibabel takes time to import
        import nibabel as nib
        from nipy.io.nifti_ref import nifti2nipy, nipy2nifti
        data4d = np.zeros(shape_r)
        # Loop across 4th dimension and resample each 3d volume
        for it in range(img.shape[3]):
            # Create dummy 3d nipy image
            nii_tmp = nib.nifti1.Nifti1Image(img.get_data()[..., it], affine)
            img3d_r = n_resample(nifti2nipy(nii_tmp), transform=R, reference=(shape_r[:-1], affine_r),
                                 mov_voxel_coords=True, ref_voxel_coords=True, dtype='double',
                                 interp_order=interp_order, mode='nearest')
            data4d[..., it] = img3d_r.get_data()
        # Create 4d nipy Image
        nii4d = nib.nifti1.Nifti1Image(data4d, affine_r)
        # Convert to nipy object
        img_r = nifti2nipy(nii4d)

    return img_r


def resample_file(fname_data, fname_out, new_size, new_size_type, interpolation, verbose):
    """This function will resample the specified input
    image file to the target size.
    Can deal with 2d, 3d or 4d image objects.
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

    nii_r = resample_nipy(nii, new_size, new_size_type,
                          interpolation, verbose)

    # build output file name
    if fname_out == '':
        fname_out = sct.add_suffix(fname_data, '_r')
    else:
        fname_out = fname_out

    # save data
    nipy.save_image(nii_r, fname_out)

    # to view results
    sct.display_viewer_syntax([fname_out], verbose=verbose)

    return nii_r

