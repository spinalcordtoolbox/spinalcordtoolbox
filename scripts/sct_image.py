#!/usr/bin/env python
##############################################################################
#
# Perform operations on images
#
# ----------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad, Sara Dupont
#
# About the license: see the file LICENSE.TXT
##############################################################################

from __future__ import absolute_import

import os
import sys
import argparse
import numpy as np

import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.image import Image, concat_data
from spinalcordtoolbox.utils import Metavar, SmartFormatter

import sct_utils as sct


def get_parser():

    parser = argparse.ArgumentParser(
        description='Perform manipulations on images (e.g., pad, change space, split along dimension). '
                    'Inputs can be a number, a 4d image, or several 3d images separated with ","',
        formatter_class=SmartFormatter,
        add_help=None,
        prog=os.path.basename(__file__).strip('.py'))
    mandatory = parser.add_argument_group('MANDATORY ARGUMENTS')
    mandatory.add_argument(
        '-i',
        nargs='+',
        metavar=Metavar.file,
        help='Input file(s). If several inputs: separate them by white space. Example: "data.nii.gz"',
        required = True)
    optional = parser.add_argument_group('OPTIONAL ARGUMENTS')
    optional.add_argument(
        '-h',
        '--help',
        action='help',
        help='Show this help message and exit')
    optional.add_argument(
        '-o',
        metavar=Metavar.file,
        help='Output file. Example: data_pad.nii.gz',
        required = False)

    image = parser.add_argument_group('IMAGE OPERATIONS')
    image.add_argument(
        '-pad',
        metavar=Metavar.list,
        help='Pad 3D image. Specify padding as: "x,y,z" (in voxel). Example: "0,0,1"',
        required = False)
    image.add_argument(
        '-pad-asym',
        metavar=Metavar.list,
        help='Pad 3D image with asymmetric padding. Specify padding as: "x_i,x_f,y_i,y_f,z_i,z_f" (in voxel). '
             'Example: "0,0,5,10,1,1"',
        required = False)
    image.add_argument(
        '-split',
        help='Split data along the specified dimension. The suffix _DIM+NUMBER will be added to the intput file name.',
        required = False,
        choices = ('x', 'y', 'z', 't'))
    image.add_argument(
        '-concat',
        help='Concatenate data along the specified dimension',
        required = False,
        choices = ('x', 'y', 'z', 't'))
    image.add_argument(
        '-remove-vol',
        metavar=Metavar.list,
        help='Remove specific volumes from a 4d volume. Separate with ",". Example: "0,5,10"',
        required=False)
    image.add_argument(
        '-keep-vol',
        metavar=Metavar.list,
        help='Keep specific volumes from a 4d volume (remove others). Separate with ",". Example: "1,2,3,11"',
        required=False)
    image.add_argument(
        '-type',
        help='Change file type',
        required = False,
        choices = ('uint8','int16','int32','float32','complex64','float64','int8','uint16','uint32','int64','uint64'))

    header = parser.add_argument_group('HEADER OPERATIONS')
    header.add_argument(
        '-copy-header',
        metavar=Metavar.file,
        help='Copy the header of the source image (specified in -i) to the destination image (specified here). '
             'Example: data_dest.nii.gz',
        required = False)

    orientation = parser.add_argument_group('ORIENTATION OPERATIONS')
    orientation.add_argument(
        '-getorient',
        help='Get orientation of the input image',
        action='store_true',
        required=False)
    orientation.add_argument(
        '-setorient',
        help='Set orientation of the input image (only modifies the header).',
        choices='RIP LIP RSP LSP RIA LIA RSA LSA IRP ILP SRP SLP IRA ILA SRA SLA RPI LPI RAI LAI RPS LPS RAS LAS PRI PLI ARI ALI PRS PLS ARS ALS IPR SPR IAR SAR IPL SPL IAL SAL PIR PSR AIR ASR PIL PSL AIL ASL'.split(),
        required = False)
    orientation.add_argument(
        '-setorient-data',
        help='Set orientation of the input image\'s data (does NOT modify the header, but the data). Use with care !',
        choices='RIP LIP RSP LSP RIA LIA RSA LSA IRP ILP SRP SLP IRA ILA SRA SLA RPI LPI RAI LAI RPS LPS RAS LAS PRI PLI ARI ALI PRS PLS ARS ALS IPR SPR IAR SAR IPL SPL IAL SAL PIR PSR AIR ASR PIL PSL AIL ASL'.split(),
        required = False)

    multi = parser.add_argument_group('MULTI-COMPONENT OPERATIONS ON ITK COMPOSITE WARPING FIELDS')
    multi.add_argument(
        '-mcs',
        action='store_true',
        help='Multi-component split: Split ITK warping field into three separate displacement fields. '
             'The suffix _X, _Y and _Z will be added to the input file name.',
        required=False)
    multi.add_argument(
        '-omc',
        action='store_true',
        help='Multi-component merge: Merge inputted images into one multi-component image. Requires several inputs.',
        required=False)

    warping = parser.add_argument_group('WARPING FIELD OPERATIONSWarping field operations:')
    warping.add_argument(
        '-display-warp',
        action='store_true',
        help='Create a grid and deform it using provided warping field.',
        required=False)

    misc = parser.add_argument_group('Misc')
    misc.add_argument(
        '-v',
        type=int,
        help='Verbose. 0: nothing. 1: basic. 2: extended.',
        required=False,
        default=1,
        choices=(0, 1, 2))

    return parser


def main(args=None):
    """
    Main function
    :param args:
    :return:
    """
    # initializations
    output_type = None
    dim_list = ['x', 'y', 'z', 't']

    # Get parser args
    if args is None:
        args = None if sys.argv[1:] else ['--help']
    parser = get_parser()
    arguments = parser.parse_args(args=args)
    fname_in = arguments.i
    n_in = len(fname_in)
    verbose = arguments.v
    sct.init_sct(log_level=verbose, update=True)  # Update log level

    if arguments.o is not None:
        fname_out = arguments.o
    else:
        fname_out = None

    # run command
    if arguments.concat is not None:
        dim = arguments.concat
        assert dim in dim_list
        dim = dim_list.index(dim)
        im_out = [concat_data(fname_in, dim)]  # TODO: adapt to fname_in

    elif arguments.copy_header is not None:
        im_in = Image(fname_in[0])
        im_dest = Image(arguments.copy_header)
        im_dest_new = im_in.copy()
        im_dest_new.data = im_dest.data.copy()
        # im_dest.header = im_in.header
        im_dest_new.absolutepath = im_dest.absolutepath
        im_out = [im_dest_new]
        fname_out = arguments.copy_header

    elif arguments.display_warp:
        im_in = fname_in[0]
        visualize_warp(im_in, fname_grid=None, step=3, rm_tmp=True)
        im_out = None

    elif arguments.getorient:
        im_in = Image(fname_in[0])
        orient = im_in.orientation
        im_out = None

    elif arguments.keep_vol is not None:
        index_vol = (arguments.keep_vol).split(',')
        for iindex_vol, vol in enumerate(index_vol):
                index_vol[iindex_vol] = int(vol)
        im_in = Image(fname_in[0])
        im_out = [remove_vol(im_in, index_vol, todo='keep')]

    elif arguments.mcs:
        im_in = Image(fname_in[0])
        if n_in != 1:
            sct.printv(parser.usage.generate(error='ERROR: -mcs need only one input'))
        if len(im_in.data.shape) != 5:
            sct.printv(parser.usage.generate(error='ERROR: -mcs input need to be a multi-component image'))
        im_out = multicomponent_split(im_in)

    elif arguments.omc:
        im_ref = Image(fname_in[0])
        for fname in fname_in:
            im = Image(fname)
            if im.data.shape != im_ref.data.shape:
                sct.printv(parser.usage.generate(error='ERROR: -omc inputs need to have all the same shapes'))
            del im
        im_out = [multicomponent_merge(fname_in)]  # TODO: adapt to fname_in

    elif arguments.pad is not None:
        im_in = Image(fname_in[0])
        ndims = len(im_in.data.shape)
        if ndims != 3:
            sct.printv('ERROR: you need to specify a 3D input file.', 1, 'error')
            return

        pad_arguments = arguments.pad.split(',')
        if len(pad_arguments) != 3:
            sct.printv('ERROR: you need to specify 3 padding values.', 1, 'error')

        padx, pady, padz = pad_arguments
        padx, pady, padz = int(padx), int(pady), int(padz)
        im_out = [pad_image(im_in, pad_x_i=padx, pad_x_f=padx, pad_y_i=pady,
                            pad_y_f=pady, pad_z_i=padz, pad_z_f=padz)]

    elif arguments.pad_asym is not None:
        im_in = Image(fname_in[0])
        ndims = len(im_in.data.shape)
        if ndims != 3:
            sct.printv('ERROR: you need to specify a 3D input file.', 1, 'error')
            return

        pad_arguments = arguments.pad_asym.split(',')
        if len(pad_arguments) != 6:
            sct.printv('ERROR: you need to specify 6 padding values.', 1, 'error')

        padxi, padxf, padyi, padyf, padzi, padzf = pad_arguments
        padxi, padxf, padyi, padyf, padzi, padzf = int(padxi), int(padxf), int(padyi), int(padyf), int(padzi), int(padzf)
        im_out = [pad_image(im_in, pad_x_i=padxi, pad_x_f=padxf, pad_y_i=padyi, pad_y_f=padyf, pad_z_i=padzi, pad_z_f=padzf)]

    elif arguments.remove_vol is not None:
        index_vol = (arguments.remove_vol).split(',')
        for iindex_vol, vol in enumerate(index_vol):
            index_vol[iindex_vol] = int(vol)
        im_in = Image(fname_in[0])
        im_out = [remove_vol(im_in, index_vol, todo='remove')]

    elif arguments.setorient is not None:
        sct.printv(fname_in[0])
        im_in = Image(fname_in[0])
        im_out = [msct_image.change_orientation(im_in, arguments.setorient)]

    elif arguments.setorient_data is not None:
        im_in = Image(fname_in[0])
        im_out = [msct_image.change_orientation(im_in, arguments.setorient_data, data_only=True)]

    elif arguments.split is not None:
        dim = arguments.split
        assert dim in dim_list
        im_in = Image(fname_in[0])
        dim = dim_list.index(dim)
        im_out = split_data(im_in, dim)

    elif arguments.type is not None:
        output_type = arguments.type
        im_in = Image(fname_in[0])
        im_out = [im_in]  # TODO: adapt to fname_in

    else:
        im_out = None
        sct.printv(parser.usage.generate(error='ERROR: you need to specify an operation to do on the input image'))

    # in case fname_out is not defined, use first element of input file name list
    if fname_out is None:
        fname_out = fname_in[0]

    # Write output
    if im_out is not None:
        sct.printv('Generate output files...', verbose)
        # if only one output
        if len(im_out) == 1 and not '-split' in arguments:
            im_out[0].save(fname_out, dtype=output_type, verbose=verbose)
            sct.display_viewer_syntax([fname_out], verbose=verbose)
        if arguments.mcs:
            # use input file name and add _X, _Y _Z. Keep the same extension
            l_fname_out = []
            for i_dim in range(3):
                l_fname_out.append(sct.add_suffix(fname_out or fname_in[0], '_' + dim_list[i_dim].upper()))
                im_out[i_dim].save(l_fname_out[i_dim], verbose=verbose)
            sct.display_viewer_syntax(fname_out)
        if arguments.split is not None:
            # use input file name and add _"DIM+NUMBER". Keep the same extension
            l_fname_out = []
            for i, im in enumerate(im_out):
                l_fname_out.append(sct.add_suffix(fname_out or fname_in[0], '_' + dim_list[dim].upper() + str(i).zfill(4)))
                im.save(l_fname_out[i])
            sct.display_viewer_syntax(l_fname_out)

    elif arguments.getorient:
        sct.printv(orient)

    elif arguments.display_warp:
        sct.printv('Warping grid generated.', verbose, 'info')


def pad_image(im, pad_x_i=0, pad_x_f=0, pad_y_i=0, pad_y_f=0, pad_z_i=0, pad_z_f=0):

    nx, ny, nz, nt, px, py, pz, pt = im.dim
    pad_x_i, pad_x_f, pad_y_i, pad_y_f, pad_z_i, pad_z_f = int(pad_x_i), int(pad_x_f), int(pad_y_i), int(pad_y_f), int(pad_z_i), int(pad_z_f)

    if len(im.data.shape) == 2:
        new_shape = list(im.data.shape)
        new_shape.append(1)
        im.data = im.data.reshape(new_shape)

    # initialize padded_data, with same type as im.data
    padded_data = np.zeros((nx + pad_x_i + pad_x_f, ny + pad_y_i + pad_y_f, nz + pad_z_i + pad_z_f), dtype=im.data.dtype)

    if pad_x_f == 0:
        pad_x_f = None
    elif pad_x_f > 0:
        pad_x_f *= -1
    if pad_y_f == 0:
        pad_y_f = None
    elif pad_y_f > 0:
        pad_y_f *= -1
    if pad_z_f == 0:
        pad_z_f = None
    elif pad_z_f > 0:
        pad_z_f *= -1

    padded_data[pad_x_i:pad_x_f, pad_y_i:pad_y_f, pad_z_i:pad_z_f] = im.data
    im_out = im.copy()
    # TODO: Do not copy the Image(), because the dim field and hdr.get_data_shape() will not be updated properly.
    #   better to just create a new Image() from scratch.
    im_out.data = padded_data  # done after the call of the function
    im_out.absolutepath = sct.add_suffix(im_out.absolutepath, "_pad")

    # adapt the origin in the sform and qform matrix
    new_origin = np.dot(im_out.hdr.get_qform(), [-pad_x_i, -pad_y_i, -pad_z_i, 1])

    im_out.hdr.structarr['qoffset_x'] = new_origin[0]
    im_out.hdr.structarr['qoffset_y'] = new_origin[1]
    im_out.hdr.structarr['qoffset_z'] = new_origin[2]
    im_out.hdr.structarr['srow_x'][-1] = new_origin[0]
    im_out.hdr.structarr['srow_y'][-1] = new_origin[1]
    im_out.hdr.structarr['srow_z'][-1] = new_origin[2]

    return im_out


def split_data(im_in, dim, squeeze_data=True):
    """
    Split data
    :param im_in: input image.
    :param dim: dimension: 0, 1, 2, 3.
    :return: list of split images
    """

    dim_list = ['x', 'y', 'z', 't']
    # Parse file name
    # Open first file.
    data = im_in.data
    # in case input volume is 3d and dim=t, create new axis
    if dim + 1 > len(np.shape(data)):
        data = data[..., np.newaxis]
    # in case splitting along the last dim, make sure to remove the last dim to avoid singleton
    if dim + 1 == len(np.shape(data)):
        if squeeze_data:
            do_reshape = True
        else:
            do_reshape = False
    else:
        do_reshape = False
    # Split data into list
    data_split = np.array_split(data, data.shape[dim], dim)
    # Write each file
    im_out_list = []
    for idx_img, dat in enumerate(data_split):
        im_out = msct_image.empty_like(im_in)
        if do_reshape:
            im_out.data = dat.reshape(tuple([ x for (idx_shape, x) in enumerate(data.shape) if idx_shape != dim]))
        else:
            im_out.data = dat
        im_out.absolutepath = sct.add_suffix(im_in.absolutepath, "_{}{}".format(dim_list[dim].upper(), str(idx_img).zfill(4)))
        im_out_list.append(im_out)

    return im_out_list


def remove_vol(im_in, index_vol_user, todo):
    """
    Remove specific volumes from 4D data.
    :param im_in: [str] input image.
    :param index_vol: [int] list of indices corresponding to volumes to remove
    :param todo: {keep, remove} what to do
    :return: 4d volume
    """
    # get data
    data = im_in.data
    nt = data.shape[3]
    # define index list of volumes to keep/remove
    if todo == 'remove':
        index_vol = [i for i in range(0, nt) if i not in index_vol_user]
    elif todo == 'keep':
        index_vol = index_vol_user
    else:
        sct.printv('ERROR: wrong assignment of variable "todo"', 1, 'error')
    # define new 4d matrix with selected volumes
    data_out = data[:, :, :, index_vol]
    # save matrix inside new Image object
    im_out = im_in.copy()
    im_out.data = data_out
    return im_out


def concat_warp2d(fname_list, fname_warp3d, fname_dest):
    """
    Concatenate 2d warping fields into a 3d warping field along z dimension. The 3rd dimension of the resulting warping
    field will be zeroed.
    :param
    fname_list: list of 2d warping fields (along X and Y).
    fname_warp3d: output name of 3d warping field
    fname_dest: 3d destination file (used to copy header information)
    :return: none
    """
    from numpy import zeros
    import nibabel as nib

    # get dimensions
    # nib.load(fname_list[0])
    # im_0 = Image(fname_list[0])
    nx, ny = nib.load(fname_list[0]).shape[0:2]
    nz = len(fname_list)
    # warp3d = tuple([nx, ny, nz, 1, 3])
    warp3d = zeros([nx, ny, nz, 1, 3])
    for iz, fname in enumerate(fname_list):
        warp2d = nib.load(fname).get_data()
        warp3d[:, :, iz, 0, 0] = warp2d[:, :, 0, 0, 0]
        warp3d[:, :, iz, 0, 1] = warp2d[:, :, 0, 0, 1]
        del warp2d
    # save new image
    im_dest = nib.load(fname_dest)
    affine_dest = im_dest.get_affine()
    im_warp3d = nib.Nifti1Image(warp3d, affine_dest)
    # set "intent" code to vector, to be interpreted as warping field
    im_warp3d.header.set_intent('vector', (), '')
    nib.save(im_warp3d, fname_warp3d)
    # copy header from 2d warping field
    #
    # im_dest = Image(fname_dest)
    # im_warp3d = im_dest.copy()
    # im_warp3d.data = warp3d.astype('float32')
    # # add dimension between 3rd and 5th
    # im_warp3d.hdr.set_data_shape([nx, ny, nz, 1, 3])
    #
    # im_warp3d.hdr.set_intent('vector', (), '')
    # im_warp3d.absolutepath = fname_warp3d
    # # save 3d warping field
    # im_warp3d.save()
    # return im_out


def multicomponent_split(im):
    """
    Convert composite image (e.g., ITK warping field, 5dim) into several 3d volumes.
    Replaces "c3d -mcs warp_comp.nii -oo warp_vecx.nii warp_vecy.nii warp_vecz.nii"
    :param im:
    :return:
    """
    data = im.data
    assert len(data.shape) == 5
    data_out = []
    for i in range(data.shape[-1]):
        dat_out = data[:, :, :, :, i]
        '''
        while dat_out.shape[-1] == 1:
            dat_out = reshape(dat_out, dat_out.shape[:-1])
        '''
        data_out.append(dat_out)  # .astype('float32'))
    im_out = [im.copy() for j in range(len(data_out))]
    for i, im in enumerate(im_out):
        im.data = data_out[i]
        im.hdr.set_intent('vector', (), '')
        im.absolutepath = sct.add_suffix(im.absolutepath, '_{}'.format(i))
    return im_out


def multicomponent_merge(fname_list):
    from numpy import zeros
    # WARNING: output multicomponent is not optimal yet, some issues may be related to the use of this function

    im_0 = Image(fname_list[0])
    new_shape = list(im_0.data.shape)
    if len(new_shape) == 3:
        new_shape.append(1)
    new_shape.append(len(fname_list))
    new_shape = tuple(new_shape)

    data_out = zeros(new_shape)
    for i, fname in enumerate(fname_list):
        im = Image(fname)
        dat = im.data
        if len(dat.shape) == 2:
            data_out[:, :, 0, 0, i] = dat.astype('float32')
        elif len(dat.shape) == 3:
            data_out[:, :, :, 0, i] = dat.astype('float32')
        elif len(dat.shape) == 4:
            data_out[:, :, :, :, i] = dat.astype('float32')
        del im
        del dat
    im_out = im_0.copy()
    im_out.data = data_out.astype('float32')
    im_out.hdr.set_intent('vector', (), '')
    im_out.absolutepath = sct.add_suffix(im_out.absolutepath, '_multicomponent')
    return im_out


def visualize_warp(fname_warp, fname_grid=None, step=3, rm_tmp=True):
    if fname_grid is None:
        from numpy import zeros
        tmp_dir = sct.tmp_create()
        im_warp = Image(fname_warp)
        status, out = sct.run(['fslhd', fname_warp])
        curdir = os.getcwd()
        os.chdir(tmp_dir)
        dim1 = 'dim1           '
        dim2 = 'dim2           '
        dim3 = 'dim3           '
        nx = int(out[out.find(dim1):][len(dim1):out[out.find(dim1):].find('\n')])
        ny = int(out[out.find(dim2):][len(dim2):out[out.find(dim2):].find('\n')])
        nz = int(out[out.find(dim3):][len(dim3):out[out.find(dim3):].find('\n')])
        sq = zeros((step, step))
        sq[step - 1] = 1
        sq[:, step - 1] = 1
        dat = zeros((nx, ny, nz))
        for i in range(0, dat.shape[0], step):
            for j in range(0, dat.shape[1], step):
                for k in range(dat.shape[2]):
                    if dat[i:i + step, j:j + step, k].shape == (step, step):
                        dat[i:i + step, j:j + step, k] = sq
        fname_grid = 'grid_' + str(step) + '.nii.gz'
        im_grid = Image(param=dat)
        grid_hdr = im_warp.hdr
        im_grid.hdr = grid_hdr
        im_grid.absolutepath = fname_grid
        im_grid.save()
        fname_grid_resample = sct.add_suffix(fname_grid, '_resample')
        sct.run(['sct_resample', '-i', fname_grid, '-f', '3x3x1', '-x', 'nn', '-o', fname_grid_resample])
        fname_grid = os.path.join(tmp_dir, fname_grid_resample)
        os.chdir(curdir)
    path_warp, file_warp, ext_warp = sct.extract_fname(fname_warp)
    grid_warped = os.path.join(path_warp, sct.extract_fname(fname_grid)[1] + '_' + file_warp + ext_warp)
    sct.run(['sct_apply_transfo', '-i', fname_grid, '-d', fname_grid, '-w', fname_warp, '-o', grid_warped])
    if rm_tmp:
        sct.rmtree(tmp_dir)


if __name__ == "__main__":
    sct.init_sct()
    main()
