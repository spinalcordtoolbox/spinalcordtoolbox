#!/usr/bin/env python
#########################################################################################
#
# Perform operations on images
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad, Sara Dupont
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os
import sys

from msct_image import Image, get_dimension
from msct_parser import Parser
from numpy import newaxis, shape
from sct_utils import add_suffix, extract_fname, printv, run, tmp_create
import sct_utils as sct

class Param:
    def __init__(self):
        self.verbose = '1'

# PARSER
# ==========================================================================================


def get_parser():
    # initialize default param
    param_default = Param()
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Perform manipulations on images (e.g., pad, change space, split along dimension). Inputs can be a number, a 4d image, or several 3d images separated with ","')
    parser.add_option(name="-i",
                      type_value=[[','], 'file'],
                      description="Input file(s). If several inputs: separate them by a coma without white space.\n",
                      mandatory=True,
                      example="data.nii.gz")
    parser.add_option(name="-o",
                      type_value='file_output',
                      description='Output file.',
                      mandatory=False,
                      example='data_pad.nii.gz')

    parser.usage.addSection('\nBasic image operations:')
    parser.add_option(name="-pad",
                      type_value="str",
                      description='Pad 3D image. Specify padding as: "x,y,z" (in voxel)',
                      mandatory=False,
                      example='0,0,1')
    parser.add_option(name="-pad-asym",
                      type_value="str",
                      description='Pad 3D image with asymmetric padding. Specify padding as: "x_i,x_f,y_i,y_f,z_i,z_f" (in voxel)',
                      mandatory=False,
                      example='0,0,5,10,1,1')
    parser.add_option(name="-copy-header",
                      type_value="file",
                      description='Copy the header of the input image (specified in -i) to the destination image (specified here)',
                      mandatory=False,
                      example='data_dest.nii.gz')
    parser.add_option(name="-split",
                      type_value="multiple_choice",
                      description='Split data along the specified dimension. The suffix _DIM+NUMBER will be added to the intput file name.',
                      mandatory=False,
                      example=['x', 'y', 'z', 't'])
    parser.add_option(name="-concat",
                      type_value="multiple_choice",
                      description='Concatenate data along the specified dimension',
                      mandatory=False,
                      example=['x', 'y', 'z', 't'])
    parser.add_option(name='-remove-vol',
                      type_value=[[','], 'int'],
                      description='Remove specific volumes from a 4d volume. Separate with ","',
                      mandatory=False,
                      example='0,5,10')
    parser.add_option(name='-keep-vol',
                      type_value=[[','], 'int'],
                      description='Keep specific volumes from a 4d volume (remove others). Separate with ","',
                      mandatory=False,
                      example='1,2,3,11')
    parser.add_option(name='-type',
                      type_value='multiple_choice',
                      description='Change file type',
                      mandatory=False,
                      example=['uint8', 'int16', 'int32', 'float32', 'complex64', 'float64', 'int8', 'uint16', 'uint32', 'int64', 'uint64'])

    parser.usage.addSection("\nOrientation operations: ")
    parser.add_option(name="-getorient",
                      description='Get orientation of the input image',
                      mandatory=False)
    parser.add_option(name="-setorient",
                      type_value="multiple_choice",
                      description='Set orientation of the input image',
                      mandatory=False,
                      example='RIP LIP RSP LSP RIA LIA RSA LSA IRP ILP SRP SLP IRA ILA SRA SLA RPI LPI RAI LAI RPS LPS RAS LAS PRI PLI ARI ALI PRS PLS ARS ALS IPR SPR IAR SAR IPL SPL IAL SAL PIR PSR AIR ASR PIL PSL AIL ASL'.split())
    parser.add_option(name="-setorient-data",
                      type_value="multiple_choice",
                      description='Set orientation of the input image\'s data. Use with care !ro',
                      mandatory=False,
                      example='RIP LIP RSP LSP RIA LIA RSA LSA IRP ILP SRP SLP IRA ILA SRA SLA RPI LPI RAI LAI RPS LPS RAS LAS PRI PLI ARI ALI PRS PLS ARS ALS IPR SPR IAR SAR IPL SPL IAL SAL PIR PSR AIR ASR PIL PSL AIL ASL'.split())

    parser.usage.addSection("\nMulti-component operations on ITK composite warping fields:")
    parser.add_option(name='-mcs',
                      description='Multi-component split: Split ITK warping field into three separate displacement fields. The suffix _X, _Y and _Z will be added to the input file name.',
                      mandatory=False)
    parser.add_option(name='-omc',
                      description='Multi-component merge: Merge inputted images into one multi-component image. Requires several inputs.',
                      mandatory=False)

    parser.usage.addSection("\nWarping field operations:")
    parser.add_option(name='-display-warp',
                      description='Create a grid and deform it using provided warping field.',
                      mandatory=False)

    parser.usage.addSection("\nMisc")
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="""Verbose. 0: nothing. 1: basic. 2: extended.""",
                      mandatory=False,
                      default_value=param_default.verbose,
                      example=['0', '1', '2'])
    return parser


def main(args=None):

    # initializations
    output_type = ''
    param = Param()
    dim_list = ['x', 'y', 'z', 't']

    # check user arguments
    if not args:
        args = sys.argv[1:]

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(args)
    fname_in = arguments["-i"]
    n_in = len(fname_in)
    verbose = int(arguments['-v'])

    if "-o" in arguments:
        fname_out = arguments["-o"]
    else:
        fname_out = None

    # Open file(s)
    # im_in_list = [Image(fn) for fn in fname_in]

    # run command
    if "-concat" in arguments:
        dim = arguments["-concat"]
        assert dim in dim_list
        dim = dim_list.index(dim)
        im_out = [concat_data(fname_in, dim)]  # TODO: adapt to fname_in

    elif "-copy-header" in arguments:
        im_in = Image(fname_in[0])
        im_dest = Image(arguments["-copy-header"])
        im_out = [copy_header(im_in, im_dest)]

    elif '-display-warp' in arguments:
        im_in = fname_in[0]
        visualize_warp(im_in, fname_grid=None, step=3, rm_tmp=True)
        im_out = None

    elif "-getorient" in arguments:
        im_in = Image(fname_in[0])
        orient = orientation(im_in, get=True, verbose=verbose)
        im_out = None

    elif '-keep-vol' in arguments:
        index_vol = arguments['-keep-vol']
        im_in = Image(fname_in[0])
        im_out = [remove_vol(im_in, index_vol, todo='keep')]

    elif '-mcs' in arguments:
        im_in = Image(fname_in[0])
        if n_in != 1:
            printv(parser.usage.generate(error='ERROR: -mcs need only one input'))
        if len(im_in.data.shape) != 5:
            printv(parser.usage.generate(error='ERROR: -mcs input need to be a multi-component image'))
        im_out = multicomponent_split(im_in)

    elif '-omc' in arguments:
        im_ref = Image(fname_in[0])
        for fname in fname_in:
            im = Image(fname)
            if im.data.shape != im_ref.data.shape:
                printv(parser.usage.generate(error='ERROR: -omc inputs need to have all the same shapes'))
            del im
        im_out = [multicomponent_merge(fname_in)]  # TODO: adapt to fname_in

    elif "-pad" in arguments:
        im_in = Image(fname_in[0])
        ndims = len(im_in.getDataShape())
        if ndims != 3:
            printv('ERROR: you need to specify a 3D input file.', 1, 'error')
            return

        pad_arguments = arguments["-pad"].split(',')
        if len(pad_arguments) != 3:
            printv('ERROR: you need to specify 3 padding values.', 1, 'error')

        padx, pady, padz = pad_arguments
        padx, pady, padz = int(padx), int(pady), int(padz)
        im_out = [pad_image(im_in, pad_x_i=padx, pad_x_f=padx, pad_y_i=pady,
                            pad_y_f=pady, pad_z_i=padz, pad_z_f=padz)]

    elif "-pad-asym" in arguments:
        im_in = Image(fname_in[0])
        ndims = len(im_in.getDataShape())
        if ndims != 3:
            printv('ERROR: you need to specify a 3D input file.', 1, 'error')
            return

        pad_arguments = arguments["-pad-asym"].split(',')
        if len(pad_arguments) != 6:
            printv('ERROR: you need to specify 6 padding values.', 1, 'error')

        padxi, padxf, padyi, padyf, padzi, padzf = pad_arguments
        padxi, padxf, padyi, padyf, padzi, padzf = int(padxi), int(padxf), int(padyi), int(padyf), int(padzi), int(padzf)
        im_out = [pad_image(im_in, pad_x_i=padxi, pad_x_f=padxf, pad_y_i=padyi, pad_y_f=padyf, pad_z_i=padzi, pad_z_f=padzf)]

    elif '-remove-vol' in arguments:
        index_vol = arguments['-remove-vol']
        im_in = Image(fname_in[0])
        im_out = [remove_vol(im_in, index_vol, todo='remove')]

    elif "-setorient" in arguments:
        sct.printv(fname_in[0])
        im_in = Image(fname_in[0])
        im_out = [orientation(im_in, ori=arguments["-setorient"], set=True, verbose=verbose, fname_out=fname_out)]

    elif "-setorient-data" in arguments:
        im_in = Image(fname_in[0])
        im_out = [orientation(im_in, ori=arguments["-setorient-data"], set_data=True, verbose=verbose)]

    elif "-split" in arguments:
        dim = arguments["-split"]
        assert dim in dim_list
        im_in = Image(fname_in[0])
        dim = dim_list.index(dim)
        im_out = split_data(im_in, dim)

    elif '-type' in arguments:
        output_type = arguments['-type']
        im_in = Image(fname_in[0])
        im_out = [im_in]  # TODO: adapt to fname_in

    else:
        im_out = None
        printv(parser.usage.generate(error='ERROR: you need to specify an operation to do on the input image'))

    # Write output
    if im_out is not None:
        printv('Generate output files...', verbose)
        # if only one output
        if len(im_out) == 1:
            im_out[0].setFileName(fname_out) if fname_out is not None else None
            im_out[0].save(squeeze_data=False, type=output_type)
        if '-mcs' in arguments:
            # use input file name and add _X, _Y _Z. Keep the same extension
            fname_out = []
            for i_dim in xrange(3):
                fname_out.append(add_suffix(fname_in[0], '_' + dim_list[i_dim].upper()))
                im_out[i_dim].setFileName(fname_out[i_dim])
                im_out[i_dim].save()
        if '-split' in arguments:
            # use input file name and add _"DIM+NUMBER". Keep the same extension
            fname_out = []
            for i, im in enumerate(im_out):
                fname_out.append(add_suffix(fname_in[0], '_' + dim_list[dim].upper() + str(i).zfill(4)))
                im.setFileName(fname_out[i])
                im.save()

        # To view results
        printv('Finished. To view results, type:', param.verbose)
        printv('fslview ' + str(fname_out) + ' &', param.verbose, 'info')
    elif "-getorient" in arguments:
        sct.printv(orient)
    elif '-display-warp' in arguments:
        printv('Warping grid generated.', verbose, 'info')


def pad_image(im, pad_x_i=0, pad_x_f=0, pad_y_i=0, pad_y_f=0, pad_z_i=0, pad_z_f=0):
    from numpy import zeros, dot
    nx, ny, nz, nt, px, py, pz, pt = im.dim
    pad_x_i, pad_x_f, pad_y_i, pad_y_f, pad_z_i, pad_z_f = int(pad_x_i), int(pad_x_f), int(pad_y_i), int(pad_y_f), int(pad_z_i), int(pad_z_f)

    if len(im.data.shape) == 2:
        new_shape = list(im.data.shape)
        new_shape.append(1)
        im.data = im.data.reshape(new_shape)

    # initialize padded_data, with same type as im.data
    padded_data = zeros((nx + pad_x_i + pad_x_f, ny + pad_y_i + pad_y_f, nz + pad_z_i + pad_z_f), dtype=im.data.dtype)

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
    im_out.data = padded_data  # done after the call of the function
    im_out.setFileName(im_out.file_name + '_pad' + im_out.ext)

    # adapt the origin in the sform and qform matrix
    new_origin = dot(im_out.hdr.get_qform(), [-pad_x_i, -pad_y_i, -pad_z_i, 1])

    im_out.hdr.structarr['qoffset_x'] = new_origin[0]
    im_out.hdr.structarr['qoffset_y'] = new_origin[1]
    im_out.hdr.structarr['qoffset_z'] = new_origin[2]
    im_out.hdr.structarr['srow_x'][-1] = new_origin[0]
    im_out.hdr.structarr['srow_y'][-1] = new_origin[1]
    im_out.hdr.structarr['srow_z'][-1] = new_origin[2]

    return im_out


def copy_header(im_src, im_dest):
    """
    Copy header from the source image to the destination image
    :param im_src: source image
    :param im_dest: destination image
    :return im_src: destination data with the source header
    """
    im_out = im_src.copy()
    im_out.data = im_dest.data
    im_out.setFileName(im_dest.absolutepath)
    return im_out


def split_data(im_in, dim):
    """
    Split data
    :param im_in: input image.
    :param dim: dimension: 0, 1, 2, 3.
    :return: list of split images
    """
    from numpy import array_split
    dim_list = ['x', 'y', 'z', 't']
    # Parse file name
    # Open first file.
    data = im_in.data
    if dim + 1 > len(shape(data)):  # in case input volume is 3d and dim=t
        data = data[..., newaxis]
    # Split data into list
    data_split = array_split(data, data.shape[dim], dim)
    # Write each file
    im_out_list = []
    for i, dat in enumerate(data_split):
        im_out = im_in.copy()
        im_out.data = dat
        im_out.setFileName(im_out.file_name + '_' + dim_list[dim].upper() + str(i).zfill(4) + im_out.ext)
        im_out_list.append(im_out)

    return im_out_list


def concat_data(fname_in_list, dim, pixdim=None):
    """
    Concatenate data
    :param im_in_list: list of images.
    :param dim: dimension: 0, 1, 2, 3.
    :param pixdim: pixel resolution to join to image header
    :return im_out: concatenated image
    """
    # WARNING: calling concat_data in python instead of in command line causes a non understood issue (results are different with both options)
    from numpy import concatenate, expand_dims

    dat_list = []
    data_concat_list = []

    # check if shape of first image is smaller than asked dim to concatenate along
    # data0 = Image(fname_in_list[0]).data
    # if len(data0.shape) <= dim:
    #     expand_dim = True
    # else:
    #     expand_dim = False

    for i, fname in enumerate(fname_in_list):
        # if there is more than 100 images to concatenate, then it does it iteratively to avoid memory issue.
        if i != 0 and i % 100 == 0:
            data_concat_list.append(concatenate(dat_list, axis=dim))
            im = Image(fname)
            dat = im.data
            # if image shape is smaller than asked dim, then expand dim
            if len(dat.shape) <= dim:
                dat = expand_dims(dat, dim)
            dat_list = [dat]
            del im
            del dat
        else:
            im = Image(fname)
            dat = im.data
            # if image shape is smaller than asked dim, then expand dim
            if len(dat.shape) <= dim:
                dat = expand_dims(dat, dim)
            dat_list.append(dat)
            del im
            del dat
    if data_concat_list:
        data_concat_list.append(concatenate(dat_list, axis=dim))
        data_concat = concatenate(data_concat_list, axis=dim)
    else:
        data_concat = concatenate(dat_list, axis=dim)
    # write file
    im_out = Image(fname_in_list[0]).copy()
    im_out.data = data_concat
    im_out.setFileName(im_out.file_name + '_concat' + im_out.ext)

    if pixdim is not None:
        im_out.hdr['pixdim'] = pixdim

    return im_out


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
        index_vol = [i for i in range(0, nt) if not i in index_vol_user]
    elif todo == 'keep':
        index_vol = index_vol_user
    else:
        printv('ERROR: wrong assignment of variable "todo"', 1, 'error')
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
    # im_warp3d.setFileName(fname_warp3d)
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
        im.setFileName(im.file_name + '_' + str(i) + im.ext)
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
    im_out.setFileName(im_out.file_name + '_multicomponent' + im_out.ext)
    return im_out


def orientation(im, ori=None, set=False, get=False, set_data=False, verbose=1, fname_out=''):
    verbose = 0 if get else verbose
    printv('Get dimensions of data...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = get_dimension(im)

    printv(str(nx) + ' x ' + str(ny) + ' x ' + str(nz) + ' x ' + str(nt), verbose)

    # if data are 2d or 3d, get orientation from header using fslhd

    if (nz == 1 or nt == 1) and len(im.data.shape) < 5:
        if get:
            try:
                printv('Get orientation...', verbose)
                im_out = None
                ori = get_orientation(im)
            except Exception, e:
                printv('ERROR: an error occurred: ' + str(e), verbose, 'error')
            return ori
        elif set:
            # set orientation
            printv('Change orientation...', verbose)
            im_out = set_orientation(im, ori)
        elif set_data:
            im_out = set_orientation(im, ori, True)
        else:
            im_out = None

    else:
        from os import chdir
        # 4D data: split along T dimension
        # or 5D data: split along 5th dimension
        # Create a temporary directory and go in it
        tmp_folder = tmp_create(verbose)
        chdir(tmp_folder)
        if len(im.data.shape) == 5 and im.data.shape[-1] not in [0, 1]:
            # 5D data
            printv('Split along 5th dimension...', verbose)
            im_split_list = multicomponent_split(im)
            dim = 5
        else:
            # 4D data
            printv('Split along T dimension...', verbose)
            im_split_list = split_data(im, 3)
            dim = 4
        for im_s in im_split_list:
            im_s.save(verbose=verbose)

        if get:
            # get orientation
            printv('Get orientation...', verbose)
            im_out = None
            ori = get_orientation(im_split_list[0])
            chdir('..')
            run('rm -rf ' + tmp_folder, error_exit='warning')
            return ori
        elif set:
            # set orientation
            printv('Change orientation...', verbose)
            im_changed_ori_list = []
            for im_s in im_split_list:
                im_set = set_orientation(im_s, ori)
                im_changed_ori_list.append(im_set)
            printv('Merge file back...', verbose)
            if dim == 4:
                im_out = concat_data(im_changed_ori_list, 3)
            elif dim == 5:
                fname_changed_ori_list = [im_ch_ori.absolutepath for im_ch_ori in im_changed_ori_list]
                im_out = multicomponent_merge(fname_changed_ori_list)
        elif set_data:
            printv('Set orientation of the data only is not compatible with 4D data...', verbose, 'error')
        else:
            im_out = None

        # Go back to previous directory:
        chdir('..')
        run('rm -rf ' + tmp_folder, error_exit='warning')

    if fname_out:
        im_out.setFileName(fname_out)
        if fname_out != im.file_name + '_' + ori + im.ext:
            run('rm -f ' + im.file_name + '_' + ori + im.ext)
    else:
        im_out.setFileName(im.file_name + '_' + ori + im.ext)
    return im_out


def get_orientation(im):
    from nibabel import orientations
    orientation_dic = {
        (0, 1): 'L',
        (0, -1): 'R',
        (1, 1): 'P',
        (1, -1): 'A',
        (2, 1): 'I',
        (2, -1): 'S',
    }

    orientation_matrix = orientations.io_orientation(im.hdr.get_best_affine())
    ori = orientation_dic[tuple(orientation_matrix[0])] + orientation_dic[tuple(orientation_matrix[1])] + orientation_dic[tuple(orientation_matrix[2])]

    return ori


def get_orientation_3d(im, filename=False):
    """
    Get orientation from 3D data
    :param im:
    :return:
    """
    from sct_utils import run
    string_out = 'Input image orientation : '
    # get orientation
    if filename:
        status, output = run('isct_orientation3d -i ' + im + ' -get ', 0)
    else:
        status, output = run('isct_orientation3d -i ' + im.absolutepath + ' -get ', 0)
    # check status
    if status != 0:
        printv('ERROR in get_orientation.', 1, 'error')
    orientation = output[output.index(string_out) + len(string_out):]
    # orientation = output[26:]
    return orientation


def set_orientation(im, orientation, data_inversion=False, filename=False, fname_out=''):
    """
    Set orientation on image
    :param im: either Image object or file name. Carefully set param filename.
    :param orientation:
    :param data_inversion:
    :param filename:
    :return:
    """

    if fname_out:
        pass
    elif filename:
        path, fname, ext = extract_fname(im)
        fname_out = fname + '_' + orientation + ext
    else:
        fname_out = im.file_name + '_' + orientation + im.ext

    if not data_inversion:
        from sct_utils import run
        if filename:
            run('isct_orientation3d -i ' + im + ' -orientation ' + orientation + ' -o ' + fname_out, 0)
            im_out = fname_out
        else:
            fname_in = im.absolutepath
            if not os.path.exists(fname_in):
                im.save()
            run('isct_orientation3d -i ' + im.absolutepath + ' -orientation ' + orientation + ' -o ' + fname_out, 0)
            im_out = Image(fname_out)
    else:
        im_out = im.copy()
        im_out.change_orientation(orientation, True)
        im_out.setFileName(fname_out)
    return im_out


def visualize_warp(fname_warp, fname_grid=None, step=3, rm_tmp=True):
    if fname_grid is None:
        from numpy import zeros
        tmp_dir = tmp_create()
        im_warp = Image(fname_warp)
        status, out = run('fslhd ' + fname_warp)
        from os import chdir
        chdir(tmp_dir)
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
        im_grid.setFileName(fname_grid)
        im_grid.save()
        fname_grid_resample = add_suffix(fname_grid, '_resample')
        run('sct_resample -i ' + fname_grid + ' -f 3x3x1 -x nn -o ' + fname_grid_resample)
        fname_grid = tmp_dir + fname_grid_resample
        chdir('..')
    path_warp, file_warp, ext_warp = extract_fname(fname_warp)
    grid_warped = path_warp + extract_fname(fname_grid)[1] + '_' + file_warp + ext_warp
    run('sct_apply_transfo -i ' + fname_grid + ' -d ' + fname_grid + ' -w ' + fname_warp + ' -o ' + grid_warped)
    if rm_tmp:
        run('rm -rf ' + tmp_dir, error_exit='warning')


if __name__ == "__main__":
    sct.start_stream_logger()
    # # initialize parameters
    param = Param()
    # call main function
    main()
