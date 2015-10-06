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

import sys
from numpy import concatenate, shape, newaxis
from msct_parser import Parser
from msct_image import Image
from sct_utils import extract_fname, printv, add_suffix


class Param:
    def __init__(self):
        self.verbose = '1'

# PARSER
# ==========================================================================================
def get_parser():

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Perform mathematical operations on images. Some inputs can be either a number or a 4d image or several 3d images separated with ","')
    parser.add_option(name="-i",
                      type_value=[[','], 'file'],
                      description="Input file(s). If several inputs: separate them by a coma without white space.\n",
                      mandatory=True,
                      example="data.nii.gz")
    parser.add_option(name="-o",
                      type_value='file_output',
                      description='Output file.',
                      mandatory=True,
                      example='data_pad.nii.gz')

    parser.usage.addSection('\nBasic image operations:')
    parser.add_option(name="-pad",
                      type_value="str",
                      description='Pad 3d image. Specify padding as: "x,y,z" (in voxel)',
                      mandatory=False,
                      example='0,0,1')
    parser.add_option(name="-copy-header",
                      type_value="file",
                      description='Copy the header of the input image (specified in -i) to the destination image (specified here)',
                      mandatory=False,
                      example='data_dest.nii.gz')
    parser.add_option(name="-split",
                      type_value="multiple_choice",
                      description='Split data along the specified dimension',
                      mandatory=False,
                      example=['x', 'y', 'z', 't'])
    parser.add_option(name="-concat",
                      type_value="multiple_choice",
                      description='Concatenate data along the specified dimension',
                      mandatory=False,
                      example=['x', 'y', 'z', 't'])

    parser.usage.addSection("\nMulti-component operations:")
    parser.add_option(name='-mcs',
                      description='Multi-component split. Outputs the components separately. (The sufix _x, _y and _z are added to the specified output) \n'
                                  'Only one input',
                      mandatory=False)
    parser.add_option(name='-omc',
                      description='Multi-component output. Merge inputted images into one multi-component image. (need several inputs.)',
                      mandatory=False)

    parser.usage.addSection("\nMisc")
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="""Verbose. 0: nothing. 1: basic. 2: extended.""",
                      mandatory=False,
                      default_value=param.verbose,
                      example=['0', '1', '2'])
    return parser


# MAIN
# ==========================================================================================
def main(args = None):
    dim_list = ['x', 'y', 'z', 't']

    if not args:
        args = sys.argv[1:]

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    fname_in = arguments["-i"]
    n_in = len(fname_in)
    fname_out = arguments["-o"]
    verbose = int(arguments['-v'])

    # Open file(s)
    im_in = [Image(fn) for fn in fname_in]

    # run command
    if "-pad" in arguments:
        # TODO: check input is 3d
        padx, pady, padz = arguments["-pad"].split(',')
        padx, pady, padz = int(padx), int(pady), int(padz)
        im_out = [pad_image(im_in[0], padding_x=padx, padding_y=pady, padding_z=padz)]

    elif "-copy-header" in arguments:
        im_dest = Image(arguments["-copy-header"])
        im_out = [copy_header(im_in[0], im_dest)]

    elif "-split" in arguments:
        dim = arguments["-split"]
        assert dim in dim_list
        dim = dim_list.index(dim)
        im_out = split_data(im_in[0], dim)

    elif '-mcs' in arguments:
        if n_in != 1:
            printv(parser.usage.generate(error='ERROR: -mcs need only one input'))
        if len(im_in[0].data.shape) != 5:
            printv(parser.usage.generate(error='ERROR: -mcs input need to be a multi-component image'))
        im_out = multicomponent_split(im_in[0])

    elif '-omc' in arguments:
        for im in im_in:
            if im.data.shape != im_in[0].data.shape:
                printv(parser.usage.generate(error='ERROR: -omc inputs need to have all the same shapes'))
        im_out = [multicomponent_merge(im_in)]
    else:
        im_out = None
        printv(parser.usage.generate(error='ERROR: you need to specify an operation to do on the input image'))

    # Write output
    if len(im_out) == 1:
        im_out[0].setFileName(fname_out)
        im_out[0].save()
    else:
        for i, im in enumerate(im_out):
            if len(im_out)<= len(dim_list):
                suffix = '_'+dim_list[i].upper()
            else:
                suffix = '_'+str(i)
            if "-split" in arguments:
                suffix = '_'+dim_list[dim].upper()+str(i).zfill(4)
            im.setFileName(add_suffix(fname_out, suffix))
            im.save()

    printv('Created file(s):\n--> '+str([im.file_name+im.ext for im in im_out])+'\n', verbose, 'info')


def pad_image(im, padding_x=0, padding_y=0, padding_z=0):
    from numpy import zeros, dot
    nx, ny, nz, nt, px, py, pz, pt = im.dim
    padding_x, padding_y, padding_z = int(padding_x), int(padding_y), int(padding_z)
    padded_data = zeros((nx+2*padding_x, ny+2*padding_y, nz+2*padding_z))

    if padding_x == 0:
        padxi = None
        padxf = None
    else:
        padxi=padding_x
        padxf=-padding_x

    if padding_y == 0:
        padyi = None
        padyf = None
    else:
        padyi = padding_y
        padyf = -padding_y

    if padding_z == 0:
        padzi = None
        padzf = None
    else:
        padzi = padding_z
        padzf = -padding_z

    padded_data[padxi:padxf, padyi:padyf, padzi:padzf] = im.data
    im_out = im.copy()
    im_out.data = padded_data  # done after the call of the function

    # adapt the origin in the sform and qform matrix
    new_origin = dot(im_out.hdr.get_best_affine(), [-padding_x, -padding_y, -padding_z, 1])

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
    if dim+1 > len(shape(data)):  # in case input volume is 3d and dim=t
        data = data[..., newaxis]
    # Split data into list
    data_split = array_split(data, data.shape[dim], dim)
    # Write each file
    im_out_list = []
    for i, dat in enumerate(data_split):
        im_out = im_in.copy()
        im_out.data = dat
        im_out.setFileName(add_suffix(im_out.absolutepath, '_'+dim_list[dim].upper()+str(i).zfill(4)))
        im_out_list.append(im_out)

    return im_out_list


def concat_data(im_in_list, dim):
    """
    Concatenate data
    :param im_in_list: list of images.
    :param dim: dimension: 0, 1, 2, 3.
    :return im_out: concatenated image
    """
    from numpy import concatenate, expand_dims

    data_list = [im.data for im in im_in_list]
    # expand dimension of all elements in the list if necessary
    if dim > im_in_list[0].data.ndim-1:
        list_data = [expand_dims(dat, dim) for dat in data_list]
    # concatenate
    try:
        data_concat = concatenate(data_list, axis=dim)
    except Exception as e:
        printv('\nERROR: Concatenation on line {}'.format(sys.exc_info()[-1].tb_lineno)+'\n'+str(e)+'\n', 1, 'error')
        data_concat = None

    # write file
    im_out = im_in_list[0].copy()
    im_out.data = data_concat

    return im_out


def multicomponent_split(im):
    from numpy import reshape
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
    im_out = [im.copy() for dat in data_out]
    for i, im in enumerate(im_out):
        im.data = data_out[i]
        im.hdr.set_intent('vector', (), '')
    return im_out


def multicomponent_merge(im_list):
    from numpy import zeros, reshape
    # WARNING: output multicomponent is not optimal yet, some issues may be related to the use of this function
    data_list = [im.data for im in im_list]
    new_shape = list(data_list[0].shape)
    if len(new_shape) == 3:
        new_shape.append(1)
    new_shape.append(len(data_list))
    new_shape = tuple(new_shape)

    data_out = zeros(new_shape)
    for i, dat in enumerate(data_list):
        if len(dat.shape) == 2:
            data_out[:, :, 0, 0, i] = dat.astype('float32')
        elif len(dat.shape) == 3:
            data_out[:, :, :, 0, i] = dat.astype('float32')
        elif len(dat.shape) == 4:
            data_out[:, :, :, :, i] = dat.astype('float32')
    im_out = im_list[0].copy()
    im_out.data = data_out.astype('float32')
    im_out.hdr.set_intent('vector', (), '')
    return im_out



# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # # initialize parameters
    param = Param()
    # call main function
    main()
