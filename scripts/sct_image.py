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
from sct_utils import extract_fname, printv


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

    parser.usage.addSection('\nBasic operations:')
    parser.add_option(name="-pad",
                      type_value="str",
                      description='Pad 3d image. Specify padding as: "x,y,z" (in voxel)',
                      mandatory=False,
                      example='0,0,1')

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
    im_in = [Image(fn) for fn in fname_in]  # 3d or 4d numpy array

    # run command
    if "-pad" in arguments:
        # TODO: check input is 3d
        padx, pady, padz = arguments["-pad"].split(',')
        padx, pady, padz = int(padx), int(pady), int(padz)
        im_out = [pad_image(im_in[0], padding_x=padx, padding_y=pady, padding_z=padz)]

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
            from sct_utils import add_suffix
            im.setFileName(add_suffix(fname_out, '_'+dim_list[i]))
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
    im.data = padded_data  # done after the call of the function

    # adapt the origin in the sform and qform matrix
    new_origin = dot(im.hdr.get_best_affine(), [-padding_x, -padding_y, -padding_z, 1])

    im.hdr.structarr['qoffset_x'] = new_origin[0]
    im.hdr.structarr['qoffset_y'] = new_origin[1]
    im.hdr.structarr['qoffset_z'] = new_origin[2]
    im.hdr.structarr['srow_x'][-1] = new_origin[0]
    im.hdr.structarr['srow_y'][-1] = new_origin[1]
    im.hdr.structarr['srow_z'][-1] = new_origin[2]

    return im


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
    im_out = [im for dat in data_out]
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
    im_out = im_list[0]
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
