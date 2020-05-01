#!/usr/bin/env python
#########################################################################################
#
# Create mask along z direction.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad
# Modified: 2014-10-11
#
# About the license: see the file LICENSE.TXT
#########################################################################################


# TODO: scale size in mm.

from __future__ import division, absolute_import

import sys
import os
import argparse

import numpy as np

import nibabel
from scipy import ndimage

import sct_utils as sct
import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.image import Image
from sct_image import concat_data
from spinalcordtoolbox.utils import Metavar, SmartFormatter


# DEFAULT PARAMETERS
class Param:
    def __init__(self):
        self.debug = 0
        self.fname_data = ''
        self.fname_out = ''
        self.process_list = ['coord', 'point', 'centerline', 'center']
        self.process = 'center'  # default method
        self.shape_list = ['cylinder', 'box', 'gaussian']
        self.shape = 'cylinder'  # default shape
        self.size = '41'  # in voxel. if gaussian, size corresponds to sigma.
        self.even = 0
        self.file_prefix = 'mask_'  # output prefix
        self.verbose = 1
        self.remove_temp_files = 1
        self.offset = '0,0'


def get_parser():
    # Initialize default parameters
    param_default = Param()
    # Initialize the parser

    parser = argparse.ArgumentParser(
        description='Create mask along z direction.',
        add_help=None,
        prog=os.path.basename(__file__).strip(".py"),
        formatter_class= SmartFormatter)

    mandatoryArguments = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatoryArguments.add_argument(
        '-i',
        required=True,
        help='Image to create mask on. Only used to get header. Must be 3D. Example: data.nii.gz',
        metavar=Metavar.file,
        )
    mandatoryArguments.add_argument(
        '-p',
        default=param_default.process,
        required=True,
        help='R|Process to generate mask.\n'
             '  <coord,XxY>: Center mask at the X,Y coordinates. (e.g. "coord,20x15")\n'
             '  <point,FILE>: Center mask at the X,Y coordinates of the label defined in input volume FILE. (e.g. "point,label.nii.gz")\n'
             '  <center>: Center mask in the middle of the FOV (nx/2, ny/2).\n'
             '  <centerline,FILE>: At each slice, the mask is centered at the spinal cord centerline, defined by the input segmentation FILE. (e.g. "centerline,t2_seg.nii.gz")',
        metavar=Metavar.str,
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")
    optional.add_argument(
        '-size',
        help='Size of the mask in the axial plane, given in pixel (Example: 35) or in millimeter (Example: 35mm). '
             'If shape=gaussian, size corresponds to "sigma" (Example: 45).',
        metavar=Metavar.str,
        required = False,
        default = param_default.size)
    optional.add_argument(
        '-f',
        help='Shape of the mask',
        required = False,
        default = param_default.shape,
        choices=('cylinder', 'box', 'gaussian'))
    optional.add_argument(
        '-o',
        metavar=Metavar.str,
        help='Name of output mask, Example: data.nii',
        required = False)
    optional.add_argument(
        "-r",
        type=int,
        help='Remove temporary files',
        required = False,
        default = 1,
        choices = (0, 1))
    optional.add_argument(
        "-v",
        type=int,
        help="Verbose: 0 = nothing, 1 = classic, 2 = expended ",
        required=False,
        choices=(0, 1, 2),
        default = 1)

    return parser


def main(args=None):
    """
    Main function
    :param args:
    :return:
    """
    # get parser args
    if args is None:
        args = None if sys.argv[1:] else ['--help']
    parser = get_parser()
    arguments = parser.parse_args(args=args)

    param = Param()
    param.fname_data = os.path.abspath(arguments.i)

    if arguments.p is not None:
        param.process = (arguments.p).split(',')
        if param.process[0] not in param.process_list:
            sct.printv(parser.error('ERROR: Process ' + param.process[0] + ' is not recognized.'))
    if arguments.size is not None:
        param.size = arguments.size
    if arguments.f is not None:
        param.shape = arguments.f
    if arguments.o is not None:
        param.fname_out = os.path.abspath(arguments.o)
    if arguments.r is not None:
        param.remove_temp_files = arguments.r

    param.verbose = arguments.v
    sct.init_sct(log_level=param.verbose, update=True)  # Update log level

    # run main program
    create_mask(param)


def create_mask(param):
    # parse argument for method
    method_type = param.process[0]
    # check method val
    if not method_type == 'center':
        method_val = param.process[1]

    # check existence of input files
    if method_type == 'centerline':
        sct.check_file_exist(method_val, param.verbose)

    # Extract path/file/extension
    path_data, file_data, ext_data = sct.extract_fname(param.fname_data)

    # Get output folder and file name
    if param.fname_out == '':
        param.fname_out = os.path.abspath(param.file_prefix + file_data + ext_data)

    path_tmp = sct.tmp_create(basename="create_mask", verbose=param.verbose)

    sct.printv('\nOrientation:', param.verbose)
    orientation_input = Image(param.fname_data).orientation
    sct.printv('  ' + orientation_input, param.verbose)

    # copy input data to tmp folder and re-orient to RPI
    Image(param.fname_data).change_orientation("RPI").save(os.path.join(path_tmp, "data_RPI.nii"))
    if method_type == 'centerline':
        Image(method_val).change_orientation("RPI").save(os.path.join(path_tmp, "centerline_RPI.nii"))
    if method_type == 'point':
        Image(method_val).change_orientation("RPI").save(os.path.join(path_tmp, "point_RPI.nii"))

    # go to tmp folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    # Get dimensions of data
    im_data = Image('data_RPI.nii')
    nx, ny, nz, nt, px, py, pz, pt = im_data.dim
    sct.printv('\nDimensions:', param.verbose)
    sct.printv(im_data.dim, param.verbose)
    # in case user input 4d data
    if nt != 1:
        sct.printv('WARNING in ' + os.path.basename(__file__) + ': Input image is 4d but output mask will be 3D from first time slice.', param.verbose, 'warning')
        # extract first volume to have 3d reference
        nii = msct_image.empty_like(Image('data_RPI.nii'))
        data3d = nii.data[:, :, :, 0]
        nii.data = data3d
        nii.save('data_RPI.nii')

    if method_type == 'coord':
        # parse to get coordinate
        coord = [x for x in map(int, method_val.split('x'))]

    if method_type == 'point':
        # get file name
        # extract coordinate of point
        sct.printv('\nExtract coordinate of point...', param.verbose)
        # TODO: change this way to remove dependence to sct.run. ProcessLabels.display_voxel returns list of coordinates
        status, output = sct.run(['sct_label_utils', '-i', 'point_RPI.nii', '-display'], verbose=param.verbose)
        # parse to get coordinate
        # TODO fixup... this is quite magic
        coord = output[output.find('Position=') + 10:-17].split(',')

    if method_type == 'center':
        # set coordinate at center of FOV
        coord = np.round(float(nx) / 2), np.round(float(ny) / 2)

    if method_type == 'centerline':
        # get name of centerline from user argument
        fname_centerline = 'centerline_RPI.nii'
    else:
        # generate volume with line along Z at coordinates 'coord'
        sct.printv('\nCreate line...', param.verbose)
        fname_centerline = create_line(param, 'data_RPI.nii', coord, nz)

    # create mask
    sct.printv('\nCreate mask...', param.verbose)
    centerline = nibabel.load(fname_centerline)  # open centerline
    hdr = centerline.get_header()  # get header
    hdr.set_data_dtype('uint8')  # set imagetype to uint8
    spacing = hdr.structarr['pixdim']
    data_centerline = centerline.get_data()  # get centerline
    # if data is 2D, reshape with empty third dimension
    if len(data_centerline.shape) == 2:
        data_centerline_shape = list(data_centerline.shape)
        data_centerline_shape.append(1)
        data_centerline = data_centerline.reshape(data_centerline_shape)
    z_centerline_not_null = [iz for iz in range(0, nz, 1) if data_centerline[:, :, iz].any()]
    # get center of mass of the centerline
    cx = [0] * nz
    cy = [0] * nz
    for iz in range(0, nz, 1):
        if iz in z_centerline_not_null:
            cx[iz], cy[iz] = ndimage.measurements.center_of_mass(np.array(data_centerline[:, :, iz]))
    # create 2d masks
    file_mask = 'data_mask'
    for iz in range(nz):
        if iz not in z_centerline_not_null:
            # write an empty nifty volume
            img = nibabel.Nifti1Image(data_centerline[:, :, iz], None, hdr)
            nibabel.save(img, (file_mask + str(iz) + '.nii'))
        else:
            center = np.array([cx[iz], cy[iz]])
            mask2d = create_mask2d(param, center, param.shape, param.size, im_data=im_data)
            # Write NIFTI volumes
            img = nibabel.Nifti1Image(mask2d, None, hdr)
            nibabel.save(img, (file_mask + str(iz) + '.nii'))

    fname_list = [file_mask + str(iz) + '.nii' for iz in range(nz)]
    im_out = concat_data(fname_list, dim=2).save('mask_RPI.nii.gz')

    im_out.change_orientation(orientation_input)
    im_out.header = Image(param.fname_data).header
    im_out.save(param.fname_out)

    # come back
    os.chdir(curdir)

    # Remove temporary files
    if param.remove_temp_files == 1:
        sct.printv('\nRemove temporary files...', param.verbose)
        sct.rmtree(path_tmp)

    sct.display_viewer_syntax([param.fname_data, param.fname_out], colormaps=['gray', 'red'], opacities=['', '0.5'])


def create_line(param, fname, coord, nz):
    """
    Create vertical line in 3D volume
    :param param:
    :param fname:
    :param coord:
    :param nz:
    :return:
    """

    # duplicate volume (assumes input file is nifti)
    sct.copy(fname, 'line.nii', verbose=param.verbose)

    # set all voxels to zero
    sct.run(['sct_maths', '-i', 'line.nii', '-mul', '0', '-o', 'line.nii'], param.verbose)

    cmd = ['sct_label_utils', '-i', 'line.nii', '-o', 'line.nii', '-create-add']
    for iz in range(nz):
        if iz == nz - 1:
            cmd += [str(int(coord[0])) + ',' + str(int(coord[1])) + ',' + str(iz) + ',1']
        else:
            cmd += [str(int(coord[0])) + ',' + str(int(coord[1])) + ',' + str(iz) + ',1:']

    sct.run(cmd, param.verbose)

    return 'line.nii'


def create_mask2d(param, center, shape, size, im_data):
    """
    Create a 2D mask
    :param param:
    :param center:
    :param shape:
    :param size:
    :param im_data: Image object for input data.
    :return:
    """
    # get dim
    nx, ny, nz, nt, px, py, pz, pt = im_data.dim
    # extract offset d = 2r+1 --> r=ceil((d-1)/2.0)
    offset = param.offset.split(',')
    offset[0] = int(offset[0])
    offset[1] = int(offset[1])
    # px, py = spacing[0], spacing[1]

    # initialize 2d grid
    xx, yy = np.mgrid[:nx, :ny]
    mask2d = np.zeros((nx, ny))
    xc = center[0]
    yc = center[1]
    if 'mm' in size:
        size = float(size[:-2])
        radius_x = np.ceil((int(np.round(size / px)) - 1) / 2.0)
        radius_y = np.ceil((int(np.round(size / py)) - 1) / 2.0)
    else:
        radius_x = np.ceil((int(size) - 1) / 2.0)
        radius_y = radius_x

    if shape == 'box':
        mask2d = ((abs(xx + offset[0] - xc) <= radius_x) & (abs(yy + offset[1] - yc) <= radius_y)) * 1

    elif shape == 'cylinder':
        mask2d = (((xx + offset[0] - xc) / radius_x) ** 2 + ((yy + offset[1] - yc) / radius_y) ** 2 <= 1) * 1

    elif shape == 'gaussian':
        sigma = float(radius_x)
        mask2d = np.exp(-(((xx + offset[0] - xc)**2) / (2 * (sigma**2)) + ((yy + offset[1] - yc)**2) / (2 * (sigma**2))))

    return mask2d


if __name__ == "__main__":
    sct.init_sct()
    main()
