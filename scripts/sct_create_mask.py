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

import sys
import os

import time

import numpy
import nibabel
from scipy import ndimage

import sct_utils as sct
import msct_image
from msct_image import Image
from sct_image import concat_data
from msct_parser import Parser


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
# param = Param()
# param_default = Param()


# main
#=======================================================================================================================
def main(args=None):

    if args is None:
        args = sys.argv[1:]
    param = Param()

    # Check input parameters
    parser = get_parser()
    arguments = parser.parse(args)

    param.fname_data = arguments['-i']

    if '-p' in arguments:
        param.process = arguments['-p']
        if param.process[0] not in param.process_list:
            sct.printv(parser.usage.generate(error='ERROR: Process ' + param.process[0] + ' is not recognized.'))
    if '-size' in arguments:
        param.size = arguments['-size']
    if '-f' in arguments:
        param.shape = arguments['-f']
    if '-o' in arguments:
        param.fname_out = os.path.abspath(arguments['-o'])
    if '-r' in arguments:
        param.remove_temp_files = int(arguments['-r'])
    if '-v' in arguments:
        param.verbose = int(arguments['-v'])

    # run main program
    create_mask(param)


# create_mask
#=======================================================================================================================
def create_mask(param):
    fsloutput = 'export FSLOUTPUTTYPE=NIFTI; '  # for faster processing, all outputs are in NIFTI

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

    sct.printv('\nCheck orientation...', param.verbose)
    orientation_input = Image(param.fname_data).orientation
    sct.printv('.. ' + orientation_input, param.verbose)
    reorient_coordinates = False

    # copy input data to tmp folder
    Image(param.fname_data).change_orientation("RPI").save(os.path.join(path_tmp, "data_RPI.nii"))
    if method_type == 'centerline':
        Image(method_val).change_orientation("RPI").save(os.path.join(path_tmp, "centerline_RPI.nii"))
    if method_type == 'point':
        Image(method_val).change_orientation("RPI").save(os.path.join(path_tmp, "point_RPI.nii"))

    # go to tmp folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    #
    # if method_type == 'centerline':
    #     orientation_centerline = Image(method_val).orientation
    #     if not orientation_centerline == 'RPI':
    #         sct.run('sct_image -i ' + method_val + ' -o ' + os.path.join(path_tmp, "centerline.nii.gz") + ' -setorient RPI -v 0', verbose=False)
    #     else:
    #         convert(method_val, path_tmp+'centerline.nii.gz')

    # Get dimensions of data
    sct.printv('\nGet dimensions of data...', param.verbose)
    nx, ny, nz, nt, px, py, pz, pt = Image('data_RPI.nii').dim
    sct.printv('  ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz) + ' x ' + str(nt), param.verbose)
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
        fname_point = method_val
        # extract coordinate of point
        sct.printv('\nExtract coordinate of point...', param.verbose)
        # TODO: change this way to remove dependence to sct.run. ProcessLabels.display_voxel returns list of coordinates
        status, output = sct.run(['sct_label_utils', '-i', 'point_RPI.nii', '-display'], verbose=param.verbose)
        # parse to get coordinate
        # TODO fixup... this is quite magic
        coord = output[output.find('Position=') + 10:-17].split(',')

    if method_type == 'center':
        # set coordinate at center of FOV
        coord = round(float(nx) / 2), round(float(ny) / 2)

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
            cx[iz], cy[iz] = ndimage.measurements.center_of_mass(numpy.array(data_centerline[:, :, iz]))
    # create 2d masks
    file_mask = 'data_mask'
    for iz in range(nz):
        if iz not in z_centerline_not_null:
            # write an empty nifty volume
            img = nibabel.Nifti1Image(data_centerline[:, :, iz], None, hdr)
            nibabel.save(img, (file_mask + str(iz) + '.nii'))
        else:
            center = numpy.array([cx[iz], cy[iz]])
            mask2d = create_mask2d(param, center, param.shape, param.size, nx, ny, even=param.even, spacing=spacing)
            # Write NIFTI volumes
            img = nibabel.Nifti1Image(mask2d, None, hdr)
            nibabel.save(img, (file_mask + str(iz) + '.nii'))
    # merge along Z
    # cmd = 'fslmerge -z mask '

    # CHANGE THAT CAN IMPACT SPEED:
    # related to issue #755, we cannot open more than 256 files at one time.
    # to solve this issue, we do not open more than 100 files
    '''
    im_list = []
    im_temp = []
    for iz in range(nz_not_null):
        if iz != 0 and iz % 100 == 0:
            im_temp.append(concat_data(im_list, 2))
            im_list = [Image(file_mask + str(iz) + '.nii')]
        else:
            im_list.append(Image(file_mask+str(iz)+'.nii'))

    if im_temp:
        im_temp.append(concat_data(im_list, 2))
        im_out = concat_data(im_temp, 2, no_expand=True)
    else:
        im_out = concat_data(im_list, 2)
    '''
    fname_list = [file_mask + str(iz) + '.nii' for iz in range(nz)]
    im_out = concat_data(fname_list, dim=2) \
     .save('mask_RPI.nii.gz')

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


# create_line
# ==========================================================================================
def create_line(param, fname, coord, nz):

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


# create_mask2d
# ==========================================================================================
def create_mask2d(param, center, shape, size, nx, ny, even=0, spacing=None):
    # extract offset d = 2r+1 --> r=ceil((d-1)/2.0)
    # s=11 -> r=5
    # s=10 -> r=5
    offset = param.offset.split(',')
    offset[0] = int(offset[0])
    offset[1] = int(offset[1])

    # initialize 2d grid
    xx, yy = numpy.mgrid[:nx, :ny]
    mask2d = numpy.zeros((nx, ny))
    xc = center[0]
    yc = center[1]
    if 'mm' in size:
        from numpy import ceil
        size = int(size[:-2])
        mean_spacing_xy = (spacing[1] + spacing[2]) / 2.0
        length = round(float(size) / mean_spacing_xy)
        radius = ceil((int(length) - 1) / 2.0)
    else:
        from numpy import ceil
        radius = ceil((int(size) - 1) / 2.0)

    if shape == 'box':
        mask2d[int(xc - radius):int(xc + radius) + 1, int(yc - radius):int(yc + radius) + 1] = 1

    elif shape == 'cylinder':
        mask2d = ((xx + offset[0] - xc)**2 + (yy + offset[1] - yc)**2 <= radius**2) * 1

    elif shape == 'gaussian':
        sigma = float(radius)
        mask2d = numpy.exp(-(((xx + offset[0] - xc)**2) / (2 * (sigma**2)) + ((yy + offset[1] - yc)**2) / (2 * (sigma**2))))

    # import matplotlib.pyplot as plt
    # plt.imshow(mask2d)
    # plt.show()

    return mask2d


def get_parser():
    # Initialize default parameters
    param_default = Param()
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Create mask along z direction.')
    parser.add_option(name='-i',
                      type_value='file',
                      description='Image to create mask on. Only used to get header. Must be 3D.',
                      mandatory=True,
                      example='data.nii.gz')
    parser.add_option(name='-p',
                      type_value=[[','], 'str'],
                      description='Process to generate mask.\n'
                                  'coord: X,Y coordinate of center of mask. E.g.: coord,20x15\n'
                                  'point: volume that contains a single point. E.g.: point,label.nii.gz\n'
                                  'center: mask is created at center of FOV.\n'
                                  'centerline: volume that contains centerline or segmentation. E.g.: centerline,t2_seg.nii.gz',
                      mandatory=True,
                      default_value=param_default.process,
                      example=['centerline,data_centerline.nii.gz'])
    parser.add_option(name='-m',
                      type_value=None,
                      description='Process to generate mask and associated value.\n'
                                  '  coord: X,Y coordinate of center of mask. E.g.: coord,20x15'
                                  '  point: volume that contains a single point. E.g.: point,label.nii.gz'
                                  '  center: mask is created at center of FOV. In that case, "val" is not required.'
                                  '  centerline: volume that contains centerline. E.g.: centerline,my_centerline.nii',
                      mandatory=False,
                      deprecated_by='-p')
    parser.add_option(name='-size',
                      type_value='str',
                      description='Size of the mask in the axial plane, given in pixel (ex: 35) or in millimeter (ex: 35mm). If shape=gaussian, size corresponds to "sigma"',
                      mandatory=False,
                      default_value=param_default.size,
                      example=['45'])
    parser.add_option(name='-s',
                      type_value=None,
                      description='Size in voxel. Odd values are better (for mask to be symmetrical). If shape=gaussian, size corresponds to "sigma"',
                      mandatory=False,
                      deprecated_by='-size')
    parser.add_option(name='-f',
                      type_value='multiple_choice',
                      description='Shape of the mask.',
                      mandatory=False,
                      default_value=param_default.shape,
                      example=param_default.shape_list)
    parser.add_option(name='-o',
                      type_value='file_output',
                      description='Name of output mask.',
                      mandatory=False,
                      example=['data.nii'])
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description='Remove temporary files.',
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])
    parser.add_option(name="-v",
                      type_value='multiple_choice',
                      description="verbose: 0 = nothing, 1 = classic, 2 = expended",
                      mandatory=False,
                      example=['0', '1', '2'],
                      default_value='1')

    return parser

#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    sct.init_sct()
    main()
