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
import getopt
import commands
import sct_utils as sct
import time
import numpy
import nibabel
from scipy import ndimage
from sct_image import get_orientation
from sct_convert import convert
from msct_image import Image
from sct_image import copy_header, concat_data
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
        self.size = 41  # in voxel. if gaussian, size corresponds to sigma.
        self.even = 0
        self.file_prefix = 'mask_'  # output prefix
        self.verbose = 1
        self.remove_tmp_files = 1
        self.offset = '0,0'


# main
#=======================================================================================================================
def main():

    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        # get path of the testing data
        status, path_sct_data = commands.getstatusoutput('echo $SCT_TESTING_DATA_DIR')
        param.fname_data = path_sct_data+'/mt/mt1.nii.gz'
        param.process = 'point,'+path_sct_data+'/mt/mt1_point.nii.gz' #'centerline,/Users/julien/data/temp/sct_example_data/t2/t2_centerlinerpi.nii.gz'  #coord,68x69'
        param.shape = 'cylinder'
        param.size = 20
        param.remove_tmp_files = 1
        param.verbose = 1
    else:
        # Check input parameters
        parser = get_parser()
        arguments = parser.parse(sys.argv[1:])

        param.fname_data = arguments['-i']

        if '-p' in arguments:
            param.process = arguments['-p']
            if param.process[0] not in param.process_list:
                sct.printv(parser.usage.generate(error='ERROR: Process '+param.process[0]+' is not recognized.'))
        if '-size' in arguments:
            param.size = arguments['-size']
        if '-f' in arguments:
            param.shape = arguments['-f']
        if '-o' in arguments:
            param.fname_out = arguments['-o']
        if '-r' in arguments:
            param.remove_tmp_files = int(arguments['-r'])
        if '-v' in arguments:
            param.verbose = int(arguments['-v'])

    # run main program
    create_mask()


# create_mask
#=======================================================================================================================
def create_mask():
    fsloutput = 'export FSLOUTPUTTYPE=NIFTI; '  # for faster processing, all outputs are in NIFTI

    # parse argument for method
    method_type = param.process[0]
    # check method val
    if not method_type == 'center':
        method_val = param.process[1]

    # check existence of input files
    if method_type == 'centerline':
        sct.check_file_exist(method_val, param.verbose)

    # check if orientation is RPI
    sct.printv('\nCheck if orientation is RPI...', param.verbose)
    ori = get_orientation(param.fname_data, filename=True)
    if not ori == 'RPI':
        sct.printv('\nERROR in '+os.path.basename(__file__)+': Orientation of input image should be RPI. Use sct_image -setorient to put your image in RPI.\n', 1, 'error')

    # Extract path/file/extension
    path_data, file_data, ext_data = sct.extract_fname(param.fname_data)

    # Get output folder and file name
    if param.fname_out == '':
        param.fname_out = param.file_prefix+file_data+ext_data
    #fname_out = os.path.abspath(path_out+file_out+ext_out)

    # create temporary folder
    sct.printv('\nCreate temporary folder...', param.verbose)
    path_tmp = sct.slash_at_the_end('tmp.'+time.strftime("%y%m%d%H%M%S"), 1)
    sct.run('mkdir '+path_tmp, param.verbose)

    # Copying input data to tmp folder and convert to nii
    sct.printv('\nCopying input data to tmp folder and convert to nii...', param.verbose)
    convert(param.fname_data, path_tmp+'data.nii')
    # sct.run('cp '+param.fname_data+' '+path_tmp+'data'+ext_data, param.verbose)
    if method_type == 'centerline':
        convert(method_val, path_tmp+'centerline.nii.gz')

    # go to tmp folder
    os.chdir(path_tmp)

    # Get dimensions of data
    sct.printv('\nGet dimensions of data...', param.verbose)
    nx, ny, nz, nt, px, py, pz, pt = Image('data.nii').dim
    sct.printv('  ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz)+ ' x ' + str(nt), param.verbose)
    # in case user input 4d data
    if nt != 1:
        sct.printv('WARNING in '+os.path.basename(__file__)+': Input image is 4d but output mask will 3D.', param.verbose, 'warning')
        # extract first volume to have 3d reference
        nii = Image('data.nii')
        data3d = nii.data[:,:,:,0]
        nii.data = data3d
        nii.save()

    if method_type == 'coord':
        # parse to get coordinate
        coord = map(int, method_val.split('x'))

    if method_type == 'point':
        # get file name
        fname_point = method_val
        # extract coordinate of point
        sct.printv('\nExtract coordinate of point...', param.verbose)
        status, output = sct.run('sct_label_utils -i '+fname_point+' -p display-voxel', param.verbose)
        # parse to get coordinate
        coord = output[output.find('Position=')+10:-17].split(',')

    if method_type == 'center':
        # set coordinate at center of FOV
        coord = round(float(nx)/2), round(float(ny)/2)

    if method_type == 'centerline':
        # get name of centerline from user argument
        fname_centerline = 'centerline.nii.gz'
    else:
        # generate volume with line along Z at coordinates 'coord'
        sct.printv('\nCreate line...', param.verbose)
        fname_centerline = create_line('data.nii', coord, nz)

    # create mask
    sct.printv('\nCreate mask...', param.verbose)
    centerline = nibabel.load(fname_centerline)  # open centerline
    hdr = centerline.get_header()  # get header
    hdr.set_data_dtype('uint8')  # set imagetype to uint8
    data_centerline = centerline.get_data()  # get centerline
    z_centerline = [iz for iz in range(0, nz, 1) if data_centerline[:, :, iz].any()]
    nz = len(z_centerline)
    # get center of mass of the centerline
    cx = [0] * nz
    cy = [0] * nz
    for iz in range(0, nz, 1):
        cx[iz], cy[iz] = ndimage.measurements.center_of_mass(numpy.array(data_centerline[:, :, z_centerline[iz]]))
    # create 2d masks
    file_mask = 'data_mask'
    for iz in range(nz):
        center = numpy.array([cx[iz], cy[iz]])
        mask2d = create_mask2d(center, param.shape, param.size, nx, ny, param.even)
        # Write NIFTI volumes
        img = nibabel.Nifti1Image(mask2d, None, hdr)
        nibabel.save(img, (file_mask+str(iz)+'.nii'))
    # merge along Z
    # cmd = 'fslmerge -z mask '
    im_list = []
    for iz in range(nz):
        im_list.append(Image(file_mask+str(iz)+'.nii'))
    im_out = concat_data(im_list, 2)
    im_out.setFileName('mask.nii.gz')
    im_out.save()

    # copy geometry
    im_dat = Image('data.nii')
    im_mask = Image('mask.nii.gz')
    im_mask = copy_header(im_dat, im_mask)
    im_mask.save()

    # come back to parent folder
    os.chdir('..')

    # Generate output files
    sct.printv('\nGenerate output files...', param.verbose)
    sct.generate_output_file(path_tmp+'mask.nii.gz', param.fname_out)

    # Remove temporary files
    if param.remove_tmp_files == 1:
        sct.printv('\nRemove temporary files...', param.verbose)
        sct.run('rm -rf '+path_tmp, param.verbose, error_exit='warning')

    # to view results
    sct.printv('\nDone! To view results, type:', param.verbose)
    sct.printv('fslview '+param.fname_data+' '+param.fname_out+' -l Red -t 0.5 &', param.verbose, 'info')
    print


# create_line
# ==========================================================================================
def create_line(fname, coord, nz):

    # duplicate volume (assumes input file is nifti)
    sct.run('cp '+fname+' line.nii', param.verbose)

    # set all voxels to zero
    sct.run('sct_maths -i line.nii -mul 0 -o line.nii', param.verbose)

    cmd = 'sct_label_utils -i line.nii -o line.nii -p add -coord '
    for iz in range(nz):
        if iz == nz-1:
            cmd += str(int(coord[0]))+','+str(int(coord[1]))+','+str(iz)+',1'
        else:
            cmd += str(int(coord[0]))+','+str(int(coord[1]))+','+str(iz)+',1:'

    sct.run(cmd, param.verbose)

    return 'line.nii'


# create_mask2d
# ==========================================================================================
def create_mask2d(center, shape, size, nx, ny, even=0):
    # extract offset
    offset = param.offset.split(',')
    offset[0] = int(offset[0])
    offset[1] = int(offset[1])

    # initialize 2d grid
    xx, yy = numpy.mgrid[:nx, :ny]
    mask2d = numpy.zeros((nx, ny))
    xc = center[0]
    yc = center[1]
    if even != 0:
        radius = int(size / 2)
    else:
        radius = round(float(size + 1) / 2)  # add 1 because the radius includes the center.

    if shape == 'box':
        if even != 0:
            mask2d[xc - radius:xc + radius, yc - radius:yc + radius] = 1
        else:
            mask2d[xc-radius:xc+radius+1, yc-radius:yc+radius+1] = 1

    elif shape == 'cylinder':
        mask2d = ((xx+offset[0]-xc)**2 + (yy+offset[1]-yc)**2 <= radius**2)*1

    elif shape == 'gaussian':
        sigma = float(radius)
        mask2d = numpy.exp(-(((xx+offset[0]-xc)**2)/(2*(sigma**2)) + ((yy+offset[1]-yc)**2)/(2*(sigma**2))))

    # import matplotlib.pyplot as plt
    # plt.imshow(mask2d)
    # plt.show()

    return mask2d

def get_parser():
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
                      description='Process to generate mask and associated value.\n'
                                  '   coord: X,Y coordinate of center of mask. E.g.: coord,20x15\n'
                                  '   point: volume that contains a single point. E.g.: point,label.nii.gz\n'
                                  '   center: mask is created at center of FOV. In that case, "val" is not required.\n'
                                  '   centerline: volume that contains centerline. E.g.: centerline,my_centerline.nii',
                      mandatory=True,
                      default_value=param_default.process,
                      example=['centerline,data_centerline.nii.gz'])
    parser.add_option(name='-m',
                      type_value=None,
                      description='Process to generate mask and associated value.'
                                  '   coord: X,Y coordinate of center of mask. E.g.: coord,20x15'
                                  '   point: volume that contains a single point. E.g.: point,label.nii.gz'
                                  '   center: mask is created at center of FOV. In that case, "val" is not required.'
                                  '   centerline: volume that contains centerline. E.g.: centerline,my_centerline.nii',
                      mandatory=False,
                      deprecated_by='-p')
    parser.add_option(name='-size',
                      type_value='int',
                      description='Size in voxel. Odd values are better (for mask to be symmetrical). If shape=gaussian, size corresponds to "sigma"',
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
                      example=param.shape_list)
    parser.add_option(name='-o',
                      type_value='file_output',
                      description='Name of output mask.',
                      mandatory=False,
                      example=['data.nii'])
    parser.usage.addSection('MISC')
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
    param = Param()
    param_default = Param()
    main()
