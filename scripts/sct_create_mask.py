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


# DEFAULT PARAMETERS
class Param:
    def __init__(self):
        self.debug = 0
        self.fname_data = ''
        self.fname_out = ''
        self.method_list = ['coord', 'point', 'centerline', 'center']
        self.method = 'center'  # default method
        self.shape_list = ['cylinder', 'box', 'gaussian']
        self.shape = 'cylinder'  # default shape
        self.size = 41  # in voxel. if gaussian, size corresponds to sigma.
        self.even = 0
        self.file_prefix = 'mask_'  # output prefix
        self.verbose = 1
        self.remove_tmp_files = 1


# main
#=======================================================================================================================
def main():

    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        # get path of the testing data
        status, path_sct_data = commands.getstatusoutput('echo $SCT_TESTING_DATA_DIR')
        param.fname_data = path_sct_data+'/mt/mt1.nii.gz'
        param.method = 'point,'+path_sct_data+'/mt/mt1_point.nii.gz' #'centerline,/Users/julien/data/temp/sct_example_data/t2/t2_centerlinerpi.nii.gz'  #coord,68x69'
        param.shape = 'cylinder'
        param.size = 20
        param.remove_tmp_files = 1
        param.verbose = 1
    else:
        # Check input parameters
        try:
            opts, args = getopt.getopt(sys.argv[1:], 'hf:i:m:o:r:s:e:v:')
        except getopt.GetoptError:
            usage()
        if not opts:
            usage()
        for opt, arg in opts:
            if opt == '-h':
                usage()
            elif opt in '-f':
                param.shape = arg
            elif opt in '-i':
                param.fname_data = arg
            elif opt in '-m':
                param.method = arg
            elif opt in '-o':
                param.fname_out = arg
            elif opt in '-r':
                param.remove_tmp_files = int(arg)
            elif opt in '-s':
                param.size = int(arg)
            elif opt in '-e':
                param.even = int(arg)
            elif opt in '-v':
                param.verbose = int(arg)

    # run main program
    create_mask()


# create_mask
#=======================================================================================================================
def create_mask():

    fsloutput = 'export FSLOUTPUTTYPE=NIFTI; '  # for faster processing, all outputs are in NIFTI

    # display usage if a mandatory argument is not provided
    if param.fname_data == '' or param.method == '':
        sct.printv('\nERROR: All mandatory arguments are not provided. See usage (add -h).\n', 1, 'error')

    # parse argument for method
    method_list = param.method.replace(' ', '').split(',')  # remove spaces and parse with comma
    # method_list = param.method.split(',')  # parse with comma
    method_type = method_list[0]

    # check existence of method type
    if not method_type in param.method_list:
        sct.printv('\nERROR in '+os.path.basename(__file__)+': Method "'+method_type+'" is not recognized. See usage (add -h).\n', 1, 'error')

    # check method val
    if not method_type == 'center':
        method_val = method_list[1]
    del method_list

    # check existence of shape
    if not param.shape in param.shape_list:
        sct.printv('\nERROR in '+os.path.basename(__file__)+': Shape "'+param.shape+'" is not recognized. See usage (add -h).\n', 1, 'error')

    # check existence of input files
    sct.printv('\ncheck existence of input files...', param.verbose)
    sct.check_file_exist(param.fname_data, param.verbose)
    if method_type == 'centerline':
        sct.check_file_exist(method_val, param.verbose)

    # check if orientation is RPI
    sct.printv('\nCheck if orientation is RPI...', param.verbose)
    ori = get_orientation(param.fname_data, filename=True)
    if not ori == 'RPI':
        sct.printv('\nERROR in '+os.path.basename(__file__)+': Orientation of input image should be RPI. Use sct_image -setorient to put your image in RPI.\n', 1, 'error')

    # display input parameters
    sct.printv('\nInput parameters:', param.verbose)
    sct.printv('  data ..................'+param.fname_data, param.verbose)
    sct.printv('  method ................'+method_type, param.verbose)

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
        status, output = sct.run('sct_label_utils -i '+fname_point+' -t display-voxel', param.verbose)
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

    cmd = 'sct_label_utils -i line.nii -o line.nii -t add -x '
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
        mask2d = ((xx-xc)**2 + (yy-yc)**2 <= radius**2)*1

    elif shape == 'gaussian':
        sigma = float(radius)
        mask2d = numpy.exp(-(((xx-xc)**2)/(2*(sigma**2)) + ((yy-yc)**2)/(2*(sigma**2))))

    # import matplotlib.pyplot as plt
    # plt.imshow(mask2d)
    # plt.show()

    return mask2d

# Print usage
# ==========================================================================================
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  Create mask along z direction.

USAGE
  """+os.path.basename(__file__)+""" -i <data> -m <method,val> -s <size>

MANDATORY ARGUMENTS
  -i <data>        image to create mask on. Only used to get header. Must be 3D.

OPTIONAL ARGUMENTS
  -m <method,val>  method to generate mask and associated value. Default="""+str(param_default.method)+"""
                     coord: X,Y coordinate of center of mask. E.g.: coord,20x15
                     point: volume that contains a single point. E.g.: point,label.nii.gz
                     center: mask is created at center of FOV. In that case, "val" is not required.
                     centerline: volume that contains centerline. E.g.: centerline,my_centerline.nii
  -s <size>        size in voxel. Odd values are better (for mask to be symmetrical). Default="""+str(param_default.size)+"""
                   If shape=gaussian, size corresponds to "sigma".
  -e {0,1}         0: box size is odd. 1: box size is even.
  -f {box,cylinder,gaussian}  shape of the mask. Default="""+str(param_default.shape)+"""
  -o <output>      name of output mask. Default is "mask_INPUTFILE".
  -r {0,1}         remove temporary files. Default="""+str(param_default.remove_tmp_files)+"""
  -v {0,1}         verbose. Default="""+str(param_default.verbose)+"""
  -h               help. Show this message

EXAMPLE
  """+os.path.basename(__file__)+""" -i dwi_mean.nii -m coord,35x42 -s 20 -f box\n"""

    # exit program
    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    param = Param()
    param_default = Param()
    main()
