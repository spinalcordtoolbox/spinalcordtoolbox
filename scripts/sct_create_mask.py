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


# DEFAULT PARAMETERS
class param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.fname_data = ''
        self.method = 'coord'  # {coord,point,center,centerline}
        self.shape = 'gaussian'  # box | cylinder | gaussian
        self.size = '40'  # in mm. if gaussian, size corresponds to sigma.
        self.file_suffix = '_mask'  # output suffix
        self.verbose = 1
        self.remove_tmp_files = 1


# main
#=======================================================================================================================
def main(param):

    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        # get path of the testing data
        status, path_sct_data = commands.getstatusoutput('echo $SCT_TESTING_DATA_DIR')
        param.fname_data = path_sct_data+'/mt/mt0.nii.gz'
        param.method = 'coord,20x20'  # coord | point | centerline | center
        param.remove_tmp_files = 0
        param.verbose = 1

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hf:i:m:r:s:v:')
    except getopt.GetoptError:
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
        elif opt in '-r':
            param.remove_tmp_files = int(arg)
        elif opt in '-s':
            param.size = int(arg)
        elif opt in '-v':
            param.verbose = int(arg)

    # run main program
    create_mask(param)


# create_mask
#=======================================================================================================================
def create_mask(param):

    fsloutput = 'export FSLOUTPUTTYPE=NIFTI; '  # for faster processing, all outputs are in NIFTI

    # display usage if a mandatory argument is not provided
    if param.fname_data == '':
        sct.printv('ERROR: All mandatory arguments are not provided. See usage.', 1, 'error')
        usage()

    # parse argument for method
    method_list = param.method.replace(' ', '').split(',')  # remove spaces and parse with comma
    # method_list = param.method.split(',')  # parse with comma
    method_type = method_list[0]
    method_val = method_list[1]
    del method_list

    # check existence of input files
    sct.printv('\ncheck existence of input files...', param.verbose)
    sct.check_file_exist(param.fname_data, param.verbose)

    # display input parameters
    sct.printv('\nInput parameters:', param.verbose)
    sct.printv('  data ..................'+param.fname_data, param.verbose)
    sct.printv('  method ................'+method_type, param.verbose)

    # Extract path/file/extension
    path_data, file_data, ext_data = sct.extract_fname(param.fname_data)
    path_out, file_out, ext_out = '', file_data, ext_data

    # create temporary folder
    sct.printv('\nCreate temporary folder...', param.verbose)
    path_tmp = sct.slash_at_the_end('tmp.'+time.strftime("%y%m%d%H%M%S"), 1)
    sct.run('mkdir '+path_tmp, param.verbose)

    # Copying input data to tmp folder and convert to nii
    # NB: cannot use c3d here because c3d cannot convert 4D data.
    sct.printv('\nCopying input data to tmp folder and convert to nii...', param.verbose)
    sct.run('cp '+param.fname_data+' '+path_tmp+'data'+ext_data, param.verbose)

    # go to tmp folder
    os.chdir(path_tmp)

    # convert to nii format
    sct.run('fslchfiletype NIFTI data', param.verbose)

    # Get dimensions of data
    sct.printv('\nGet dimensions of data...', param.verbose)
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension('data.nii')
    sct.printv('  ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz)+ ' x ' + str(nt), param.verbose)
    # in case user input 4d data
    if nt != 1:
        sct.printv('WARNING '+os.path.basename(__file__)+': Input image is 4d but output mask will 3D.', param.verbose, 'warning')
        # extract first volume to have 3d reference
        sct.run(fsloutput+'fslroi data data -0 1', param.verbose)

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
        # TODO

    if method_type == 'center':
        # set coordinate at center of FOV
        coord = round(float(nx)/2), round(float(ny)/2)

    if method_type == 'centerline':
        # get name of centerline from user argument
        fname_centerline = method_val
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
    cx, cy, cz = numpy.where(data_centerline > 0)
    arg = numpy.argsort(cz)
    cz = cz[arg]
    cx = cx[arg]
    cy = cy[arg]
    file_mask = 'data_mask'
    for iz in range(nz):
        center = numpy.array([cx[iz], cy[iz]])
        mask2d = create_mask2d(center, param.shape, param.size, nx, ny)
        # Write NIFTI volumes
        img = nibabel.Nifti1Image(mask2d, None, hdr)
        nibabel.save(img, (file_mask+str(iz)+'.nii'))
    # merge along Z
    cmd = 'fslmerge -z mask '
    for iz in range(nz):
        cmd = cmd + file_mask+str(iz)+' '
    status, output = sct.run(cmd, param.verbose)
    # copy geometry
    sct.run(fsloutput+'fslcpgeom data mask', param.verbose)
    # sct.run('fslchfiletype NIFTI mask', param.verbose)

    # come back to parent folder
    os.chdir('..')

    # Generate output files
    sct.printv('\nGenerate output files...', param.verbose)
    sct.generate_output_file(path_tmp+'mask.nii.gz', path_out+file_out+param.file_suffix+ext_out)

    # Remove temporary files
    if param.remove_tmp_files == 1:
        print('\nRemove temporary files...')
        sct.run('rm -rf '+path_tmp, param.verbose)

    # to view results
    sct.printv('\nDone! To view results, type:', param.verbose)
    sct.printv('fslview '+path_out+file_out+param.file_suffix+ext_out+' &', param.verbose, 'code')
    print


# create_line
# ==========================================================================================
def create_line(fname, coord, nz):

    # duplicate volume (assumes input file is nifti)
    sct.run('cp '+fname+' line.nii', param.verbose)

    # set all voxels to zero
    sct.run('sct_c3d line.nii -scale 0 -o line.nii', param.verbose)

    # loop across z and create a voxel at a given XY coordinate
    for iz in range(nz):
        sct.run('sct_ImageMath 3 line.nii SetOrGetPixel line.nii 1 '+str(coord[0])+' '+str(coord[1])+' '+str(iz), param.verbose)

    return 'line.nii'


# create_mask2d
# ==========================================================================================
def create_mask2d(center, shape, size, nx, ny):

    # initialize 2d plane
    xx, yy = numpy.mgrid[:nx, :ny]
    mask2d = numpy.zeros((nx, ny))
    # x = numpy.zeros((nx, ny))
    # y = numpy.zeros((nx, ny))
    xc = center[0]
    yc = center[1]
    radius = round(float(size)/2)

    if shape == 'box':
        mask2d[center[0]-radius:center[0]+radius, center[1]-radius:center[1]+radius] = 1

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
  -m <method,val>  method to generate mask and associated value:
                     coord: X,Y coordinate of center of mask. E.g.: coord,20x15
                     point: volume that contains a single point. E.g.: point,label.nii.gz
                     center: mask is created at center of FOV. In that case, "val" is not required.
                     centerline: volume that contains centerline. E.g.: centerline,my_centerline.nii
  -s <size>        size in mm. if shape=gaussian, size corresponds to "sigma".

OPTIONAL ARGUMENTS
  -f {box,cylinder,gaussian}  shape of the mask. Default="""+str(param.shape)+"""
  -r {0,1}         remove temporary files. Default="""+str(param.remove_tmp_files)+"""
  -v {0,1}         verbose. Default="""+str(param.verbose)+"""
  -h               help. Show this message

EXAMPLE
  """+os.path.basename(__file__)+""" -i dwi_mean.nii -m coord,35x42 -s 20 -f box\n"""

    # exit program
    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = param()
    # call main function
    main(param)
