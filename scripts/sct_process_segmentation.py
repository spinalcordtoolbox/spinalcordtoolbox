#!/usr/bin/env python
#########################################################################################
#
# Perform various types of processing from the spinal cord segmentation (e.g. extract centerline, compute CSA, etc.).
# (extract_centerline) extract the spinal cord centerline from the segmentation. Output file is an image in the same
# space as the segmentation.
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener, Julien Touati, Gabriel Mangeat
# Modified: 2014-07-20 by jcohenadad
#
# About the license: see the file LICENSE.TXT
#########################################################################################
# TODO: the import of scipy.misc imsave was moved to the specific cases (orth and ellipse) in order to avoid issue #62. This has to be cleaned in the future.

import sys
import getopt
import os
import commands
from random import randint
import time
import numpy as np
import scipy
import nibabel
import sct_utils as sct
from msct_nurbs import NURBS
from sct_image import get_orientation_3d, set_orientation
from sct_straighten_spinalcord import smooth_centerline
from msct_image import Image
from shutil import move, copyfile
from msct_parser import Parser


# DEFAULT PARAMETERS
class Param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.verbose = 1  # verbose
        self.step = 1  # step of discretized plane in mm default is min(x_scale,py)
        self.remove_temp_files = 1
        self.smoothing_param = 0  # window size (in mm) for smoothing CSA along z. 0 for no smoothing.
        self.figure_fit = 0
        self.fname_csa = 'csa.txt'  # output name for txt CSA
        self.file_csa_volume = 'csa_volume.nii.gz'
        # self.fname_output = 'csa_volume.nii.gz'  # output name for slice CSA
        self.name_method = 'counting_z_plane'  # for compute_CSA
        self.slices = ''
        self.vertebral_levels = ''
        self.type_window = 'hanning'  # for smooth_centerline @sct_straighten_spinalcord
        self.window_length = 50  # for smooth_centerline @sct_straighten_spinalcord
        self.algo_fitting = 'hanning'  # nurbs, hanning
        self.fname_vertebral_labeling = './label/template/MNI-Poly-AMU_level.nii.gz'


def get_parser():
    """
    :return: Returns the parser with the command line documentation contained in it.
    """
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description("""This program is used to get the centerline of the spinal cord of a subject by using one of the three methods describe in the -method flag .""")
    parser.add_option(name='-i',
                      type_value='image_nifti',
                      description='Spinal Cord segmentation',
                      mandatory=True,
                      example='seg.nii.gz')
    parser.add_option(name='-p',
                      type_value='multiple_choice',
                      description='type of process to be performed:\n'
                                  '- centerline: extract centerline as binary file.\n'
                                  '- length: compute length of the segmentation.\n'
                                  '- csa: computes cross-sectional area by counting pixels in each.\n'
                                  '  slice and then geometrically adjusting using centerline orientation. Outputs:\n'
                                  '  - csa.txt: text file with z (1st column) and CSA in mm^2 (2nd column),\n'
                                  '  - csa_volume.nii.gz: segmentation where each slice\'s value is equal to the CSA (mm^2).\n',
                      mandatory=True,
                      example=['centerline', 'length', 'csa'])
    parser.usage.addSection('Optional Arguments')
    parser.add_option(name='-s',
                      type_value=None,
                      description='Window size (in mm) for smoothing CSA. 0 for no smoothing.',
                      mandatory=False,
                      deprecated_by='-size')
    parser.add_option(name='-z',
                      type_value='str',
                      description= 'Slice range to compute the CSA across (requires \"-p csa\").',
                      mandatory=False,
                      example='5:23')
    parser.add_option(name='-l',
                      type_value='str',
                      description= 'Vertebral levels to compute the CSA across (requires \"-p csa\"). Example: 2:9 for C2 to T2.',
                      mandatory=False,
                      deprecated_by='-vert',
                      example='2:9')
    parser.add_option(name='-vert',
                      type_value='str',
                      description= 'Vertebral levels to compute the CSA across (requires \"-p csa\"). Example: 2:9 for C2 to T2.',
                      mandatory=False,
                      example='2:9')
    parser.add_option(name='-t',
                      type_value='image_nifti',
                      description='Vertebral labeling file. Only use with flag -vert',
                      mandatory=False,
                      default_value='label/template/MNI-Poly-AMU_level.nii.gz',
                      example='label/template/MNI-Poly-AMU_level.nii.gz')
    parser.add_option(name='-m',
                      type_value='multiple_choice',
                      description='Method to compute CSA',
                      mandatory=False,
                      default_value='counting_z_plane',
                      deprecated_by='-method',
                      example=['counting_ortho_plane', 'counting_z_plane', 'ellipse_ortho_plane', 'ellipse_z_plane'])
    parser.add_option(name='-method',
                      type_value='multiple_choice',
                      description='Method to compute CSA',
                      mandatory=False,
                      default_value='counting_z_plane',
                      example=['counting_ortho_plane', 'counting_z_plane', 'ellipse_ortho_plane', 'ellipse_z_plane'])
    parser.add_option(name='-r',
                      type_value='multiple_choice',
                      description= 'Removes the temporary folder and debug folder used for the algorithm at the end of execution',
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])
    parser.add_option(name='-size',
                      type_value='int',
                      description='Window size (in mm) for smoothing CSA. 0 for no smoothing.',
                      mandatory=False,
                      default_value=0)
    parser.add_option(name='-a',
                      type_value='multiple_choice',
                      description= 'Algorithm for curve fitting.',
                      mandatory=False,
                      default_value='hanning',
                      example=['hanning', 'nurbs'])
    parser.add_option(name='-v',
                      type_value='multiple_choice',
                      description='1: display on, 0: display off (default)',
                      mandatory=False,
                      example=['0', '1', '2'],
                      default_value='1')
    parser.add_option(name='-h',
                      type_value=None,
                      description='display this help',
                      mandatory=False)

    return parser



# MAIN
# ==========================================================================================
def main(args):

    parser = get_parser()
    arguments = parser.parse(args)

    # Initialization
    path_script = os.path.dirname(__file__)
    fsloutput = 'export FSLOUTPUTTYPE=NIFTI; ' # for faster processing, all outputs are in NIFTI
    # THIS DOES NOT WORK IN MY LAPTOP: path_sct = os.environ['SCT_DIR'] # path to spinal cord toolbox
    #path_sct = path_script[:-8] # TODO: make it cleaner!
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
    fname_segmentation = ''
    name_process = ''
    processes = ['centerline', 'csa', 'length']
    method_CSA = ['counting_ortho_plane', 'counting_z_plane', 'ellipse_ortho_plane', 'ellipse_z_plane']
    name_method = param.name_method
    verbose = param.verbose
    start_time = time.time()
    remove_temp_files = param.remove_temp_files
    # spline_smoothing = param.spline_smoothing
    step = param.step
    smoothing_param = param.smoothing_param
    figure_fit = param.figure_fit
    slices = param.slices
    vert_lev = param.vertebral_levels
    fname_vertebral_labeling = param.fname_vertebral_labeling

    if '-i' in arguments:
        fname_segmentation = arguments['-i']
    if '-p' in arguments:
        name_process = arguments['-p']
    if '-method' in arguments:
        name_method = arguments['-method']
    if '-vert' in arguments:
        vert_lev = arguments['-vert']
    if '-r' in arguments:
        remove_temp_files = arguments['-r']
    if '-size' in arguments:
        smoothing_param = arguments['-size']
    if '-t' in arguments:
        fname_vertebral_labeling = arguments['-t']
    if '-v' in arguments:
        verbose = arguments['-v']
    if '-z' in arguments:
        slices = arguments['-z']
    if '-a' in arguments:
        param.algo_fitting = arguments['-a']

    # # Check input parameters
    # try:
    #      opts, args = getopt.getopt(sys.argv[1:], 'hi:p:m:l:r:s:t:f:v:z:a:')
    # except getopt.GetoptError:
    #     usage()
    # for opt, arg in opts :
    #     if opt in ('-f'):
    #         figure_fit = int(arg)

    # display usage if incorrect method
    if name_process == 'csa' and (name_method not in method_CSA):
        sct.printv(parser.usage.generate(error='ERROR: wrong method for CSA process'))

    # display usage if no method provided
    if name_process == 'csa' and method_CSA == '':
        sct.printv(parser.usage.generate(error='ERROR: no method for CSA process'))

    # update fields
    param.verbose = verbose

    # print arguments
    print '\nCheck parameters:'
    print '.. segmentation file:             '+fname_segmentation

    if name_process == 'centerline':
        fname_output = extract_centerline(fname_segmentation, remove_temp_files, verbose=param.verbose, algo_fitting=param.algo_fitting)
        # to view results
        sct.printv('\nDone! To view results, type:', param.verbose)
        sct.printv('fslview '+fname_segmentation+' '+fname_output+' -l Red &\n', param.verbose, 'info')

    if name_process == 'csa':
        compute_csa(fname_segmentation, verbose, remove_temp_files, step, smoothing_param, figure_fit, param.file_csa_volume, slices, vert_lev, fname_vertebral_labeling, algo_fitting = param.algo_fitting, type_window= param.type_window, window_length=param.window_length)

        sct.printv('\nDone!', param.verbose)
        sct.printv('Output CSA volume: '+param.file_csa_volume, param.verbose, 'info')
        if slices or vert_lev:
            sct.printv('Output file of the mean CSA: csa_mean.txt', param.verbose, 'info')
            sct.printv('Output file of the volume between the selected slices: volume.txt', param.verbose, 'info')
        sct.printv('Output file of CSA for each slice: '+param.fname_csa+'\n', param.verbose, 'info')

    if name_process == 'length':
        result_length = compute_length(fname_segmentation, remove_temp_files, verbose=verbose)
        sct.printv('\nLength of the segmentation = '+str(round(result_length,2))+' mm\n', verbose, 'info')

    # End of Main



# compute the length of the spinal cord
# ==========================================================================================
def compute_length(fname_segmentation, remove_temp_files, verbose = 0):
    from math import sqrt

    # Extract path, file and extension
    fname_segmentation = os.path.abspath(fname_segmentation)
    path_data, file_data, ext_data = sct.extract_fname(fname_segmentation)

    # create temporary folder
    sct.printv('\nCreate temporary folder...', verbose)
    path_tmp = sct.slash_at_the_end('tmp.'+time.strftime("%y%m%d%H%M%S") + '_'+str(randint(1, 1000000)), 1)
    sct.run('mkdir '+path_tmp, verbose)

    # copy files into tmp folder
    sct.run('cp '+fname_segmentation+' '+path_tmp)

    # go to tmp folder
    os.chdir(path_tmp)

    # Change orientation of the input centerline into RPI
    sct.printv('\nOrient centerline to RPI orientation...', param.verbose)
    im_seg = Image(file_data+ext_data)
    fname_segmentation_orient = 'segmentation_rpi' + ext_data
    im_seg_orient = set_orientation(im_seg, 'RPI')
    im_seg_orient.setFileName(fname_segmentation_orient)
    im_seg_orient.save()

    # Get dimension
    sct.printv('\nGet dimensions...', param.verbose)
    nx, ny, nz, nt, px, py, pz, pt = im_seg_orient.dim
    sct.printv('.. matrix size: '+str(nx)+' x '+str(ny)+' x '+str(nz), param.verbose)
    sct.printv('.. voxel size:  '+str(px)+'mm x '+str(py)+'mm x '+str(pz)+'mm', param.verbose)

    # smooth segmentation/centerline
    #x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv,y_centerline_deriv,z_centerline_deriv = smooth_centerline(fname_segmentation_orient, param, 'hanning', 1)
    x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv,y_centerline_deriv,z_centerline_deriv = smooth_centerline(fname_segmentation_orient, type_window='hanning', window_length=80, algo_fitting='hanning', verbose = verbose)
    # compute length of centerline
    result_length = 0.0
    for i in range(len(x_centerline_fit)-1):
        result_length += sqrt(((x_centerline_fit[i+1]-x_centerline_fit[i])*px)**2+((y_centerline_fit[i+1]-y_centerline_fit[i])*py)**2+((z_centerline[i+1]-z_centerline[i])*pz)**2)

    return result_length



# extract_centerline
# ==========================================================================================
def extract_centerline(fname_segmentation, remove_temp_files, verbose = 0, algo_fitting = 'hanning', type_window = 'hanning', window_length = 80):

    # Extract path, file and extension
    fname_segmentation = os.path.abspath(fname_segmentation)
    path_data, file_data, ext_data = sct.extract_fname(fname_segmentation)

    # create temporary folder
    sct.printv('\nCreate temporary folder...', verbose)
    path_tmp = sct.slash_at_the_end('tmp.'+time.strftime("%y%m%d%H%M%S") + '_'+str(randint(1, 1000000)), 1)
    sct.run('mkdir '+path_tmp, verbose)

    # Copying input data to tmp folder
    sct.printv('\nCopying data to tmp folder...', verbose)
    sct.run('sct_convert -i '+fname_segmentation+' -o '+path_tmp+'segmentation.nii.gz', verbose)

    # go to tmp folder
    os.chdir(path_tmp)

    # Change orientation of the input centerline into RPI
    sct.printv('\nOrient centerline to RPI orientation...', verbose)
    # fname_segmentation_orient = 'segmentation_RPI.nii.gz'
    # BELOW DOES NOT WORK (JULIEN, 2015-10-17)
    # im_seg = Image(file_data+ext_data)
    # set_orientation(im_seg, 'RPI')
    # im_seg.setFileName(fname_segmentation_orient)
    # im_seg.save()
    sct.run('sct_image -i segmentation.nii.gz -setorient RPI -o segmentation_RPI.nii.gz', verbose)

    # Open segmentation volume
    sct.printv('\nOpen segmentation volume...', verbose)
    im_seg = Image('segmentation_RPI.nii.gz')
    data = im_seg.data

    # Get size of data
    sct.printv('\nGet data dimensions...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = im_seg.dim
    sct.printv('.. matrix size: '+str(nx)+' x '+str(ny)+' x '+str(nz), verbose)
    sct.printv('.. voxel size:  '+str(px)+'mm x '+str(py)+'mm x '+str(pz)+'mm', verbose)

    # # Get dimension
    # sct.printv('\nGet dimensions...', verbose)
    # nx, ny, nz, nt, px, py, pz, pt = im_seg.dim
    #
    # # Extract orientation of the input segmentation
    # orientation = get_orientation(im_seg)
    # sct.printv('\nOrientation of segmentation image: ' + orientation, verbose)
    #
    # sct.printv('\nOpen segmentation volume...', verbose)
    # data = im_seg.data
    # hdr = im_seg.hdr

    # Extract min and max index in Z direction
    X, Y, Z = (data>0).nonzero()
    min_z_index, max_z_index = min(Z), max(Z)
    x_centerline = [0 for i in range(0,max_z_index-min_z_index+1)]
    y_centerline = [0 for i in range(0,max_z_index-min_z_index+1)]
    z_centerline = [iz for iz in range(min_z_index, max_z_index+1)]
    # Extract segmentation points and average per slice
    for iz in range(min_z_index, max_z_index+1):
        x_seg, y_seg = (data[:,:,iz]>0).nonzero()
        x_centerline[iz-min_z_index] = np.mean(x_seg)
        y_centerline[iz-min_z_index] = np.mean(y_seg)
    for k in range(len(X)):
        data[X[k], Y[k], Z[k]] = 0

    # extract centerline and smooth it
    x_centerline_fit, y_centerline_fit, z_centerline_fit, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline('segmentation_RPI.nii.gz', type_window = type_window, window_length = window_length, algo_fitting = algo_fitting, verbose = verbose)

    if verbose == 2:
            import matplotlib.pyplot as plt

            #Creation of a vector x that takes into account the distance between the labels
            nz_nonz = len(z_centerline)
            x_display = [0 for i in range(x_centerline_fit.shape[0])]
            y_display = [0 for i in range(y_centerline_fit.shape[0])]
            for i in range(0, nz_nonz, 1):
                x_display[int(z_centerline[i]-z_centerline[0])] = x_centerline[i]
                y_display[int(z_centerline[i]-z_centerline[0])] = y_centerline[i]

            plt.figure(1)
            plt.subplot(2,1,1)
            plt.plot(z_centerline_fit, x_display, 'ro')
            plt.plot(z_centerline_fit, x_centerline_fit)
            plt.xlabel("Z")
            plt.ylabel("X")
            plt.title("x and x_fit coordinates")

            plt.subplot(2,1,2)
            plt.plot(z_centerline_fit, y_display, 'ro')
            plt.plot(z_centerline_fit, y_centerline_fit)
            plt.xlabel("Z")
            plt.ylabel("Y")
            plt.title("y and y_fit coordinates")
            plt.show()


    # Create an image with the centerline
    for iz in range(min_z_index, max_z_index+1):
        data[round(x_centerline_fit[iz-min_z_index]), round(y_centerline_fit[iz-min_z_index]), iz] = 1 # if index is out of bounds here for hanning: either the segmentation has holes or labels have been added to the file
    # Write the centerline image in RPI orientation
    # hdr.set_data_dtype('uint8') # set imagetype to uint8
    sct.printv('\nWrite NIFTI volumes...', verbose)
    im_seg.data = data
    im_seg.setFileName('centerline_RPI.nii.gz')
    im_seg.changeType('uint8')
    im_seg.save()

    sct.printv('\nSet to original orientation...', verbose)
    # get orientation of the input data
    im_seg_original = Image('segmentation.nii.gz')
    orientation = im_seg_original.orientation
    sct.run('sct_image -i centerline_RPI.nii.gz -setorient '+orientation+' -o centerline.nii.gz')

    # create a txt file with the centerline
    name_output_txt = 'centerline.txt'
    sct.printv('\nWrite text file...', verbose)
    file_results = open(name_output_txt, 'w')
    for i in range(min_z_index, max_z_index+1):
        file_results.write(str(int(i)) + ' ' + str(x_centerline_fit[i-min_z_index]) + ' ' + str(y_centerline_fit[i-min_z_index]) + '\n')
    file_results.close()

    # come back to parent folder
    os.chdir('..')

    # Generate output files
    sct.printv('\nGenerate output files...', verbose)
    sct.generate_output_file(path_tmp+'centerline.nii.gz', file_data+'_centerline.nii.gz')
    sct.generate_output_file(path_tmp+'centerline.txt', file_data+'_centerline.txt')

    # Remove temporary files
    if remove_temp_files:
        sct.printv('\nRemove temporary files...', verbose)
        sct.run('rm -rf '+path_tmp, verbose)

    return file_data+'_centerline.nii.gz'


# compute_csa
# ==========================================================================================
def compute_csa(fname_segmentation, verbose, remove_temp_files, step, smoothing_param, figure_fit, file_csa_volume, slices, vert_levels, fname_vertebral_labeling='', algo_fitting = 'hanning', type_window = 'hanning', window_length = 80):

    # Extract path, file and extension
    fname_segmentation = os.path.abspath(fname_segmentation)
    path_data, file_data, ext_data = sct.extract_fname(fname_segmentation)

    # create temporary folder
    sct.printv('\nCreate temporary folder...', verbose)
    path_tmp = sct.slash_at_the_end('tmp.'+time.strftime("%y%m%d%H%M%S") + '_'+str(randint(1, 1000000)), 1)
    sct.run('mkdir '+path_tmp, verbose)

    # Copying input data to tmp folder
    sct.printv('\nCopying input data to tmp folder and convert to nii...', verbose)
    sct.run('sct_convert -i '+fname_segmentation+' -o '+path_tmp+'segmentation.nii.gz', verbose)
    # go to tmp folder
    os.chdir(path_tmp)
    # Change orientation of the input segmentation into RPI
    sct.printv('\nChange orientation to RPI...', verbose)
    sct.run('sct_image -i segmentation.nii.gz -setorient RPI -o segmentation_RPI.nii.gz', verbose)

    # Open segmentation volume
    sct.printv('\nOpen segmentation volume...', verbose)
    im_seg = Image('segmentation_RPI.nii.gz')
    data_seg = im_seg.data
    # hdr_seg = im_seg.hdr

    # Get size of data
    sct.printv('\nGet data dimensions...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = im_seg.dim
    sct.printv('  ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz), verbose)

    # # Extract min and max index in Z direction
    X, Y, Z = (data_seg > 0).nonzero()
    min_z_index, max_z_index = min(Z), max(Z)

    # extract centerline and smooth it
    x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline('segmentation_RPI.nii.gz', algo_fitting=algo_fitting, type_window=type_window, window_length=window_length, verbose=verbose)
    z_centerline_scaled = [x*pz for x in z_centerline]

    # Compute CSA
    sct.printv('\nCompute CSA...', verbose)

    # Empty arrays in which CSA for each z slice will be stored
    csa = np.zeros(max_z_index-min_z_index+1)

    for iz in xrange(min_z_index, max_z_index+1):

        # compute the vector normal to the plane
        normal = normalize(np.array([x_centerline_deriv[iz-min_z_index], y_centerline_deriv[iz-min_z_index], z_centerline_deriv[iz-min_z_index]]))

        # compute the angle between the normal vector of the plane and the vector z
        angle = np.arccos(np.dot(normal, [0, 0, 1]))

        # compute the number of voxels, assuming the segmentation is coded for partial volume effect between 0 and 1.
        number_voxels = np.sum(data_seg[:, :, iz])

        # compute CSA, by scaling with voxel size (in mm) and adjusting for oblique plane
        csa[iz-min_z_index] = number_voxels * px * py * np.cos(angle)

    sct.printv('\nSmooth CSA across slices...', verbose)
    if smoothing_param:
        from msct_smooth import smoothing_window
        sct.printv('.. Hanning window: '+str(smoothing_param)+' mm', verbose)
        csa_smooth = smoothing_window(csa, window_len=smoothing_param/pz, window='hanning', verbose=0)
        # display figure
        if verbose == 2:
            import matplotlib.pyplot as plt
            plt.figure()
            pltx, = plt.plot(z_centerline_scaled, csa, 'bo')
            pltx_fit, = plt.plot(z_centerline_scaled, csa_smooth, 'r', linewidth=2)
            plt.title("Cross-sectional area (CSA)")
            plt.xlabel('z (mm)')
            plt.ylabel('CSA (mm^2)')
            plt.legend([pltx, pltx_fit], ['Raw', 'Smoothed'])
            plt.show()
        # update variable
        csa = csa_smooth
    else:
        sct.printv('.. No smoothing!', verbose)


    # Create output text file
    sct.printv('\nWrite text file...', verbose)
    file_results = open('csa.txt', 'w')
    for i in range(min_z_index, max_z_index+1):
        file_results.write(str(int(i)) + ',' + str(csa[i-min_z_index])+'\n')
        # Display results
        sct.printv('z='+str(i-min_z_index)+': '+str(csa[i-min_z_index])+' mm^2', verbose, 'bold')
    file_results.close()

    # output volume of csa values
    sct.printv('\nCreate volume of CSA values...', verbose)
    data_csa = data_seg.astype(np.float32, copy=False)
    # loop across slices
    for iz in range(min_z_index, max_z_index+1):
        # retrieve seg pixels
        x_seg, y_seg = (data_csa[:, :, iz] > 0).nonzero()
        seg = [[x_seg[i],y_seg[i]] for i in range(0, len(x_seg))]
        # loop across pixels in segmentation
        for i in seg:
            # replace value with csa value
            data_csa[i[0], i[1], iz] = csa[iz-min_z_index]
    # replace data
    im_seg.data = data_csa
    # set original orientation
    # TODO: FIND ANOTHER WAY!!
    # im_seg.change_orientation(orientation) --> DOES NOT WORK!
    # set file name -- use .gz because faster to write
    im_seg.setFileName('csa_volume_RPI.nii.gz')
    im_seg.changeType('float32')
    # save volume
    im_seg.save()

    # get orientation of the input data
    im_seg_original = Image('segmentation.nii.gz')
    orientation = im_seg_original.orientation
    sct.run('sct_image -i csa_volume_RPI.nii.gz -setorient '+orientation+' -o '+file_csa_volume)

    # come back to parent folder
    os.chdir('..')

    # Generate output files
    sct.printv('\nGenerate output files...', verbose)
    copyfile(path_tmp+'csa.txt', path_data+param.fname_csa)
    # sct.generate_output_file(path_tmp+'csa.txt', path_data+param.fname_csa)  # extension already included in param.fname_csa
    sct.generate_output_file(path_tmp+file_csa_volume, path_data+file_csa_volume)  # extension already included in name_output

    # average csa across vertebral levels or slices if asked (flag -z or -l)
    if slices or vert_levels:
        from sct_extract_metric import save_metrics

        warning = ''
        if vert_levels and not fname_vertebral_labeling:
            sct.printv('\nERROR: Vertebral labeling file is missing. See usage.\n', 1, 'error')

        elif vert_levels and fname_vertebral_labeling:

            # from sct_extract_metric import get_slices_matching_with_vertebral_levels
            sct.printv('\tSelected vertebral levels... '+vert_levels)
            # convert the vertebral labeling file to RPI orientation
            im_vertebral_labeling = set_orientation(Image(fname_vertebral_labeling), 'RPI', fname_out=path_tmp+'vertebral_labeling_RPI.nii')

            # get the slices corresponding to the vertebral levels
            # slices, vert_levels_list, warning = get_slices_matching_with_vertebral_levels(data_seg, vert_levels, im_vertebral_labeling.data, 1)
            slices, vert_levels_list, warning = get_slices_matching_with_vertebral_levels_based_centerline(vert_levels, im_vertebral_labeling.data, x_centerline_fit, y_centerline_fit, z_centerline)

        elif not vert_levels:
            vert_levels_list = []

        sct.printv('Average CSA across slices...', type='info')

        # parse the selected slices
        slices_lim = slices.strip().split(':')
        slices_list = range(int(slices_lim[0]), int(slices_lim[1])+1)

        CSA_for_selected_slices = []
        # Read the file csa.txt and get the CSA for the selected slices
        with open(path_data+param.fname_csa) as openfile:
            for line in openfile:
                line_split = line.strip().split(',')
                if int(line_split[0]) in slices_list:
                    CSA_for_selected_slices.append(float(line_split[1]))

        # average the CSA
        mean_CSA = np.mean(np.asarray(CSA_for_selected_slices))
        std_CSA = np.std(np.asarray(CSA_for_selected_slices))

        sct.printv('Mean CSA: '+str(mean_CSA)+' +/- '+str(std_CSA)+' mm^2', type='info')

        # write result into output file
        save_metrics([0], [file_data], slices, [mean_CSA], [std_CSA], path_data + 'csa_mean.txt', path_data+file_csa_volume, 'nb_voxels x px x py x cos(theta) slice-by-slice (in mm^3)', '', actual_vert=vert_levels_list, warning_vert_levels=warning)

        # compute volume between the selected slices
        sct.printv('Compute the volume in between the selected slices...', type='info')
        nb_vox = np.sum(data_seg[:, :, slices_list])
        volume = nb_vox*px*py*pz
        sct.printv('Volume in between the selected slices: '+str(volume)+' mm^3', type='info')

        # write result into output file
        save_metrics([0], [file_data], slices, [volume], [np.nan], path_data + 'volume.txt', path_data+file_data, 'nb_voxels x px x py x pz (in mm^3)', '', actual_vert=vert_levels_list, warning_vert_levels=warning)

    # Remove temporary files
    if remove_temp_files:
        sct.printv('\nRemove temporary files...')
        sct.run('rm -rf '+path_tmp, error_exit='warning')


# ======================================================================================================================
# Find min and max slices corresponding to vertebral levels based on the fitted centerline coordinates
# ======================================================================================================================
def get_slices_matching_with_vertebral_levels_based_centerline(vertebral_levels, vertebral_labeling_data, x_centerline_fit, y_centerline_fit, z_centerline):

    # Convert the selected vertebral levels chosen into a 2-element list [start_level end_level]
    vert_levels_list = [int(x) for x in vertebral_levels.split(':')]

    # If only one vertebral level was selected (n), consider as n:n
    if len(vert_levels_list) == 1:
        vert_levels_list = [vert_levels_list[0], vert_levels_list[0]]

    # Check if there are only two values [start_level, end_level] and if the end level is higher than the start level
    if (len(vert_levels_list) > 2) or (vert_levels_list[0] > vert_levels_list[1]):
        sct.printv('\nERROR:  "' + vertebral_levels + '" is not correct. Enter format "1:4". Exit program.\n', type='error')

    # Extract the vertebral levels available in the metric image
    vertebral_levels_available = np.array(list(set(vertebral_labeling_data[vertebral_labeling_data > 0])))

    # Check if the vertebral levels selected are available
    warning=[]  # list of strings gathering the potential following warning(s) to be written in the output .txt file
    min_vert_level_available = min(vertebral_levels_available)  # lowest vertebral level available
    max_vert_level_available = max(vertebral_levels_available)  # highest vertebral level available
    if vert_levels_list[0] < min_vert_level_available:
        vert_levels_list[0] = min_vert_level_available
        warning.append('WARNING: the bottom vertebral level you selected is lower to the lowest level available --> '
                       'Selected the lowest vertebral level available: ' + str(int(vert_levels_list[0])))  # record the
                       # warning to write it later in the .txt output file
        sct.printv('WARNING: the bottom vertebral level you selected is lower to the lowest ' \
                                          'level available \n--> Selected the lowest vertebral level available: '+\
              str(int(vert_levels_list[0])), type='warning')

    if vert_levels_list[1] > max_vert_level_available:
        vert_levels_list[1] = max_vert_level_available
        warning.append('WARNING: the top vertebral level you selected is higher to the highest level available --> '
                       'Selected the highest vertebral level available: ' + str(int(vert_levels_list[1])))  # record the
        # warning to write it later in the .txt output file

        sct.printv('WARNING: the top vertebral level you selected is higher to the highest ' \
                                          'level available \n--> Selected the highest vertebral level available: ' + \
              str(int(vert_levels_list[1])), type='warning')

    if vert_levels_list[0] not in vertebral_levels_available:
        distance = vertebral_levels_available - vert_levels_list[0]  # relative distance
        distance_min_among_negative_value = min(abs(distance[distance < 0]))  # minimal distance among the negative
        # relative distances
        vert_levels_list[0] = vertebral_levels_available[distance == distance_min_among_negative_value]  # element
        # of the initial list corresponding to this minimal distance
        warning.append('WARNING: the bottom vertebral level you selected is not available --> Selected the nearest '
                       'inferior level available: ' + str(int(vert_levels_list[0])))
        sct.printv('WARNING: the bottom vertebral level you selected is not available \n--> Selected the ' \
                             'nearest inferior level available: '+str(int(vert_levels_list[0])), type='warning')  # record the
        # warning to write it later in the .txt output file

    if vert_levels_list[1] not in vertebral_levels_available:
        distance = vertebral_levels_available - vert_levels_list[1]  # relative distance
        distance_min_among_positive_value = min(abs(distance[distance > 0]))  # minimal distance among the negative
        # relative distances
        vert_levels_list[1] = vertebral_levels_available[distance == distance_min_among_positive_value]  # element
        # of the initial list corresponding to this minimal distance
        warning.append('WARNING: the top vertebral level you selected is not available --> Selected the nearest superior'
                       ' level available: ' + str(int(vert_levels_list[1])))  # record the warning to write it later in the .txt output file

        sct.printv('WARNING: the top vertebral level you selected is not available \n--> Selected the ' \
                             'nearest superior level available: ' + str(int(vert_levels_list[1])), type='warning')

    # Find slices included in the vertebral levels wanted by the user
    sct.printv('\tFind slices corresponding to vertebral levels based on the centerline...')
    nz = len(z_centerline)
    matching_slices_centerline_vert_labeling = []
    for i_z in range(0, nz):
        # if the centerline is in the vertebral levels asked by the user, record the slice number
        if vertebral_labeling_data[np.round(x_centerline_fit[i_z]), np.round(y_centerline_fit[i_z]), z_centerline[i_z]] in range(vert_levels_list[0], vert_levels_list[1]+1):
            matching_slices_centerline_vert_labeling.append(z_centerline[i_z])

    # now, find the min and max slices that are included in the vertebral levels
    slices = str(min(matching_slices_centerline_vert_labeling))+':'+str(max(matching_slices_centerline_vert_labeling))
    sct.printv('\t'+slices)

    return slices, vert_levels_list, warning

#=======================================================================================================================
# b_spline_centerline
#=======================================================================================================================
def b_spline_centerline(x_centerline,y_centerline,z_centerline):
                          
    print '\nFitting centerline using B-spline approximation...'
    points = [[x_centerline[n],y_centerline[n],z_centerline[n]] for n in range(len(x_centerline))]
    nurbs = NURBS(3,3000,points)  # BE very careful with the spline order that you choose : if order is too high ( > 4 or 5) you need to set a higher number of Control Points (cf sct_nurbs ). For the third argument (number of points), give at least len(z_centerline)+500 or higher
                          
    P = nurbs.getCourbe3D()
    x_centerline_fit=P[0]
    y_centerline_fit=P[1]
    Q = nurbs.getCourbe3D_deriv()
    x_centerline_deriv=Q[0]
    y_centerline_deriv=Q[1]
    z_centerline_deriv=Q[2]
                          
    return x_centerline_fit, y_centerline_fit,x_centerline_deriv,y_centerline_deriv,z_centerline_deriv


#=======================================================================================================================
# Normalization
#=======================================================================================================================
def normalize(vect):
    norm=np.linalg.norm(vect)
    return vect/norm


#=======================================================================================================================
# Ellipse fitting for a set of data
#=======================================================================================================================
#http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
def Ellipse_fit(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2
    C[1,1] = -1
    E, V =  np.linalg.eig(np.dot(np.linalg.inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a


#=======================================================================================================================
# Getting a and b parameter for fitted ellipse
#=======================================================================================================================
def ellipse_dim(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])


#=======================================================================================================================
# Detect edges of an image
#=======================================================================================================================
def edge_detection(f):

    import Image
    
    #sigma = 1.0
    img = Image.open(f) #grayscale
    imgdata = np.array(img, dtype = float)
    G = imgdata
    #G = ndi.filters.gaussian_filter(imgdata, sigma)
    gradx = np.array(G, dtype = float)
    grady = np.array(G, dtype = float)
 
    mask_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
          
    mask_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
 
    width = img.size[1]
    height = img.size[0]
 
    for i in range(1, width-1):
        for j in range(1, height-1):
        
            px = np.sum(mask_x*G[(i-1):(i+1)+1,(j-1):(j+1)+1])
            py = np.sum(mask_y*G[(i-1):(i+1)+1,(j-1):(j+1)+1])
            gradx[i][j] = px
            grady[i][j] = py

    mag = scipy.hypot(gradx,grady)

    treshold = np.max(mag)*0.9

    for i in range(width):
        for j in range(height):
            if mag[i][j]>treshold:
                mag[i][j]=1
            else:
                mag[i][j] = 0
   
    return mag



# START PROGRAM
# =========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    param_default = Param()
    # call main function
    main(sys.argv[1:])
