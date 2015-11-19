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
from sct_image import get_orientation, set_orientation
from sct_straighten_spinalcord import smooth_centerline
from msct_image import Image
from shutil import move, copyfile

# DEFAULT PARAMETERS
class Param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.verbose = 1  # verbose
        self.step = 1  # step of discretized plane in mm default is min(x_scale,py)
        self.remove_temp_files = 1
        self.smoothing_param = 50  # window size (in mm) for smoothing CSA along z. 0 for no smoothing.
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


# MAIN
# ==========================================================================================
def main():

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

    # Parameters for debug mode
    if param.debug:
        fname_segmentation = '/Users/julien/data/temp/sct_example_data/t2/t2_seg.nii.gz'  #path_sct+'/testing/data/errsm_23/t2/t2_segmentation_PropSeg.nii.gz'
        name_process = 'csa'
        verbose = 1
        remove_temp_files = 0
    else:
        # Check input parameters
        try:
             opts, args = getopt.getopt(sys.argv[1:], 'hi:p:m:l:r:s:t:f:v:z:a:')
        except getopt.GetoptError:
            usage()
        if not opts:
            usage()
        for opt, arg in opts :
            if opt == '-h':
                usage()
            elif opt in ("-i"):
                fname_segmentation = arg
            elif opt in ("-p"):
                name_process = arg
            elif opt in("-m"):
                name_method = arg
            elif opt in('-l'):
                vert_lev = arg
            elif opt in('-r'):
                remove_temp_files = int(arg)
            elif opt in ('-s'):
                smoothing_param = int(arg)
            elif opt in ('-f'):
                figure_fit = int(arg)
            elif opt in ('-t'):
                fname_vertebral_labeling = arg
            elif opt in ('-v'):
                verbose = int(arg)
            elif opt in ('-z'):
                slices = arg
            elif opt in ('-a'):
                param.algo_fitting = str(arg)

    # display usage if a mandatory argument is not provided
    if fname_segmentation == '' or name_process == '':
        usage()

    # display usage if the requested process is not available
    if name_process not in processes:
        usage()

    # display usage if incorrect method
    if name_process == 'csa' and (name_method not in method_CSA):
        usage()
    
    # display usage if no method provided
    if name_process == 'csa' and method_CSA == '':
        usage() 

    # update fields
    param.verbose = verbose

    # check existence of input files
    sct.check_file_exist(fname_segmentation)
    
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
            sct.printv('Output CSA file (averaged): csa_mean.txt', param.verbose, 'info')
        sct.printv('Output CSA file (all slices): '+param.fname_csa+'\n', param.verbose, 'info')

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
    sct.run('sct_image -i segmentation.nii.gz -setorient RPI -o segmentation_RPI.nii.gz')

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
    x_centerline_fit, y_centerline_fit, z_centerline_fit, x_centerline_deriv,y_centerline_deriv,z_centerline_deriv = smooth_centerline('segmentation_RPI.nii.gz', type_window = type_window, window_length = window_length, algo_fitting = algo_fitting, verbose = verbose)

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
    # Xp, Yp = (data_seg[:, :, 0] >= 0).nonzero()  # X and Y range

    # extract centerline and smooth it
    x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline('segmentation_RPI.nii.gz', algo_fitting=algo_fitting, type_window=type_window, window_length=window_length, verbose=verbose)
    z_centerline_scaled = [x*pz for x in z_centerline]

    # Compute CSA
    sct.printv('\nCompute CSA...', verbose)

    # Empty arrays in which CSA for each z slice will be stored
    csa = np.zeros(max_z_index-min_z_index+1)
    # csa = [0.0 for i in xrange(0, max_z_index-min_z_index+1)]

    for iz in xrange(min_z_index, max_z_index+1):

        # compute the vector normal to the plane
        normal = normalize(np.array([x_centerline_deriv[iz-min_z_index], y_centerline_deriv[iz-min_z_index], z_centerline_deriv[iz-min_z_index]]))

        # compute the angle between the normal vector of the plane and the vector z
        angle = np.arccos(np.dot(normal, [0, 0, 1]))

        # compute the number of voxels, assuming the segmentation is coded for partial volume effect between 0 and 1.
        number_voxels = np.sum(data_seg[:, :, iz])

        # compute CSA, by scaling with voxel size (in mm) and adjusting for oblique plane
        csa[iz-min_z_index] = number_voxels * px * py * np.cos(angle)

    if smoothing_param:
        from msct_smooth import smoothing_window
        sct.printv('\nSmooth CSA across slices...', verbose)
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
            sct.printv('\nERROR: Path to template is missing. See usage.\n', 1, 'error')
            sys.exit(2)


        elif vert_levels and fname_vertebral_labeling:

            from sct_extract_metric import get_slices_matching_with_vertebral_levels

            # convert the vertebral labeling file to RPI orientation
            im_vertebral_labeling = set_orientation(Image(fname_vertebral_labeling), 'RPI', fname_out=path_tmp+'vertebral_labeling_RPI.nii')

            # get the slices corresponding to the vertebral levels
            slices, vert_levels_list, warning = get_slices_matching_with_vertebral_levels(data_seg, vert_levels, im_vertebral_labeling.data, 1)

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
        save_metrics([0], [file_data], slices, [mean_CSA], [std_CSA], path_data + 'csa_mean.txt', path_data+file_csa_volume,
                 'weighted-average across slices', '', warning_vert_levels=warning)

    # Remove temporary files
    if remove_temp_files:
        print('\nRemove temporary files...')
        sct.run('rm -rf '+path_tmp, error_exit='warning')

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


# Print usage
# ==========================================================================================
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  This function performs various types of processing from the spinal cord segmentation:

USAGE
  """+os.path.basename(__file__)+"""  -i <segmentation> -p <process>

MANDATORY ARGUMENTS
  -i <segmentation>     spinal cord segmentation (e.g., use sct_segmentation_propagation)
  -p <process>          type of process to be performed:
                        - centerline: extract centerline as binary file.
                        - length: compute length of the segmentation
                        - csa: computes cross-sectional area by counting pixels in each
                          slice and then geometrically adjusting using centerline orientation. Outputs:
                          - csa.txt: text file with z (1st column) and CSA in mm^2 (2nd column),
                          - csa_volume.nii.gz: segmentation where each slice\'s value is equal to the CSA (mm^2).

OPTIONAL ARGUMENTS
  -s <window_smooth>    Window size (in mm) for smoothing CSA. 0 for no smoothing. Default="""+str(param_default.smoothing_param)+"""
  -z <zmin:zmax>        Slice range to compute the CSA across (requires \"-p csa\").
                          Example: 5:23. First slice is 0.
                          You can also select specific slices using commas. Example: 0,2,3,5,12
  -l <lmin:lmax>        Vertebral levels to compute the CSA across (requires \"-p csa\").
                          Example: 2:9 for C2 to T2.
  -t <vertebral_labeling_file>    Path to the vertebral labeling file warped to the space of the input (flag -i).
                                  Should be file \"MNI-Poly-AMU_level.nii.gz\" in the folder \"label/template\" once you
                                  have registered all the template items with sct_warp_template.
                                  Default: './label/template/MNI-Poly-AMU_level.nii.gz'. Only use with flag -l
  -r {0,1}              Remove temporary files. Default="""+str(param_default.remove_temp_files)+"""
  -v {0,1}              Verbose. Default="""+str(param_default.verbose)+"""
  -a {hanning,nurbs}    Algorithm for curve fitting. Default="""+str(param_default.algo_fitting)+"""
  -h                    Help. Show this message

EXAMPLE
  """+os.path.basename(__file__)+""" -i binary_segmentation.nii.gz -p csa\n
  To compute CSA across vertebral levels C2 to C4:
  """+os.path.basename(__file__)+""" -i binary_segmentation.nii.gz -p csa -t label/template -l 2:4\n"""

    # exit program
    sys.exit(2)


# START PROGRAM
# =========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    param_default = Param()
    # call main function
    main()
