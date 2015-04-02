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
import time

import numpy as np
import scipy
import nibabel

import sct_utils as sct
from msct_nurbs import NURBS
from sct_orientation import get_orientation, set_orientation
from sct_straighten_spinalcord import smooth_centerline


# DEFAULT PARAMETERS
class Param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.verbose = 1  # verbose
        self.step = 1 # step of discretized plane in mm default is min(x_scale,py)
        self.remove_temp_files = 1
        self.volume_output = 0
        self.spline_smoothing = 1
        self.smoothing_param = 700
        self.figure_fit = 0
        self.fname_csa = 'csa.txt'  # output name for txt CSA
        self.name_output = 'csa_volume.nii.gz'  # output name for slice CSA
        self.name_method = 'counting_z_plane'  # for compute_CSA
        self.slices = ''
        self.vertebral_levels = ''
        self.path_to_template = ''
        
        
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
    volume_output = param.volume_output
    verbose = param.verbose
    start_time = time.time()
    remove_temp_files = param.remove_temp_files
    spline_smoothing = param.spline_smoothing
    step = param.step
    smoothing_param = param.smoothing_param
    figure_fit = param.figure_fit
    name_output = param.name_output
    slices = param.slices
    vert_lev = param.vertebral_levels
    path_to_template = param.path_to_template
    
    # Parameters for debug mode
    if param.debug:
        fname_segmentation = '/Users/julien/data/temp/sct_example_data/t2/t2_seg.nii.gz'  #path_sct+'/testing/data/errsm_23/t2/t2_segmentation_PropSeg.nii.gz'
        name_process = 'csa'
        verbose = 1
        volume_output = 1
        remove_temp_files = 0
    else:
        # Check input parameters
        try:
             opts, args = getopt.getopt(sys.argv[1:], 'hi:p:m:b:l:r:s:t:f:o:v:z:')
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
            elif opt in('-b'):
                volume_output = int(arg)
            elif opt in('-l'):
                vert_lev = arg
            elif opt in('-r'):
                remove_temp_files = int(arg)
            elif opt in ('-s'):
                spline_smoothing = int(arg)
            elif opt in ('-f'):
                figure_fit = int(arg)
            elif opt in ('-o'):
                name_output = arg
            elif opt in ('-t'):
                path_to_template = arg
            elif opt in ('-v'):
                verbose = int(arg)
                volume_output = 1
            elif opt in ('-z'):
                slices = arg

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

    if name_process == 'extract_centerline':
        fname_output = extract_centerline(fname_segmentation,remove_temp_files)
        # to view results
        sct.printv('\nDone! To view results, type:', param.verbose)
        sct.printv('fslview '+fname_output+' &\n', param.verbose, 'info')

    if name_process == 'csa':
        volume_output = 1
        compute_csa(fname_segmentation, name_method, volume_output, verbose, remove_temp_files, spline_smoothing, step, smoothing_param, figure_fit, name_output, slices, vert_lev, path_to_template)

        sct.printv('\nDone!', param.verbose)
        if (volume_output):
            sct.printv('Output CSA volume: '+name_output, param.verbose, 'info')
        sct.printv('Output CSA file: '+param.fname_csa+'\n', param.verbose, 'info')

    if name_process == 'length':
        result_length = compute_length(fname_segmentation,remove_temp_files)
        sct.printv('\nLength of the Spinal Cord = '+str(round(result_length,2))+' mm\n', verbose, 'info')

    # End of Main



# compute the length of the spinal cord
# ==========================================================================================
def compute_length(fname_segmentation, remove_temp_files):
    from math import sqrt

    # Extract path, file and extension
    fname_segmentation = os.path.abspath(fname_segmentation)
    path_data, file_data, ext_data = sct.extract_fname(fname_segmentation)

    # create temporary folder
    path_tmp = 'tmp.'+time.strftime("%y%m%d%H%M%S")
    sct.run('mkdir '+path_tmp)

    # copy files into tmp folder
    sct.run('cp '+fname_segmentation+' '+path_tmp)

    # go to tmp folder
    os.chdir(path_tmp)

    # Change orientation of the input centerline into RPI
    sct.printv('\nOrient centerline to RPI orientation...', param.verbose)
    fname_segmentation_orient = 'segmentation_rpi' + ext_data
    set_orientation(file_data+ext_data, 'RPI', fname_segmentation_orient)

    # Get dimension
    sct.printv('\nGet dimensions...', param.verbose)
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_segmentation_orient)
    sct.printv('.. matrix size: '+str(nx)+' x '+str(ny)+' x '+str(nz), param.verbose)
    sct.printv('.. voxel size:  '+str(px)+'mm x '+str(py)+'mm x '+str(pz)+'mm', param.verbose)

    # smooth segmentation/centerline
    x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv,y_centerline_deriv,z_centerline_deriv = smooth_centerline(fname_segmentation_orient, param)

    # compute length of centerline
    result_length = 0.0
    for i in range(len(x_centerline_fit)-1):
        result_length += sqrt(((x_centerline_fit[i+1]-x_centerline_fit[i])*px)**2+((y_centerline_fit[i+1]-y_centerline_fit[i])*py)**2+((z_centerline[i+1]-z_centerline[i])*pz)**2)

    return result_length



# extract_centerline
# ==========================================================================================
def extract_centerline(fname_segmentation, remove_temp_files):

    # Extract path, file and extension
    fname_segmentation = os.path.abspath(fname_segmentation)
    path_data, file_data, ext_data = sct.extract_fname(fname_segmentation)

    # create temporary folder
    path_tmp = 'tmp.'+time.strftime("%y%m%d%H%M%S")
    sct.run('mkdir '+path_tmp)

    # copy files into tmp folder
    sct.run('cp '+fname_segmentation+' '+path_tmp)

    # go to tmp folder
    os.chdir(path_tmp)

    # Change orientation of the input centerline into RPI
    sct.printv('\nOrient centerline to RPI orientation...', param.verbose)
    fname_segmentation_orient = 'segmentation_rpi' + ext_data
    set_orientation(file_data+ext_data, 'RPI', fname_segmentation_orient)

    # Get dimension
    sct.printv('\nGet dimensions...', param.verbose)
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_segmentation_orient)
    sct.printv('.. matrix size: '+str(nx)+' x '+str(ny)+' x '+str(nz), param.verbose)
    sct.printv('.. voxel size:  '+str(px)+'mm x '+str(py)+'mm x '+str(pz)+'mm', param.verbose)

    # Extract orientation of the input segmentation
    orientation = get_orientation(file_data+ext_data)
    sct.printv('\nOrientation of segmentation image: ' + orientation, param.verbose)

    sct.printv('\nOpen segmentation volume...', param.verbose)
    file = nibabel.load(fname_segmentation_orient)
    data = file.get_data()
    hdr = file.get_header()

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
    x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv,y_centerline_deriv,z_centerline_deriv = smooth_centerline(fname_segmentation_orient, param)

    # Create an image with the centerline
    for iz in range(min_z_index, max_z_index+1):
        data[round(x_centerline_fit[iz-min_z_index]), round(y_centerline_fit[iz-min_z_index]), iz] = 1
    # Write the centerline image in RPI orientation
    hdr.set_data_dtype('uint8') # set imagetype to uint8
    sct.printv('\nWrite NIFTI volumes...', param.verbose)
    img = nibabel.Nifti1Image(data, None, hdr)
    nibabel.save(img, 'centerline.nii.gz')
    sct.generate_output_file('centerline.nii.gz', file_data+'_centerline'+ext_data, param.verbose)

    # create a txt file with the centerline
    # TODO

    del data

    # come back to parent folder
    os.chdir('..')

    # Change orientation of the output centerline into input orientation
    sct.printv('\nOrient centerline image to input orientation: ' + orientation, param.verbose)
    fname_segmentation_orient = 'tmp.segmentation_rpi' + ext_data
    set_orientation(path_tmp+'/'+file_data+'_centerline'+ext_data, orientation, file_data+'_centerline'+ext_data)

   # Remove temporary files
    if remove_temp_files:
        sct.printv('\nRemove temporary files...', param.verbose)
        sct.run('rm -rf '+path_tmp, param.verbose)

    return file_data+'_centerline'+ext_data


# compute_csa
# ==========================================================================================
def compute_csa(fname_segmentation, name_method, volume_output, verbose, remove_temp_files, spline_smoothing, step, smoothing_param, figure_fit, name_output, slices, vert_levels, path_to_template):

    # Extract path, file and extension
    fname_segmentation = os.path.abspath(fname_segmentation)
    path_data, file_data, ext_data = sct.extract_fname(fname_segmentation)

    # create temporary folder
    sct.printv('\nCreate temporary folder...', verbose)
    path_tmp = sct.slash_at_the_end('tmp.'+time.strftime("%y%m%d%H%M%S"), 1)
    sct.run('mkdir '+path_tmp, verbose)

    # Copying input data to tmp folder and convert to nii
    sct.printv('\nCopying input data to tmp folder and convert to nii...', verbose)
    sct.run('sct_c3d '+fname_segmentation+' -o '+path_tmp+'segmentation.nii')

    # go to tmp folder
    os.chdir(path_tmp)
        
    # Change orientation of the input segmentation into RPI
    sct.printv('\nChange orientation of the input segmentation into RPI...', verbose)
    fname_segmentation_orient = set_orientation('segmentation.nii', 'RPI', 'segmentation_orient.nii')

    # Get size of data
    sct.printv('\nGet data dimensions...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_segmentation_orient)
    sct.printv('  ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz), verbose)

    # Open segmentation volume
    sct.printv('\nOpen segmentation volume...', verbose)
    file_seg = nibabel.load(fname_segmentation_orient)
    data_seg = file_seg.get_data()
    hdr_seg = file_seg.get_header()

    #
    # # Extract min and max index in Z direction
    X, Y, Z = (data_seg > 0).nonzero()
    # coords_seg = np.array([str([X[i], Y[i], Z[i]]) for i in xrange(0,len(Z))])  # don't know why but finding strings in array of array of strings is WAY faster than doing the same with integers
    min_z_index, max_z_index = min(Z), max(Z)
    Xp,Yp = (data_seg[:,:,0]>=0).nonzero() # X and Y range
    #
    # x_centerline = [0 for i in xrange(0,max_z_index-min_z_index+1)]
    # y_centerline = [0 for i in xrange(0,max_z_index-min_z_index+1)]
    # z_centerline = np.array([iz for iz in xrange(min_z_index, max_z_index+1)])
    #
    # # Extract segmentation points and average per slice
    # for iz in xrange(min_z_index, max_z_index+1):
    #     x_seg, y_seg = (data_seg[:,:,iz]>0).nonzero()
    #     x_centerline[iz-min_z_index] = np.mean(x_seg)
    #     y_centerline[iz-min_z_index] = np.mean(y_seg)
    #
    # # Fit the centerline points with spline and return the new fitted coordinates
    # x_centerline_fit, y_centerline_fit,x_centerline_deriv,y_centerline_deriv,z_centerline_deriv = b_spline_centerline(x_centerline,y_centerline,z_centerline)

    # extract centerline and smooth it
    x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv,y_centerline_deriv,z_centerline_deriv = smooth_centerline(fname_segmentation_orient, param)
    z_centerline_scaled = [x*pz for x in z_centerline]

   # # 3D plot of the fit
 #    fig=plt.figure()
 #    ax=Axes3D(fig)
 #    ax.plot(x_centerline,y_centerline,z_centerline,zdir='z')
 #    ax.plot(x_centerline_fit,y_centerline_fit,z_centerline,zdir='z')
 #    plt.show()

    # Defining cartesian basis vectors 
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])
    
    # Creating folder in which JPG files will be stored
    sct.printv('\nCreating folder in which JPG files will be stored...', verbose)
    sct.create_folder('JPG_Results')

    # Compute CSA
    sct.printv('\nCompute CSA...', verbose)

    # Empty arrays in which CSA for each z slice will be stored
    csa = [0 for i in xrange(0,max_z_index-min_z_index+1)]
    # sections_ortho_counting = [0 for i in xrange(0,max_z_index-min_z_index+1)]
    # sections_ortho_ellipse = [0 for i in xrange(0,max_z_index-min_z_index+1)]
    # sections_z_ellipse = [0 for i in xrange(0,max_z_index-min_z_index+1)]
    # sections_z_counting = [0 for i in xrange(0,max_z_index-min_z_index+1)]
    sct.printv('\nCross-Section Area:', verbose, 'bold')

    for iz in xrange(0, len(z_centerline)):

        # Equation of the the plane which is orthogonal to the spline at z=iz
        a = x_centerline_deriv[iz]
        b = y_centerline_deriv[iz]
        c = z_centerline_deriv[iz]

        #vector normal to the plane
        normal = normalize(np.array([a, b, c]))

        # angle between normal vector and z
        angle = np.arccos(np.dot(normal, z))

        if name_method == 'counting_ortho_plane' or name_method == 'ellipse_ortho_plane':

            x_center = x_centerline_fit[iz]
            y_center = y_centerline_fit[iz]
            z_center = z_centerline[iz]

            # use of x in order to get orientation of each plane, basis_1 is in the plane ax+by+cz+d=0
            basis_1 = normalize(np.cross(normal,x))
            basis_2 = normalize(np.cross(normal,basis_1))

            # maximum dimension of the tilted plane. Try multiply numerator by sqrt(2) ?
            max_diameter = (max([(max(X)-min(X))*px,(max(Y)-min(Y))*py]))/(np.cos(angle))

            # Forcing the step to be the min of x and y scale (default value is 1 mm)
            step = min([px,py])

            # discretized plane which will be filled with 0/1
            plane_seg = np.zeros((int(max_diameter/step),int(max_diameter/step)))

            # how the plane will be skimmed through
            plane_grid = np.linspace(-int(max_diameter/2),int(max_diameter/2),int(max_diameter/step))

            # we go through the plane
            for i_b1 in plane_grid :

                for i_b2 in plane_grid :

                    point = np.array([x_center*px,y_center*py,z_center*pz]) + i_b1*basis_1 +i_b2*basis_2

                    # to which voxel belongs each point of the plane
                    coord_voxel = str([ int(point[0]/px), int(point[1]/py), int(point[2]/pz)])

                    if (coord_voxel in coords_seg) is True :  # if this voxel is 1
                        plane_seg[int((plane_grid==i_b1).nonzero()[0])][int((plane_grid==i_b2).nonzero()[0])] = 1

                        # number of voxels that are in the intersection of each plane and the nonzeros values of segmentation, times the area of one cell of the discretized plane
                        if name_method == 'counting_ortho_plane':
                            csa[iz] = len((plane_seg>0).nonzero()[0])*step*step

            # if verbose ==1 and name_method == 'counting_ortho_plane' :

                # print('Cross-Section Area : ' + str(csa[iz]) + ' mm^2')

            if name_method == 'ellipse_ortho_plane':

                # import scipy stuff
                from scipy.misc import imsave

                os.chdir('JPG_Results')
                imsave('plane_ortho_' + str(iz) + '.jpg', plane_seg)

                # Tresholded gradient image
                mag = edge_detection('plane_ortho_' + str(iz) + '.jpg')

                #Coordinates of the contour
                x_contour,y_contour = (mag>0).nonzero()

                x_contour = x_contour*step
                y_contour = y_contour*step

                #Fitting an ellipse
                fit = Ellipse_fit(x_contour,y_contour)

                # Semi-minor axis, semi-major axis
                a_ellipse, b_ellipse = ellipse_dim(fit)

                #Section = pi*a*b
                csa[iz] = a_ellipse*b_ellipse*np.pi

                # if verbose == 1 and name_method == 'ellipse_ortho_plane':
                #     print('Cross-Section Area : ' + str(csa[iz]) + ' mm^2')
                # os.chdir('..')

        if name_method == 'counting_z_plane' or name_method == 'ellipse_z_plane':

            # getting the segmentation for each z plane
            x_seg, y_seg = (data_seg[:,:,iz+min_z_index]>0).nonzero()
            seg = [[x_seg[i],y_seg[i]] for i in range(0,len(x_seg))]

            plane = np.zeros((max(Xp),max(Yp)))

            for i in seg:
                # filling the plane with 0 and 1 regarding to the segmentation
                plane[i[0] - 1][i[1] - 1] = 1

            if name_method == 'counting_z_plane' :
                csa[iz] = len((plane>0).nonzero()[0])*px*py*np.cos(angle)

            # if verbose == 1 and name_method == 'counting_z_plane':
            #     print('Cross-Section Area : ' + str(csa[iz]) + ' mm^2')

            if name_method == 'ellipse_z_plane':

                # import scipy stuff
                from scipy.misc import imsave

                os.chdir('JPG_Results')
                imsave('plane_z_' + str(iz) + '.jpg', plane)

                # Tresholded gradient image
                mag = edge_detection('plane_z_' + str(iz) + '.jpg')

                x_contour,y_contour = (mag>0).nonzero()

                x_contour = x_contour*px
                y_contour = y_contour*py

                # Fitting an ellipse
                fit = Ellipse_fit(x_contour,y_contour)
                a_ellipse, b_ellipse = ellipse_dim(fit)
                csa[iz] = a_ellipse*b_ellipse*np.pi*np.cos(angle)

                 # if verbose == 1 and name_method == 'ellipse_z_plane':
                 #     print('Cross-Section Area : ' + str(csa[iz]) + ' mm^2')

    if spline_smoothing == 1:
        sct.printv('\nSmoothing results with spline...', verbose)
        tck = scipy.interpolate.splrep(z_centerline_scaled, csa, s=smoothing_param)
        csa_smooth = scipy.interpolate.splev(z_centerline_scaled, tck)
        if figure_fit == 1:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(z_centerline_scaled, csa)
            plt.plot(z_centerline_scaled, csa_smooth)
            plt.legend(['CSA values', 'Smoothed values'], 2)
            plt.savefig('Spline_fit.png')
        csa = csa_smooth  # update variable

    # Create output text file
    sct.printv('\nWrite text file...', verbose)
    file_results = open('csa.txt', 'w')
    for i in range(min_z_index, max_z_index+1):
        file_results.write(str(int(i)) + ',' + str(csa[i-min_z_index])+'\n')
        # Display results
        sct.printv('z='+str(i-min_z_index)+': '+str(csa[i-min_z_index])+' mm^2', verbose, 'bold')
    file_results.close()

    # output volume of csa values
    if volume_output:
        sct.printv('\nCreate volume of CSA values...', verbose)
        # get orientation of the input data
        orientation = get_orientation('segmentation.nii')
        data_seg = data_seg.astype(np.float32, copy=False)
        # loop across slices
        for iz in range(min_z_index, max_z_index+1):
            # retrieve seg pixels
            x_seg, y_seg = (data_seg[:, :, iz] > 0).nonzero()
            seg = [[x_seg[i],y_seg[i]] for i in range(0, len(x_seg))]
            # loop across pixels in segmentation
            for i in seg:
                # replace value with csa value
                data_seg[i[0], i[1], iz] = csa[iz-min_z_index]
        # create header
        hdr_seg.set_data_dtype('float32')  # set imagetype to uint8
        # save volume
        img = nibabel.Nifti1Image(data_seg, None, hdr_seg)
        nibabel.save(img, 'csa_RPI.nii')
        # Change orientation of the output centerline into input orientation
        fname_csa_volume = set_orientation('csa_RPI.nii', orientation, 'csa_RPI_orient.nii')

    # come back to parent folder
    os.chdir('..')

    # Generate output files
    sct.printv('\nGenerate output files...', verbose)
    sct.generate_output_file(path_tmp+'csa.txt', path_data+param.fname_csa)  # extension already included in param.fname_csa
    if volume_output:
        sct.generate_output_file(fname_csa_volume, path_data+name_output)  # extension already included in name_output

    # average csa across vertebral levels or slices if asked (flag -z or -l)
    if slices or vert_levels:

        if vert_levels and not path_to_template:
            sct.printv('\nERROR: Path to template is missing. See usage.\n', 1, 'error')
            sys.exit(2)
        elif vert_levels and path_to_template:
            abs_path_to_template = os.path.abspath(path_to_template)

        # go to tmp folder
        os.chdir(path_tmp)

        # create temporary folder
        sct.printv('\nCreate temporary folder to average csa...', verbose)
        path_tmp_extract_metric = sct.slash_at_the_end('label_temp', 1)
        sct.run('mkdir '+path_tmp_extract_metric, verbose)

        # Copying output CSA volume in the temporary folder
        sct.printv('\nCopy data to tmp folder...', verbose)
        sct.run('cp '+fname_segmentation+' '+path_tmp_extract_metric)

        # create file info_label
        path_fname_seg, file_fname_seg, ext_fname_seg = sct.extract_fname(fname_segmentation)
        create_info_label('info_label.txt', path_tmp_extract_metric, file_fname_seg+ext_fname_seg)

        if slices:
            # average CSA
            os.system("sct_extract_metric -i "+path_data+name_output+" -f "+path_tmp_extract_metric+" -m wa -o "+sct.slash_at_the_end(path_data)+"mean_csa -z "+slices)
        if vert_levels:
            sct.run('cp -R '+abs_path_to_template+' .')
            # average CSA
            os.system("sct_extract_metric -i "+path_data+name_output+" -f "+path_tmp_extract_metric+" -m wa -o "+sct.slash_at_the_end(path_data)+"mean_csa -v "+vert_levels)

        os.chdir('..')

        # Remove temporary files
        print('\nRemove temporary folder used to average CSA...')
        sct.run('rm -rf '+path_tmp_extract_metric)

    # Remove temporary files
    if remove_temp_files:
        print('\nRemove temporary files...')
        sct.run('rm -rf '+path_tmp)


#=======================================================================================================================
# create text file info_label.txt
#=======================================================================================================================
def create_info_label(file_name, path_folder, fname_seg):

    os.chdir(path_folder)
    file_info_label = open(file_name, 'w')
    file_info_label.write('# Spinal cord segmentation\n')
    file_info_label.write('# ID, name, file\n')
    file_info_label.write('0, mean CSA, '+fname_seg)
    file_info_label.close()
    os.chdir('..')


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
                        - centerline: extract centerline as binary file
                        - length: compute length of the segmentation
                        - csa: computes cross-sectional area by counting pixels in each
                          slice and then geometrically adjusting using centerline orientation.
                          Output is a text file with z (1st column) and CSA in mm^2 (2nd column) and
                          a volume in which each slice\'s value is equal to the CSA (mm^2).

OPTIONAL ARGUMENTS
  -s {0,1}              Smooth CSA values with spline. Default="""+str(param_default.spline_smoothing)+"""
  -b {0,1}              Outputs a volume in which each slice\'s value is equal to the CSA in
                          mm^2. Default="""+str(param_default.volume_output)+"""
  -o <output_name>      Name of the output volume if -b 1. Default="""+str(param_default.name_output)+"""
  -z <zmin:zmax>        Slice range to compute the CSA across (requires \"-p csa\").
                          Example: 5:23. First slice is 0.
                          You can also select specific slices using commas. Example: 0,2,3,5,12
  -l <lmin:lmax>        Vertebral levels to compute the CSA across (requires \"-p csa\").
                          Example: 2:9 for C2 to T2.
  -t <path_template>    Path to warped template. Typically: ./label/template. Only use with flag -l
  -r {0,1}              Remove temporary files. Default="""+str(param_default.remove_temp_files)+"""
  -v {0,1}              Verbose. Default="""+str(param_default.verbose)+"""
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
