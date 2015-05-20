#!/usr/bin/env python
#
# This program takes as input an anatomic image and the centerline or segmentation of its spinal cord (that you can get
# using sct_get_centerline.py or sct_segmentation_propagation) and returns the anatomic image where the spinal
# cord was straightened.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad, Geoffrey Leveque, Julien Touati
# Modified: 2014-09-01
#
# License: see the LICENSE.TXT
#=======================================================================================================================

# TODO: step "Get coordinates of landmarks along straight centerline..." can be made quicker
# TODO: calculate backward transformation from forward instead of estimating it
# TODO: generate cross at both edge (top and bottom) and populate in between --> this will ensure better quality of the warping field.
# TODO: check if there is an overlap of labels, in case of high curvature and high density of cross along z.
# TODO: convert gap definition to mm (more intuitive than voxel)

# 2014-06-06: corrected bug related to small FOV volumes Solution: reduced spline order (3), computation of a lot of point (1000)

# check if needed Python libraries are already installed or not
import os
import getopt
import time
import commands
import sys

from nibabel import load, Nifti1Image, save
from numpy import array, asarray, append, insert, linalg, mean, isnan, sum
from sympy.solvers import solve
from sympy import Symbol
from scipy import ndimage

import sct_utils as sct
from msct_smooth import smoothing_window, evaluate_derivative_3D
from sct_orientation import set_orientation






## Create a structure to pass important user parameters to the main function
class Param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.deg_poly = 10  # maximum degree of polynomial function for fitting centerline.
        self.gapxy = 20  # size of cross in x and y direction for the landmarks
        self.gapz = 15  # gap between landmarks along z voxels
        self.padding = 30  # pad input volume in order to deal with the fact that some landmarks might be outside the FOV due to the curvature of the spinal cord
        self.interpolation_warp = 'spline'
        self.remove_temp_files = 1  # remove temporary files
        self.verbose = 1
        self.algo_fitting = 'hanning'  # 'hanning' or 'nurbs'
        self.type_window = 'hanning'  # !! for more choices, edit msct_smooth. Possibilities: 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
        self.window_length = 50
        self.crop = 1



#=======================================================================================================================
# main
#=======================================================================================================================
def main():
    
    # Initialization
    fname_anat = ''
    fname_centerline = ''
    gapxy = param.gapxy
    gapz = param.gapz
    padding = param.padding
    remove_temp_files = param.remove_temp_files
    verbose = param.verbose
    interpolation_warp = param.interpolation_warp
    algo_fitting = param.algo_fitting
    window_length = param.window_length
    type_window = param.type_window
    crop = param.crop

    # start timer
    start_time = time.time()

    # get path of the toolbox
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
    print path_sct

    # Parameters for debug mode
    if param.debug == 1:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        fname_anat = '/Users/julien/data/temp/sct_example_data/t2/tmp.150401221259/anat_rpi.nii'  #path_sct+'/testing/sct_testing_data/data/t2/t2.nii.gz'
        fname_centerline = '/Users/julien/data/temp/sct_example_data/t2/tmp.150401221259/centerline_rpi.nii'  # path_sct+'/testing/sct_testing_data/data/t2/t2_seg.nii.gz'
        remove_temp_files = 0
        type_window = 'hanning'
        verbose = 2
    else:
        # Check input param
        try:
            opts, args = getopt.getopt(sys.argv[1:],'hi:c:p:r:v:x:a:f:')
        except getopt.GetoptError as err:
            print str(err)
            usage()
        if not opts:
            usage()
        for opt, arg in opts:
            if opt == '-h':
                usage()
            elif opt in ('-i'):
                fname_anat = arg
            elif opt in ('-c'):
                fname_centerline = arg
            elif opt in ('-r'):
                remove_temp_files = int(arg)
            elif opt in ('-p'):
                padding = int(arg)
            elif opt in ('-x'):
                interpolation_warp = str(arg)
            elif opt in ('-a'):
                algo_fitting = str(arg)
            elif opt in ('-f'):
                crop = int(arg)
            elif opt in ('-v'):
                verbose = int(arg)

    # display usage if a mandatory argument is not provided
    if fname_anat == '' or fname_centerline == '':
        usage()

    # check if algorithm for fitting is correct
    if algo_fitting not in ['hanning','nurbs']:
        sct.printv('ERROR: wrong fitting algorithm',1,'warning')
        usage()

    # update field
    param.verbose = verbose

    # check existence of input files
    sct.check_file_exist(fname_anat)
    sct.check_file_exist(fname_centerline)

    # Display arguments
    print '\nCheck input arguments...'
    print '  Input volume ...................... '+fname_anat
    print '  Centerline ........................ '+fname_centerline
    print '  Final interpolation ............... '+interpolation_warp
    print '  Verbose ........................... '+str(verbose)
    print ''


    # Extract path/file/extension
    path_anat, file_anat, ext_anat = sct.extract_fname(fname_anat)
    path_centerline, file_centerline, ext_centerline = sct.extract_fname(fname_centerline)
    
    # create temporary folder
    path_tmp = 'tmp.'+time.strftime("%y%m%d%H%M%S")
    sct.run('mkdir '+path_tmp, verbose)

    # copy files into tmp folder
    sct.run('cp '+fname_anat+' '+path_tmp)
    sct.run('cp '+fname_centerline+' '+path_tmp)

    # go to tmp folder
    os.chdir(path_tmp)

    # Change orientation of the input centerline into RPI
    sct.printv('\nOrient centerline to RPI orientation...', verbose)
    fname_centerline_orient = file_centerline+'_rpi.nii.gz'
    set_orientation(file_centerline+ext_centerline, 'RPI', fname_centerline_orient)

    # Get dimension
    sct.printv('\nGet dimensions...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_centerline_orient)
    sct.printv('.. matrix size: '+str(nx)+' x '+str(ny)+' x '+str(nz), verbose)
    sct.printv('.. voxel size:  '+str(px)+'mm x '+str(py)+'mm x '+str(pz)+'mm', verbose)

    # smooth centerline
    x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline(fname_centerline_orient, algo_fitting=algo_fitting, type_window=type_window, window_length=window_length,verbose=verbose)

    # Get coordinates of landmarks along curved centerline
    #==========================================================================================
    sct.printv('\nGet coordinates of landmarks along curved centerline...', verbose)
    # landmarks are created along the curved centerline every z=gapz. They consist of a "cross" of size gapx and gapy. In voxel space!!!
    
    # find z indices along centerline given a specific gap: iz_curved
    nz_nonz = len(z_centerline)
    nb_landmark = int(round(float(nz_nonz)/gapz))

    if nb_landmark == 0:
        nb_landmark = 1

    if nb_landmark == 1:
        iz_curved = [0]
    else:
        iz_curved = [i*gapz for i in range(0, nb_landmark-1)]

    iz_curved.append(nz_nonz-1)
    #print iz_curved, len(iz_curved)
    n_iz_curved = len(iz_curved)
    #print n_iz_curved

    # landmark_curved initialisation
    landmark_curved = [ [ [ 0 for i in range(0, 3)] for i in range(0, 5) ] for i in iz_curved ]

    ### TODO: THIS PART IS SLOW AND CAN BE MADE FASTER
    ### >>==============================================================================================================
    for index in range(0, n_iz_curved, 1):
        # calculate d (ax+by+cz+d=0)
        # print iz_curved[index]
        a=x_centerline_deriv[iz_curved[index]]
        b=y_centerline_deriv[iz_curved[index]]
        c=z_centerline_deriv[iz_curved[index]]
        x=x_centerline_fit[iz_curved[index]]
        y=y_centerline_fit[iz_curved[index]]
        z=z_centerline[iz_curved[index]]
        d=-(a*x+b*y+c*z)
        #print a,b,c,d,x,y,z
        # set coordinates for landmark at the center of the cross
        landmark_curved[index][0][0], landmark_curved[index][0][1], landmark_curved[index][0][2] = x_centerline_fit[iz_curved[index]], y_centerline_fit[iz_curved[index]], z_centerline[iz_curved[index]]

        # set y coordinate to y_centerline_fit[iz] for elements 1 and 2 of the cross
        for i in range(1, 3):
            landmark_curved[index][i][1] = y_centerline_fit[iz_curved[index]]

        # set x and z coordinates for landmarks +x and -x, forcing de landmark to be in the orthogonal plan and the distance landmark/curve to be gapxy
        x_n = Symbol('x_n')
        landmark_curved[index][2][0], landmark_curved[index][1][0]=solve((x_n-x)**2+((-1/c)*(a*x_n+b*y+d)-z)**2-gapxy**2,x_n)  #x for -x and +x
        landmark_curved[index][1][2] = (-1/c)*(a*landmark_curved[index][1][0]+b*y+d)  # z for +x
        landmark_curved[index][2][2] = (-1/c)*(a*landmark_curved[index][2][0]+b*y+d)  # z for -x

        # set x coordinate to x_centerline_fit[iz] for elements 3 and 4 of the cross
        for i in range(3, 5):
            landmark_curved[index][i][0] = x_centerline_fit[iz_curved[index]]

        # set coordinates for landmarks +y and -y. Here, x coordinate is 0 (already initialized).
        y_n = Symbol('y_n')
        landmark_curved[index][4][1],landmark_curved[index][3][1] = solve((y_n-y)**2+((-1/c)*(a*x+b*y_n+d)-z)**2-gapxy**2,y_n)  #y for -y and +y
        landmark_curved[index][3][2] = (-1/c)*(a*x+b*landmark_curved[index][3][1]+d)  # z for +y
        landmark_curved[index][4][2] = (-1/c)*(a*x+b*landmark_curved[index][4][1]+d)  # z for -y
    ### <<==============================================================================================================

    if verbose == 2:
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot(x_centerline_fit, y_centerline_fit,z_centerline,zdir='z')
        ax.plot([landmark_curved[i][j][0] for i in range(0, n_iz_curved) for j in range(0, 5)], \
              [landmark_curved[i][j][1] for i in range(0, n_iz_curved) for j in range(0, 5)], \
              [landmark_curved[i][j][2] for i in range(0, n_iz_curved) for j in range(0, 5)], '.')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    # Get coordinates of landmarks along straight centerline
    #==========================================================================================
    sct.printv('\nGet coordinates of landmarks along straight centerline...', verbose)
    landmark_straight = [ [ [ 0 for i in range(0,3)] for i in range (0,5) ] for i in iz_curved ] # same structure as landmark_curved
    
    # calculate the z indices corresponding to the Euclidean distance between two consecutive points on the curved centerline (approximation curve --> line)
    # TODO: DO NOT APPROXIMATE CURVE --> LINE
    if nb_landmark == 1:
        iz_straight = [0 for i in range(0, nb_landmark+1)]
    else:
        iz_straight = [0 for i in range(0, nb_landmark)]

    # print iz_straight,len(iz_straight)
    iz_straight[0] = iz_curved[0]
    for index in range(1, n_iz_curved, 1):
        # compute vector between two consecutive points on the curved centerline
        vector_centerline = [x_centerline_fit[iz_curved[index]] - x_centerline_fit[iz_curved[index-1]], \
                             y_centerline_fit[iz_curved[index]] - y_centerline_fit[iz_curved[index-1]], \
                             z_centerline[iz_curved[index]] - z_centerline[iz_curved[index-1]] ]
        # compute norm of this vector
        norm_vector_centerline = linalg.norm(vector_centerline, ord=2)
        # round to closest integer value
        norm_vector_centerline_rounded = int(round(norm_vector_centerline, 0))
        # assign this value to the current z-coordinate on the straight centerline
        iz_straight[index] = iz_straight[index-1] + norm_vector_centerline_rounded
    
    # initialize x0 and y0 to be at the center of the FOV
    x0 = int(round(nx/2))
    y0 = int(round(ny/2))
    for index in range(0, n_iz_curved, 1):
        # set coordinates for landmark at the center of the cross
        landmark_straight[index][0][0], landmark_straight[index][0][1], landmark_straight[index][0][2] = x0, y0, iz_straight[index]
        # set x, y and z coordinates for landmarks +x
        landmark_straight[index][1][0], landmark_straight[index][1][1], landmark_straight[index][1][2] = x0 + gapxy, y0, iz_straight[index]
        # set x, y and z coordinates for landmarks -x
        landmark_straight[index][2][0], landmark_straight[index][2][1], landmark_straight[index][2][2] = x0-gapxy, y0, iz_straight[index]
        # set x, y and z coordinates for landmarks +y
        landmark_straight[index][3][0], landmark_straight[index][3][1], landmark_straight[index][3][2] = x0, y0+gapxy, iz_straight[index]
        # set x, y and z coordinates for landmarks -y
        landmark_straight[index][4][0], landmark_straight[index][4][1], landmark_straight[index][4][2] = x0, y0-gapxy, iz_straight[index]

    # # display
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # #ax.plot(x_centerline_fit, y_centerline_fit,z_centerline, 'r')
    # ax.plot([landmark_straight[i][j][0] for i in range(0, n_iz_curved) for j in range(0, 5)], \
    #        [landmark_straight[i][j][1] for i in range(0, n_iz_curved) for j in range(0, 5)], \
    #        [landmark_straight[i][j][2] for i in range(0, n_iz_curved) for j in range(0, 5)], '.')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # plt.show()
    #
    
    # Create NIFTI volumes with landmarks
    #==========================================================================================
    # Pad input volume to deal with the fact that some landmarks on the curved centerline might be outside the FOV
    # N.B. IT IS VERY IMPORTANT TO PAD ALSO ALONG X and Y, OTHERWISE SOME LANDMARKS MIGHT GET OUT OF THE FOV!!!
    #sct.run('fslview ' + fname_centerline_orient)
    sct.printv('\nPad input volume to account for landmarks that fall outside the FOV...', verbose)
    sct.run('isct_c3d '+fname_centerline_orient+' -pad '+str(padding)+'x'+str(padding)+'x'+str(padding)+'vox '+str(padding)+'x'+str(padding)+'x'+str(padding)+'vox 0 -o tmp.centerline_pad.nii.gz')
    
    # Open padded centerline for reading
    sct.printv('\nOpen padded centerline for reading...', verbose)
    file = load('tmp.centerline_pad.nii.gz')
    data = file.get_data()
    hdr = file.get_header()
    
    # Create volumes containing curved and straight landmarks
    data_curved_landmarks = data * 0
    data_straight_landmarks = data * 0
    # initialize landmark value
    landmark_value = 1
    # Loop across cross index
    for index in range(0, n_iz_curved, 1):
        # loop across cross element index
        for i_element in range(0, 5, 1):
            # get x, y and z coordinates of curved landmark (rounded to closest integer)
            x, y, z = int(round(landmark_curved[index][i_element][0])), int(round(landmark_curved[index][i_element][1])), int(round(landmark_curved[index][i_element][2]))
            # attribute landmark_value to the voxel and its neighbours
            data_curved_landmarks[x+padding-1:x+padding+2, y+padding-1:y+padding+2, z+padding-1:z+padding+2] = landmark_value
            # get x, y and z coordinates of straight landmark (rounded to closest integer)
            x, y, z = int(round(landmark_straight[index][i_element][0])), int(round(landmark_straight[index][i_element][1])), int(round(landmark_straight[index][i_element][2]))
            # attribute landmark_value to the voxel and its neighbours
            data_straight_landmarks[x+padding-1:x+padding+2, y+padding-1:y+padding+2, z+padding-1:z+padding+2] = landmark_value
            # increment landmark value
            landmark_value = landmark_value + 1

    # Write NIFTI volumes
    sct.printv('\nWrite NIFTI volumes...', verbose)
    hdr.set_data_dtype('uint32')  # set imagetype to uint8 #TODO: maybe use int32
    img = Nifti1Image(data_curved_landmarks, None, hdr)
    save(img, 'tmp.landmarks_curved.nii.gz')
    sct.printv('.. File created: tmp.landmarks_curved.nii.gz', verbose)
    img = Nifti1Image(data_straight_landmarks, None, hdr)
    save(img, 'tmp.landmarks_straight.nii.gz')
    sct.printv('.. File created: tmp.landmarks_straight.nii.gz', verbose)


    # Estimate deformation field by pairing landmarks
    #==========================================================================================
    
    # This stands to avoid overlapping between landmarks
    sct.printv('\nMake sure all labels between landmark_curved and landmark_curved match...', verbose)
    sct.run('sct_label_utils -t remove -i tmp.landmarks_straight.nii.gz -o tmp.landmarks_straight.nii.gz -r tmp.landmarks_curved.nii.gz', verbose)

    # convert landmarks to INT
    sct.printv('\nConvert landmarks to INT...', verbose)
    sct.run('isct_c3d tmp.landmarks_straight.nii.gz -type int -o tmp.landmarks_straight.nii.gz', verbose)
    sct.run('isct_c3d tmp.landmarks_curved.nii.gz -type int -o tmp.landmarks_curved.nii.gz', verbose)

    # Estimate rigid transformation
    sct.printv('\nEstimate rigid transformation between paired landmarks...', verbose)
    sct.run('isct_ANTSUseLandmarkImagesToGetAffineTransform tmp.landmarks_straight.nii.gz tmp.landmarks_curved.nii.gz rigid tmp.curve2straight_rigid.txt', verbose)
    
    # Apply rigid transformation
    sct.printv('\nApply rigid transformation to curved landmarks...', verbose)
    sct.run('sct_apply_transfo -i tmp.landmarks_curved.nii.gz -o tmp.landmarks_curved_rigid.nii.gz -d tmp.landmarks_straight.nii.gz -w tmp.curve2straight_rigid.txt -x nn', verbose)

    # Estimate b-spline transformation curve --> straight
    sct.printv('\nEstimate b-spline transformation: curve --> straight...', verbose)
    sct.run('isct_ANTSUseLandmarkImagesToGetBSplineDisplacementField tmp.landmarks_straight.nii.gz tmp.landmarks_curved_rigid.nii.gz tmp.warp_curve2straight.nii.gz 5x5x10 3 3 0', verbose)

    # remove padding for straight labels
    if crop == 1:
        sct.run('sct_crop_image -i tmp.landmarks_straight.nii.gz -o tmp.landmarks_straight_crop.nii.gz -dim 0 -bzmax', verbose)
        sct.run('sct_crop_image -i tmp.landmarks_straight_crop.nii.gz -o tmp.landmarks_straight_crop.nii.gz -dim 1 -bzmax', verbose)
        sct.run('sct_crop_image -i tmp.landmarks_straight_crop.nii.gz -o tmp.landmarks_straight_crop.nii.gz -dim 2 -bzmax', verbose)
    else:
        sct.run('cp tmp.landmarks_straight.nii.gz tmp.landmarks_straight_crop.nii.gz', verbose)

    # Concatenate rigid and non-linear transformations...
    sct.printv('\nConcatenate rigid and non-linear transformations...', verbose)
    #sct.run('isct_ComposeMultiTransform 3 tmp.warp_rigid.nii -R tmp.landmarks_straight.nii tmp.warp.nii tmp.curve2straight_rigid.txt')
    # !!! DO NOT USE sct.run HERE BECAUSE isct_ComposeMultiTransform OUTPUTS A NON-NULL STATUS !!!
    cmd = 'isct_ComposeMultiTransform 3 tmp.curve2straight.nii.gz -R tmp.landmarks_straight_crop.nii.gz tmp.warp_curve2straight.nii.gz tmp.curve2straight_rigid.txt'
    sct.printv(cmd, verbose, 'code')
    commands.getstatusoutput(cmd)

    # Estimate b-spline transformation straight --> curve
    # TODO: invert warping field instead of estimating a new one
    sct.printv('\nEstimate b-spline transformation: straight --> curve...', verbose)
    sct.run('isct_ANTSUseLandmarkImagesToGetBSplineDisplacementField tmp.landmarks_curved_rigid.nii.gz tmp.landmarks_straight.nii.gz tmp.warp_straight2curve.nii.gz 5x5x10 3 3 0', verbose)
    
    # Concatenate rigid and non-linear transformations...
    sct.printv('\nConcatenate rigid and non-linear transformations...', verbose)
    # cmd = 'isct_ComposeMultiTransform 3 tmp.straight2curve.nii.gz -R tmp.landmarks_straight.nii.gz -i tmp.curve2straight_rigid.txt tmp.warp_straight2curve.nii.gz'
    cmd = 'isct_ComposeMultiTransform 3 tmp.straight2curve.nii.gz -R '+file_anat+ext_anat+' -i tmp.curve2straight_rigid.txt tmp.warp_straight2curve.nii.gz'
    sct.printv(cmd, verbose, 'code')
    commands.getstatusoutput(cmd)

    # Apply transformation to input image
    sct.printv('\nApply transformation to input image...', verbose)
    sct.run('sct_apply_transfo -i '+file_anat+ext_anat+' -o tmp.anat_rigid_warp.nii.gz -d tmp.landmarks_straight_crop.nii.gz -x '+interpolation_warp+' -w tmp.curve2straight.nii.gz', verbose)

    # compute the error between the straightened centerline/segmentation and the central vertical line.
    # Ideally, the error should be zero.
    # Apply deformation to input image
    print '\nApply transformation to input image...'
    c = sct.run('sct_apply_transfo -i '+fname_centerline_orient+' -o tmp.centerline_straight.nii.gz -d tmp.landmarks_straight_crop.nii.gz -x nn -w tmp.curve2straight.nii.gz')
    #c = sct.run('sct_crop_image -i tmp.centerline_straight.nii.gz -o tmp.centerline_straight_crop.nii.gz -dim 2 -bzmax')
    from msct_image import Image
    file_centerline_straight = Image('tmp.centerline_straight.nii.gz')
    coordinates_centerline = file_centerline_straight.getNonZeroCoordinates(sorting='z')
    mean_coord = []
    for z in range(coordinates_centerline[0].z, coordinates_centerline[-1].z):
        mean_coord.append(mean([[coord.x*coord.value, coord.y*coord.value] for coord in coordinates_centerline if coord.z == z], axis=0))

    # compute error between the input data and the nurbs
    from math import sqrt
    mse_curve = 0.0
    max_dist = 0.0
    x0 = int(round(file_centerline_straight.data.shape[0]/2.0))
    y0 = int(round(file_centerline_straight.data.shape[1]/2.0))
    count_mean = 0
    for coord_z in mean_coord:
        if not isnan(sum(coord_z)):
            dist = ((x0-coord_z[0])*px)**2 + ((y0-coord_z[1])*py)**2
            mse_curve += dist
            dist = sqrt(dist)
            if dist > max_dist:
                max_dist = dist
            count_mean += 1
    mse_curve = mse_curve/float(count_mean)

    # come back to parent folder
    os.chdir('..')

    # Generate output file (in current folder)
    # TODO: do not uncompress the warping field, it is too time consuming!
    sct.printv('\nGenerate output file (in current folder)...', verbose)
    sct.generate_output_file(path_tmp+'/tmp.curve2straight.nii.gz', 'warp_curve2straight.nii.gz', verbose)  # warping field
    sct.generate_output_file(path_tmp+'/tmp.straight2curve.nii.gz', 'warp_straight2curve.nii.gz', verbose)  # warping field
    fname_straight = sct.generate_output_file(path_tmp+'/tmp.anat_rigid_warp.nii.gz', file_anat+'_straight'+ext_anat, verbose)  # straightened anatomic

    # Remove temporary files
    if remove_temp_files:
        sct.printv('\nRemove temporary files...', verbose)
        sct.run('rm -rf '+path_tmp, verbose)
    
    print '\nDone!\n'

    sct.printv('Maximum x-y error = '+str(round(max_dist,2))+' mm', verbose, 'bold')
    sct.printv('Accuracy of straightening (MSE) = '+str(round(mse_curve,2))+' mm', verbose, 'bold')
    # display elapsed time
    elapsed_time = time.time() - start_time
    sct.printv('\nFinished! Elapsed time: '+str(int(round(elapsed_time)))+'s', verbose)
    sct.printv('\nTo view results, type:', verbose)
    sct.printv('fslview '+fname_straight+' &\n', verbose, 'info')



#=======================================================================================================================
# usage
#=======================================================================================================================
def usage():
    print '\n' \
        ''+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        '  This function straightens the spinal cord using its centerline (or segmentation).\n' \
        '\n'\
        'USAGE\n' \
        '  '+os.path.basename(__file__)+' -i <data> -c <centerline>\n' \
        '\n'\
        'MANDATORY ARGUMENTS\n' \
        '  -i                input volume.\n' \
        '  -c                centerline or segmentation.\n' \
        '\n'\
        'OPTIONAL ARGUMENTS\n' \
        '  -p <padding>      amount of padding for generating labels. Default='+str(param_default.padding)+'\n' \
        '  -x {nn,linear,spline}  Final interpolation. Default='+str(param_default.interpolation_warp)+'\n' \
        '  -r {0,1}          remove temporary files. Default='+str(param_default.remove_temp_files)+'\n' \
        '  -a {hanning,nurbs}Algorithm for curve fitting. Default='+str(param_default.algo_fitting)+'\n' \
        '  -f {0,1}          Crop option. 0: no crop, 1: crop around landmarks. Default=' + str(param_default.crop) + '\n' \
        '  -v {0,1,2}        Verbose. 0: nothing, 1: basic, 2: extended. Default='+str(param_default.verbose)+'\n' \
        '  -h                help. Show this message.\n' \
        '\n'\
        'EXAMPLE:\n' \
        '  sct_straighten_spinalcord -i t2.nii.gz -c centerline.nii.gz\n'
    sys.exit(2)



# Smooth centerline
#=======================================================================================================================
def smooth_centerline(fname_centerline, algo_fitting = 'hanning', type_window = 'hanning', window_length = 80, verbose = 0):
    """
    :param fname_centerline: centerline in RPI orientation
    :return: a bunch of useful stuff
    """
    # window_length = param.window_length
    # type_window = param.type_window
    # algo_fitting = param.algo_fitting

    sct.printv('\nSmooth centerline/segmentation...', verbose)

    # get dimensions (again!)
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_centerline)

    # open centerline
    file = load(fname_centerline)
    data = file.get_data()

    # loop across z and associate x,y coordinate with the point having maximum intensity
    # N.B. len(z_centerline) = nz_nonz can be smaller than nz in case the centerline is smaller than the input volume
    z_centerline = [iz for iz in range(0, nz, 1) if data[:, :, iz].any()]
    nz_nonz = len(z_centerline)
    x_centerline = [0 for iz in range(0, nz_nonz, 1)]
    y_centerline = [0 for iz in range(0, nz_nonz, 1)]
    x_centerline_deriv = [0 for iz in range(0, nz_nonz, 1)]
    y_centerline_deriv = [0 for iz in range(0, nz_nonz, 1)]
    z_centerline_deriv = [0 for iz in range(0, nz_nonz, 1)]

    # get center of mass of the centerline/segmentation
    sct.printv('.. Get center of mass of the centerline/segmentation...', verbose)
    for iz in range(0, nz_nonz, 1):
        x_centerline[iz], y_centerline[iz] = ndimage.measurements.center_of_mass(array(data[:, :, z_centerline[iz]]))

        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # data_tmp = data
        # data_tmp[x_centerline[iz], y_centerline[iz], z_centerline[iz]] = 10
        # implot = ax.imshow(data_tmp[:, :, z_centerline[iz]].T)
        # implot.set_cmap('gray')
        # plt.show()

    sct.printv('.. Smoothing algo = '+algo_fitting, verbose)
    if algo_fitting == 'hanning':
        # 2D smoothing
        sct.printv('.. Windows length = '+str(window_length), verbose)

        # change to array
        x_centerline = asarray(x_centerline)
        y_centerline = asarray(y_centerline)


        # Smooth the curve
        x_centerline_smooth = smoothing_window(x_centerline, window_len=window_length/pz, window=type_window, verbose = verbose)
        y_centerline_smooth = smoothing_window(y_centerline, window_len=window_length/pz, window=type_window, verbose = verbose)

        # convert to list final result
        x_centerline_smooth = x_centerline_smooth.tolist()
        y_centerline_smooth = y_centerline_smooth.tolist()

        # clear variable
        del data

        x_centerline_fit = x_centerline_smooth
        y_centerline_fit = y_centerline_smooth
        z_centerline_fit = z_centerline

        # get derivative
        x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = evaluate_derivative_3D(x_centerline_fit, y_centerline_fit, z_centerline, px, py, pz)

        x_centerline_fit = asarray(x_centerline_fit)
        y_centerline_fit = asarray(y_centerline_fit)
        z_centerline_fit = asarray(z_centerline_fit)

    elif algo_fitting == 'nurbs':
        from msct_smooth import b_spline_nurbs
        x_centerline_fit, y_centerline_fit, z_centerline_fit, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = b_spline_nurbs(x_centerline, y_centerline, z_centerline, nbControl=None, verbose=verbose)


    else:
        sct.printv('ERROR: wrong algorithm for fitting',1,'error')

    return x_centerline_fit, y_centerline_fit, z_centerline_fit, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv



#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    param_default = Param()
    # call main function
    main()
