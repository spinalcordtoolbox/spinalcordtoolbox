#!/usr/bin/env python
# ==========================================================================================
#
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad, Geoffrey Leveque, Olivier Comtois
#
# License: see the LICENSE.TXT
# ==========================================================================================

from msct_base_classes import BaseScript, Algorithm
import numpy as np
from sct_straighten_spinalcord import smooth_centerline
import sct_convert as conv
from msct_parser import Parser
from nibabel import load, save, Nifti1Image
from sct_process_segmentation import extract_centerline
import os
import commands
import sys
from time import strftime, time
import sct_utils as sct
from numpy import mgrid, zeros, exp, unravel_index, argmax, poly1d, polyval, linalg, max, polyfit, sqrt, abs, savetxt
import glob
from sct_utils import fsloutput
from sct_image import get_orientation_3d, set_orientation
from sct_convert import convert
from msct_image import Image
from sct_image import copy_header, split_data, concat_data
from scipy.ndimage.filters import gaussian_filter


class Param:
   ## The constructor
   def __init__(self):
        self.debug = 0
        self.verbose = 1  # verbose
        self.remove_temp_files = 1
        self.type_window = 'hanning'  # for smooth_centerline @sct_straighten_spinalcord
        self.window_length = 80  # for smooth_centerline @sct_straighten_spinalcord
        self.algo_fitting = 'nurbs'
        self.list_file = []
        self.output_file_name = ''
        self.schedule_file = 'flirtsch/schedule_TxTy.sch'
        self.gap = 4  # default gap between co-registered slices.
        self.gaussian_kernel = 4  # gaussian kernel for creating gaussian mask from center point.
        self.deg_poly = 10  # maximum degree of polynomial function for fitting centerline.
        self.remove_tmp_files = 1  # remove temporary files


def get_centerline_from_point(input_image, point_file, gap=4, gaussian_kernel=4, remove_tmp_files=1):

    # Initialization
    fname_anat = input_image
    fname_point = point_file
    slice_gap = gap
    remove_tmp_files = remove_tmp_files
    gaussian_kernel = gaussian_kernel
    start_time = time()
    verbose = 1

    # get path of the toolbox
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
    path_sct = sct.slash_at_the_end(path_sct, 1)

    # Parameters for debug mode
    if param.debug == 1:
        sct.printv('\n*** WARNING: DEBUG MODE ON ***\n\t\t\tCurrent working directory: '+os.getcwd(), 'warning')
        status, path_sct_testing_data = commands.getstatusoutput('echo $SCT_TESTING_DATA_DIR')
        fname_anat = path_sct_testing_data+'/t2/t2.nii.gz'
        fname_point = path_sct_testing_data+'/t2/t2_centerline_init.nii.gz'
        slice_gap = 5

    # check existence of input files
    sct.check_file_exist(fname_anat)
    sct.check_file_exist(fname_point)

    # extract path/file/extension
    path_anat, file_anat, ext_anat = sct.extract_fname(fname_anat)
    path_point, file_point, ext_point = sct.extract_fname(fname_point)

    # extract path of schedule file
    # TODO: include schedule file in sct
    # TODO: check existence of schedule file
    file_schedule = path_sct + param.schedule_file

    # Get input image orientation
    input_image_orientation = get_orientation_3d(fname_anat, filename=True)

    # Display arguments
    print '\nCheck input arguments...'
    print '  Anatomical image:     '+fname_anat
    print '  Orientation:          '+input_image_orientation
    print '  Point in spinal cord: '+fname_point
    print '  Slice gap:            '+str(slice_gap)
    print '  Gaussian kernel:      '+str(gaussian_kernel)
    print '  Degree of polynomial: '+str(param.deg_poly)

    # create temporary folder
    print('\nCreate temporary folder...')
    path_tmp = 'tmp.'+strftime('%y%m%d%H%M%S')
    sct.create_folder(path_tmp)
    print '\nCopy input data...'
    sct.run('cp '+fname_anat+ ' '+path_tmp+'/tmp.anat'+ext_anat)
    sct.run('cp '+fname_point+ ' '+path_tmp+'/tmp.point'+ext_point)

    # go to temporary folder
    os.chdir(path_tmp)

    # convert to nii
    im_anat = convert('tmp.anat'+ext_anat, 'tmp.anat.nii')
    im_point = convert('tmp.point'+ext_point, 'tmp.point.nii')

    # Reorient input anatomical volume into RL PA IS orientation
    print '\nReorient input volume to RL PA IS orientation...'
    set_orientation(im_anat, 'RPI')
    im_anat.setFileName('tmp.anat_orient.nii')
    # Reorient binary point into RL PA IS orientation
    print '\nReorient binary point into RL PA IS orientation...'
    # sct.run(sct.fsloutput + 'fslswapdim tmp.point RL PA IS tmp.point_orient')
    set_orientation(im_point, 'RPI')
    im_point.setFileName('tmp.point_orient.nii')

    # Get image dimensions
    print '\nGet image dimensions...'
    nx, ny, nz, nt, px, py, pz, pt = Image('tmp.anat_orient.nii').dim
    print '.. matrix size: '+str(nx)+' x '+str(ny)+' x '+str(nz)
    print '.. voxel size:  '+str(px)+'mm x '+str(py)+'mm x '+str(pz)+'mm'

    # Split input volume
    print '\nSplit input volume...'
    im_anat_split_list = split_data(im_anat, 2)
    file_anat_split = []
    for im in im_anat_split_list:
        file_anat_split.append(im.absolutepath)
        im.save()

    im_point_split_list = split_data(im_point, 2)
    file_point_split = []
    for im in im_point_split_list:
        file_point_split.append(im.absolutepath)
        im.save()

    # Extract coordinates of input point
    data_point = Image('tmp.point_orient.nii').data
    x_init, y_init, z_init = unravel_index(data_point.argmax(), data_point.shape)
    sct.printv('Coordinates of input point: ('+str(x_init)+', '+str(y_init)+', '+str(z_init)+')', verbose)

    # Create 2D gaussian mask
    sct.printv('\nCreate gaussian mask from point...', verbose)
    xx, yy = mgrid[:nx, :ny]
    mask2d = zeros((nx, ny))
    radius = round(float(gaussian_kernel+1)/2)  # add 1 because the radius includes the center.
    sigma = float(radius)
    mask2d = exp(-(((xx-x_init)**2)/(2*(sigma**2)) + ((yy-y_init)**2)/(2*(sigma**2))))

    # Save mask to 2d file
    file_mask_split = ['tmp.mask_orient_Z'+str(z).zfill(4) for z in range(0, nz, 1)]
    nii_mask2d = Image('tmp.anat_orient_Z0000.nii')
    nii_mask2d.data = mask2d
    nii_mask2d.setFileName(file_mask_split[z_init]+'.nii')
    nii_mask2d.save()

    # initialize variables
    file_mat = ['tmp.mat_Z'+str(z).zfill(4) for z in range(0, nz, 1)]
    file_mat_inv = ['tmp.mat_inv_Z'+str(z).zfill(4) for z in range(0, nz, 1)]
    file_mat_inv_cumul = ['tmp.mat_inv_cumul_Z'+str(z).zfill(4) for z in range(0, nz, 1)]

    # create identity matrix for initial transformation matrix
    fid = open(file_mat_inv_cumul[z_init], 'w')
    fid.write('%i %i %i %i\n' % (1, 0, 0, 0))
    fid.write('%i %i %i %i\n' % (0, 1, 0, 0))
    fid.write('%i %i %i %i\n' % (0, 0, 1, 0))
    fid.write('%i %i %i %i\n' % (0, 0, 0, 1))
    fid.close()

    # initialize centerline: give value corresponding to initial point
    x_centerline = [x_init]
    y_centerline = [y_init]
    z_centerline = [z_init]
    warning_count = 0

    # go up (1), then down (2) in reference to the binary point
    for iUpDown in range(1, 3):

        if iUpDown == 1:
            # z increases
            slice_gap_signed = slice_gap
        elif iUpDown == 2:
            # z decreases
            slice_gap_signed = -slice_gap
            # reverse centerline (because values will be appended at the end)
            x_centerline.reverse()
            y_centerline.reverse()
            z_centerline.reverse()

        # initialization before looping
        z_dest = z_init  # point given by user
        z_src = z_dest + slice_gap_signed

        # continue looping if 0 <= z < nz
        while 0 <= z_src < nz:

            # print current z:
            print 'z='+str(z_src)+':'

            # estimate transformation
            sct.run(fsloutput+'flirt -in '+file_anat_split[z_src]+' -ref '+file_anat_split[z_dest]+' -schedule ' +
                    file_schedule + ' -verbose 0 -omat ' + file_mat[z_src] +
                    ' -cost normcorr -forcescaling -inweight ' + file_mask_split[z_dest] +
                    ' -refweight '+file_mask_split[z_dest])

            # display transfo
            status, output = sct.run('cat '+file_mat[z_src])
            print output

            # check if transformation is bigger than 1.5x slice_gap
            tx = float(output.split()[3])
            ty = float(output.split()[7])
            norm_txy = linalg.norm([tx, ty], ord=2)
            if norm_txy > 1.5*slice_gap:
                print 'WARNING: Transformation is too large --> using previous one.'
                warning_count = warning_count + 1
                # if previous transformation exists, replace current one with previous one
                if os.path.isfile(file_mat[z_dest]):
                    sct.run('cp '+file_mat[z_dest]+' '+file_mat[z_src])

            # estimate inverse transformation matrix
            sct.run('convert_xfm -omat '+file_mat_inv[z_src]+' -inverse '+file_mat[z_src])

            # compute cumulative transformation
            sct.run('convert_xfm -omat '+file_mat_inv_cumul[z_src]+' -concat '+file_mat_inv[z_src]+' '+file_mat_inv_cumul[z_dest])

            # apply inverse cumulative transformation to initial gaussian mask (to put it in src space)
            sct.run(fsloutput+'flirt -in '+file_mask_split[z_init]+' -ref '+file_mask_split[z_init]+' -applyxfm -init '+file_mat_inv_cumul[z_src]+' -out '+file_mask_split[z_src])

            # open inverse cumulative transformation file and generate centerline
            fid = open(file_mat_inv_cumul[z_src])
            mat = fid.read().split()
            x_centerline.append(x_init + float(mat[3]))
            y_centerline.append(y_init + float(mat[7]))
            z_centerline.append(z_src)
            #z_index = z_index+1

            # define new z_dest (target slice) and new z_src (moving slice)
            z_dest = z_dest + slice_gap_signed
            z_src = z_src + slice_gap_signed


    # Reconstruct centerline
    # ====================================================================================================

    # reverse back centerline (because it's been reversed once, so now all values are in the right order)
    x_centerline.reverse()
    y_centerline.reverse()
    z_centerline.reverse()

    # fit centerline in the Z-X plane using polynomial function
    print '\nFit centerline in the Z-X plane using polynomial function...'
    coeffsx = polyfit(z_centerline, x_centerline, deg=param.deg_poly)
    polyx = poly1d(coeffsx)
    x_centerline_fit = polyval(polyx, z_centerline)
    # calculate RMSE
    rmse = linalg.norm(x_centerline_fit-x_centerline)/sqrt( len(x_centerline) )
    # calculate max absolute error
    max_abs = max(abs(x_centerline_fit-x_centerline))
    print '.. RMSE (in mm): '+str(rmse*px)
    print '.. Maximum absolute error (in mm): '+str(max_abs*px)

    # fit centerline in the Z-Y plane using polynomial function
    print '\nFit centerline in the Z-Y plane using polynomial function...'
    coeffsy = polyfit(z_centerline, y_centerline, deg=param.deg_poly)
    polyy = poly1d(coeffsy)
    y_centerline_fit = polyval(polyy, z_centerline)
    # calculate RMSE
    rmse = linalg.norm(y_centerline_fit-y_centerline)/sqrt( len(y_centerline) )
    # calculate max absolute error
    max_abs = max( abs(y_centerline_fit-y_centerline) )
    print '.. RMSE (in mm): '+str(rmse*py)
    print '.. Maximum absolute error (in mm): '+str(max_abs*py)

    # display
    if param.debug == 1:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(z_centerline,x_centerline,'.',z_centerline,x_centerline_fit,'r')
        plt.legend(['Data','Polynomial Fit'])
        plt.title('Z-X plane polynomial interpolation')
        plt.show()

        plt.figure()
        plt.plot(z_centerline,y_centerline,'.',z_centerline,y_centerline_fit,'r')
        plt.legend(['Data','Polynomial Fit'])
        plt.title('Z-Y plane polynomial interpolation')
        plt.show()

    # generate full range z-values for centerline
    z_centerline_full = [iz for iz in range(0, nz, 1)]

    # calculate X and Y values for the full centerline
    x_centerline_fit_full = polyval(polyx, z_centerline_full)
    y_centerline_fit_full = polyval(polyy, z_centerline_full)

    # Generate fitted transformation matrices and write centerline coordinates in text file
    print '\nGenerate fitted transformation matrices and write centerline coordinates in text file...'
    file_mat_inv_cumul_fit = ['tmp.mat_inv_cumul_fit_z'+str(z).zfill(4) for z in range(0,nz,1)]
    file_mat_cumul_fit = ['tmp.mat_cumul_fit_z'+str(z).zfill(4) for z in range(0,nz,1)]
    fid_centerline = open('tmp.centerline_coordinates.txt', 'w')
    for iz in range(0, nz, 1):
        # compute inverse cumulative fitted transformation matrix
        fid = open(file_mat_inv_cumul_fit[iz], 'w')
        fid.write('%i %i %i %f\n' % (1, 0, 0, x_centerline_fit_full[iz]-x_init))
        fid.write('%i %i %i %f\n' % (0, 1, 0, y_centerline_fit_full[iz]-y_init))
        fid.write('%i %i %i %i\n' % (0, 0, 1, 0))
        fid.write('%i %i %i %i\n' % (0, 0, 0, 1))
        fid.close()
        # compute forward cumulative fitted transformation matrix
        sct.run('convert_xfm -omat '+file_mat_cumul_fit[iz]+' -inverse '+file_mat_inv_cumul_fit[iz])
        # write centerline coordinates in x, y, z format
        fid_centerline.write('%f %f %f\n' %(x_centerline_fit_full[iz], y_centerline_fit_full[iz], z_centerline_full[iz]) )
    fid_centerline.close()


    # Prepare output data
    # ====================================================================================================

    # write centerline as text file
    for iz in range(0, nz, 1):
        # compute inverse cumulative fitted transformation matrix
        fid = open(file_mat_inv_cumul_fit[iz], 'w')
        fid.write('%i %i %i %f\n' % (1, 0, 0, x_centerline_fit_full[iz]-x_init))
        fid.write('%i %i %i %f\n' % (0, 1, 0, y_centerline_fit_full[iz]-y_init))
        fid.write('%i %i %i %i\n' % (0, 0, 1, 0))
        fid.write('%i %i %i %i\n' % (0, 0, 0, 1))
        fid.close()

    # write polynomial coefficients
    savetxt('tmp.centerline_polycoeffs_x.txt',coeffsx)
    savetxt('tmp.centerline_polycoeffs_y.txt',coeffsy)

    # apply transformations to data
    print '\nApply fitted transformation matrices...'
    file_anat_split_fit = ['tmp.anat_orient_fit_z'+str(z).zfill(4) for z in range(0, nz, 1)]
    file_mask_split_fit = ['tmp.mask_orient_fit_z'+str(z).zfill(4) for z in range(0, nz, 1)]
    file_point_split_fit = ['tmp.point_orient_fit_z'+str(z).zfill(4) for z in range(0, nz, 1)]
    for iz in range(0, nz, 1):
        # forward cumulative transformation to data
        sct.run(fsloutput+'flirt -in '+file_anat_split[iz]+' -ref '+file_anat_split[iz]+' -applyxfm -init '+file_mat_cumul_fit[iz]+' -out '+file_anat_split_fit[iz])
        # inverse cumulative transformation to mask
        sct.run(fsloutput+'flirt -in '+file_mask_split[z_init]+' -ref '+file_mask_split[z_init]+' -applyxfm -init '+file_mat_inv_cumul_fit[iz]+' -out '+file_mask_split_fit[iz])
        # inverse cumulative transformation to point
        sct.run(fsloutput+'flirt -in '+file_point_split[z_init]+' -ref '+file_point_split[z_init]+' -applyxfm -init '+file_mat_inv_cumul_fit[iz]+' -out '+file_point_split_fit[iz]+' -interp nearestneighbour')

    # Merge into 4D volume
    print '\nMerge into 4D volume...'
    # im_anat_list = [Image(fname) for fname in glob.glob('tmp.anat_orient_fit_z*.nii')]
    fname_anat_list = glob.glob('tmp.anat_orient_fit_z*.nii')
    im_anat_concat = concat_data(fname_anat_list, 2)
    im_anat_concat.setFileName('tmp.anat_orient_fit.nii')
    im_anat_concat.save()

    # im_mask_list = [Image(fname) for fname in glob.glob('tmp.mask_orient_fit_z*.nii')]
    fname_mask_list = glob.glob('tmp.mask_orient_fit_z*.nii')
    im_mask_concat = concat_data(fname_mask_list, 2)
    im_mask_concat.setFileName('tmp.mask_orient_fit.nii')
    im_mask_concat.save()

    # im_point_list = [Image(fname) for fname in 	glob.glob('tmp.point_orient_fit_z*.nii')]
    fname_point_list = glob.glob('tmp.point_orient_fit_z*.nii')
    im_point_concat = concat_data(fname_point_list, 2)
    im_point_concat.setFileName('tmp.point_orient_fit.nii')
    im_point_concat.save()

    # Copy header geometry from input data
    print '\nCopy header geometry from input data...'
    im_anat = Image('tmp.anat_orient.nii')
    im_anat_orient_fit = Image('tmp.anat_orient_fit.nii')
    im_mask_orient_fit = Image('tmp.mask_orient_fit.nii')
    im_point_orient_fit = Image('tmp.point_orient_fit.nii')
    im_anat_orient_fit = copy_header(im_anat, im_anat_orient_fit)
    im_mask_orient_fit = copy_header(im_anat, im_mask_orient_fit)
    im_point_orient_fit = copy_header(im_anat, im_point_orient_fit)
    for im in [im_anat_orient_fit, im_mask_orient_fit, im_point_orient_fit]:
        im.save()

    # Reorient outputs into the initial orientation of the input image
    print '\nReorient the centerline into the initial orientation of the input image...'
    set_orientation('tmp.point_orient_fit.nii', input_image_orientation, 'tmp.point_orient_fit.nii')
    set_orientation('tmp.mask_orient_fit.nii', input_image_orientation, 'tmp.mask_orient_fit.nii')

    # Generate output file (in current folder)
    print '\nGenerate output file (in current folder)...'
    os.chdir('..')  # come back to parent folder
    fname_output_centerline = sct.generate_output_file(path_tmp+'/tmp.point_orient_fit.nii', file_anat+'_centerline'+ext_anat)

    # Delete temporary files
    if remove_tmp_files == 1:
        print '\nRemove temporary files...'
        sct.run('rm -rf '+path_tmp, error_exit='warning')

    # print number of warnings
    print '\nNumber of warnings: '+str(warning_count)+' (if >10, you should probably reduce the gap and/or increase the kernel size'

    # display elapsed time
    elapsed_time = time() - start_time
    print '\nFinished! \n\tGenerated file: '+fname_output_centerline+'\n\tElapsed time: '+str(int(round(elapsed_time)))+'s\n'


def get_centerline_from_labels(fname_in, list_fname_labels, param, output_file_name=None, remove_temp_files=1, verbose=0):

    path, file, ext = sct.extract_fname(fname_in)

    # create temporary folder
    path_tmp = sct.slash_at_the_end('tmp.'+strftime('%y%m%d%H%M%S'), 1)
    sct.run('mkdir '+path_tmp)

    # Copying input data to tmp folder
    sct.printv('\nCopying input data to tmp folder...', verbose)
    sct.run('sct_convert -i '+fname_in+' -o '+path_tmp+'data.nii')
    file_labels = []
    for i in range(len(list_fname_labels)):
        file_labels.append('labels_'+str(i)+'.nii.gz')
        sct.run('sct_convert -i '+list_fname_labels[i]+' -o '+path_tmp+file_labels[i])

    # go to tmp folder
    os.chdir(path_tmp)

    ## Concatenation of the files

    # Concatenation : sum of matrices
    file_0 = Image('data.nii')
    data_concatenation = file_0.data
    hdr_0 = file_0.hdr
    orientation_file_0 = get_orientation_3d(file_0)
    if len(list_fname_labels) > 0:
       for i in range(0, len(list_fname_labels)):
            orientation_file_temp = get_orientation_3d(file_labels[i], filename=True)
            if orientation_file_0 != orientation_file_temp :
                print 'ERROR: The files ', fname_in, ' and ', file_labels[i], ' are not in the same orientation. Use sct_image -setorient to change the orientation of a file.'
                sys.exit(2)
            file_temp = load(file_labels[i])
            data_temp = file_temp.get_data()
            data_concatenation = data_concatenation + data_temp

    # Save concatenation as a file
    print '\nWrite NIFTI volumes...'
    img = Nifti1Image(data_concatenation, None, hdr_0)
    save(img, 'concatenation_file.nii.gz')

    # Applying nurbs to the concatenation and save file as binary file
    fname_output = extract_centerline('concatenation_file.nii.gz', remove_temp_files = remove_temp_files, verbose = verbose, algo_fitting=param.algo_fitting, type_window=param.type_window, window_length=param.window_length)

    # Rename files after processing
    if output_file_name != None:
       output_file_name = output_file_name
    else : output_file_name = 'generated_centerline.nii.gz'

    os.rename(fname_output, output_file_name)
    path_binary, file_binary, ext_binary = sct.extract_fname(output_file_name)
    os.rename('concatenation_file_centerline.txt', file_binary+'.txt')

    # Process for a binary file as output:
    sct.run('cp '+output_file_name+' ../')

    # Process for a text file as output:
    sct.run('cp '+file_binary+ '.txt'+ ' ../')

    os.chdir('../')
    # Remove temporary files
    if remove_temp_files:
       print('\nRemove temporary files...')
       sct.run('rm -rf '+path_tmp, error_exit='warning')



    # Display results
    # The concatenate centerline and its fitted curve are displayed whithin extract_centerline


def smooth_minimal_path(img, nb_pixels=1):
    """
    Function intended to smooth the minimal path result in the R-L/A-P directions with a gaussian filter
    of a kernel of size nb_pixels
    :param img: Image to be smoothed (is intended to be minimal path image)
    :param nb_pixels: kernel size of the gaussian filter
    :return: returns a smoothed image
    """

    nx, ny, nz, nt, px, py, pz, pt = img.dim
    from scipy.ndimage.filters import gaussian_filter
    raw_orientation = img.change_orientation()

    img.data = gaussian_filter(img.data, [nb_pixels/px, nb_pixels/py, 0])

    img.change_orientation(raw_orientation)
    return img


def symmetry_detector_right_left(data, cropped_xy=0):
    """
    This function
    :param img: input image used for the algorithm
    :param cropped_xy: 1 when we want to crop around the center for the correlation, 0 when not
    :return: returns an image that is the body symmetry (correlation between left and right side of the image)
    """
    from scipy.ndimage.filters import gaussian_filter

    # Change orientation and define variables for
    data = np.squeeze(data)
    dim = data.shape

    img_data = gaussian_filter(data, [0, 5, 5])

    # Cropping around center of image to remove side noise
    if cropped_xy:
        x_mid = np.round(dim[0]/2)
        x_crop_min = int(x_mid - (0.25/2)*dim[0])
        x_crop_max = int(x_mid + (0.25/2)*dim[0])

        img_data[0:x_crop_min,:,:] = 0
        img_data[x_crop_max:-1,:,:] = 0

    # Acquiring a slice and inverted slice for correlation
    slice_p = np.squeeze(np.sum(img_data, 1))
    slice_p_reversed = np.flipud(slice_p)

    # initialise containers for correlation
    m, n = slice_p.shape
    cross_corr = ((2*m)-1, n)
    cross_corr = np.zeros(cross_corr)
    for iz in range(0, np.size(slice_p[1])):
        corr1 = slice_p[:, iz]
        corr2 = slice_p_reversed[:, iz]
        cross_corr[:, iz] = np.double(np.correlate(corr1, corr2, 'full'))
        max_value = np.max(cross_corr[:, iz])
        if max_value == 0:
            cross_corr[:, iz] = 0
        else:
            cross_corr[:, iz] = cross_corr[:, iz]/max_value
    data_out = np.zeros((dim[0], dim[2]))
    index1 = np.round(np.linspace(0,2*m-3, m))
    index2 = np.round(np.linspace(1,2*m-2, m))
    for i in range(0,m):
        indx1 = int(index1[i])
        indx2 = int(index2[i])
        out1 = cross_corr[indx1, :]
        out2 = cross_corr[indx2, :]
        data_out[i, :] = 0.5*(out1 + out2)
    result = np.hstack([data_out[:, np.newaxis, :] for i in range(0, dim[1])])

    return result


def normalize_array_histogram(array):
    """
    Equalizes the data in array
    :param array:
    :return:
    """
    array_min = np.amin(array)
    array -= array_min
    array_max = np.amax(array)
    array /= array_max

    return array


def get_minimum_path(data, smooth_factor=np.sqrt(2), invert=1, verbose=1, debug=0):
    """
    This method returns the minimal path of the image
    :param data: input data of the image
    :param smooth_factor:factor used to smooth the directions that are not up-down
    :param invert: inverts the image data for the algorithm. The algorithm works better if the image data is inverted
    :param verbose:
    :param debug:
    :return:
    """
    [m, n, p] = data.shape
    max_value = np.amax(data)
    if invert:
        data=max_value-data
    J1 = np.ones([m, n, p])*np.inf
    J2 = np.ones([m, n, p])*np.inf
    J1[:, :, 0] = 0
    for row in range(1, p):
        pJ = J1[:, :, row-1]
        cP = np.squeeze(data[1:-2, 1:-2, row])
        VI = np.dstack((cP*smooth_factor, cP*smooth_factor, cP, cP*smooth_factor, cP*smooth_factor))

        Jq = np.dstack((pJ[0:-3, 1:-2], pJ[1:-2, 0:-3], pJ[1:-2, 1:-2], pJ[1:-2, 2:-1], pJ[2:-1, 1:-2]))
        J1[1:-2, 1:-2, row] = np.min(Jq+VI, 2)
        pass

    J2[:, :, p-1] = 0
    for row in range(p-2, -1, -1):
        pJ = J2[:, :, row+1]
        cP = np.squeeze(data[1:-2, 1:-2, row])
        VI = np.dstack((cP*smooth_factor, cP*smooth_factor, cP, cP*smooth_factor, cP*smooth_factor))

        Jq = np.dstack((pJ[0:-3, 1:-2], pJ[1:-2, 0:-3], pJ[1:-2, 1:-2], pJ[1:-2, 2:-1], pJ[2:-1, 1:-2]))
        J2[1:-2, 1:-2, row] = np.min(Jq+VI, 2)
        pass

    result = J1+J2
    if invert:
        percent = np.percentile(result, 50)
        result[result > percent] = percent

        result_min = np.amin(result)
        result_max = np.amax(result)
        result = np.divide(np.subtract(result, result_min), result_max)
        result_max = np.amax(result)

    result = 1-result

    result[result == np.inf] = 0
    result[result == np.nan] = 0

    return result, J1, J2


def get_minimum_path_nii(fname):
    from msct_image import Image
    data=Image(fname)
    vesselness_data = data.data
    raw_orient=data.change_orientation()
    result ,J1, J2 = get_minimum_path(data.data, invert=1)
    data.data = result
    data.change_orientation(raw_orient)
    data.file_name += '_minimalpath'
    data.save()


def ind2sub(array_shape, ind):
    """
    :param array_shape: shape of the array
    :param ind: index number
    :return: coordinates equivalent to the index number for a given array shape
    """
    rows = (ind.astype('int') / array_shape[1])
    cols = (ind.astype('int') % array_shape[1])  # or numpy.mod(ind.astype('int'), array_shape[1])
    return rows, cols


def get_centerline(data, dim):
    """
    This function extracts the highest value per slice from a minimal path image
    and builds the centerline from it
    :param data:
    :param dim:
    :return:
    """
    centerline = np.zeros(dim)

    data[data == np.inf] = 0
    data[data == np.nan] = 0

    for iz in range(0, dim[2]):
        ind = np.argmax(data[:, :, iz])
        X, Y = ind2sub(data[:, :, iz].shape,ind)
        centerline[X,Y,iz] = 1

    return centerline


class SymmetryDetector(Algorithm):
    def __init__(self, input_image, contrast=None, verbose=0, direction='lr', nb_sections=1, crop_xy=1):
        super(SymmetryDetector, self).__init__(input_image)
        self._contrast = contrast
        self._verbose = verbose
        self.direction = direction
        self.nb_sections = nb_sections
        self.crop_xy = crop_xy

    @property
    def contrast(self):
        return self._contrast

    @contrast.setter
    def contrast(self, value):
        if value in ['t1', 't2']:
            self._contrast = value
        else:
            raise Exception('ERROR: contrast value must be t1 or t2')

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        if value in [0, 1]:
            self._verbose = value
        else:
            raise Exception('ERROR: verbose value must be an integer and equal to 0 or 1')

    def execute(self):
        """
        This method executes the symmetry detection
        :return: returns the symmetry data
        """
        img = Image(self.input_image)
        raw_orientation = img.change_orientation()
        data = np.squeeze(img.data)
        dim = data.shape
        section_length = dim[1]/self.nb_sections

        result = np.zeros(dim)

        for i in range(0, self.nb_sections):
            if (i+1)*section_length > dim[1]:
                y_length = (i+1)*section_length - ((i+1)*section_length - dim[1])
                result[:, i*section_length:i*section_length + y_length, :] = symmetry_detector_right_left(data[:, i*section_length:i*section_length + y_length, :],  cropped_xy=self.crop_xy)
            sym = symmetry_detector_right_left(data[:, i*section_length:(i+1)*section_length, :], cropped_xy=self.crop_xy)
            result[:, i*section_length:(i+1)*section_length, :] = sym

        result_image = Image(img)
        if len(result_image.data) == 4:
            result_image.data = result[:,:,:,np.newaxis]
        else:
            result_image.data = result

        result_image.change_orientation(raw_orientation)

        return result_image.data


class SCAD(Algorithm):
    def __init__(self, input_image, contrast=None, verbose=1, rm_tmp_file=0,output_filename=None, debug=0, vesselness_provided=0, minimum_path_exponent=100, enable_symmetry=0, symmetry_exponent=0, spinalcord_radius = 3, smooth_vesselness = 0):
        """
        Constructor for the automatic spinal cord detection
        :param output_filename: Name of the result file of the centerline detection. Must contain the extension (.nii / .nii.gz)
        :param input_image:
        :param contrast:
        :param verbose:
        :param rm_tmp_file:
        :param debug:
        :param produce_output: Produce output debug files,
        :param vesselness_provided: Activate if the vesselness filter image is already provided (to save time),
               the image is expected to be in the same folder as the input image
        :return:
        """
        produce_output = 0
        if verbose == 2:
            produce_output = 1
        super(SCAD, self).__init__(input_image, produce_output=produce_output)
        self._contrast = contrast
        self._verbose = verbose
        self.output_filename = input_image.file_name + '_centerline.nii.gz'
        if output_filename is not None:
            self.output_filename = output_filename
        self.rm_tmp_file = rm_tmp_file
        self.debug = debug
        self.vesselness_provided = vesselness_provided
        self.minimum_path_exponent = minimum_path_exponent
        self.enable_symmetry = enable_symmetry
        self.symmetry_exponent = symmetry_exponent
        self.spinalcord_radius = spinalcord_radius
        self.smooth_vesselness = smooth_vesselness

        # attributes used in the algorithm
        self.raw_orientation = None
        self.raw_symmetry = None
        self.J1_min_path = None
        self.J2_min_path = None
        self.minimum_path_data = None
        self.minimum_path_powered = None
        self.smoothed_min_path = None
        self.spine_detect_data = None
        self.centerline_with_outliers = None

        self.debug_folder = None
        self.path_tmp = None


    @property
    def contrast(self):
        return self._contrast

    @contrast.setter
    def contrast(self, value):
        if value in ['t1', 't2']:
            self._contrast = value
        else:
            raise Exception('ERROR: contrast value must be t1 or t2')

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        if value in [0, 1, 2]:
            self._verbose = value
        else:
            raise Exception('ERROR: verbose value must be an integer and equal to 0 or 1')

    def produce_output_files(self):
        """
        Method used to output all debug files at the same time. To be used after the algorithm is executed

        :return:
        """
        import time
        from sct_utils import slash_at_the_end
        path_tmp = slash_at_the_end('scad_output_'+time.strftime('%y%m%d%H%M%S'), 1)
        sct.run('mkdir '+path_tmp, self.verbose)
        # getting input image header
        img = self.input_image.copy()

        # saving body symmetry
        img.data = self.raw_symmetry
        img.change_orientation(self.raw_orientation)
        img.file_name += 'body_symmetry'
        img.save()

        # saving minimum paths
        img.data = self.minimum_path_data
        img.change_orientation(self.raw_orientation)
        img.file_name = 'min_path'
        img.save()
        img.data = self.J1_min_path
        img.change_orientation(self.raw_orientation)
        img.file_name = 'J1_min_path'
        img.save()
        img.data = self.J2_min_path
        img.change_orientation(self.raw_orientation)
        img.file_name = 'J2_min_path'
        img.save()

        # saving minimum path powered
        img.data = self.minimum_path_powered
        img.change_orientation(self.raw_orientation)
        img.file_name = 'min_path_powered_'+str(self.minimum_path_exponent)
        img.save()

        # saving smoothed min path
        img = self.smoothed_min_path.copy()
        img.change_orientation(self.raw_orientation)
        img.file_name = 'min_path_power_'+str(self.minimum_path_exponent)+'_smoothed'
        img.save()

        # save symmetry_weighted_minimal_path
        img.data = self.spine_detect_data
        img.change_orientation(self.raw_orientation)
        img.file_name = 'symmetry_weighted_minimal_path'
        img.save()

    def output_debug_file(self, img, data, file_name):
        """
        This method writes a nifti file that corresponds to a step in the algorithm for easy debug.
        The new nifti file uses the header from the the image passed as parameter
        :param data: data to be written to file
        :param file_name: filename...
        :return: None
        """
        if self.verbose == 2:
            current_folder = os.getcwd()
            #os.chdir(self.path_tmp)
            try:
                img = Image(img)
                img.data = data
                img.change_orientation(self.raw_orientation)
                img.file_name = file_name
                img.save()
            except Exception, e:
                print e
            #os.chdir(current_folder)

    def setup_debug_folder(self):
        """
        Sets up the folder for the step by step files for this algorithm
        The folder's absolute path can be found in the self.debug_folder property
        :return: None
        """
        if self.produce_output:
            import time
            from sct_utils import slash_at_the_end
            folder = slash_at_the_end('scad_output_'+time.strftime('%y%m%d%H%M%S'), 1)
            sct.run('mkdir '+folder, self.verbose)
            self.debug_folder = os.path.abspath(folder)
            conv.convert(str(self.input_image.absolutepath), str(self.debug_folder)+'/raw.nii.gz')

    def create_temporary_path(self):
        import time
        from sct_utils import slash_at_the_end
        path_tmp = slash_at_the_end('tmp.'+time.strftime('%y%m%d%H%M%S'), 1)
        sct.run('mkdir '+path_tmp, self.verbose)
        return path_tmp

    def execute(self):
        print 'Execution of the SCAD algorithm in '+str(os.getcwd())

        original_name = self.input_image.file_name
        vesselness_file_name = 'imageVesselNessFilter.nii.gz'
        raw_file_name = 'raw.nii'

        # self.setup_debug_folder()

        if self.debug:
            import matplotlib.pyplot as plt # import for debug purposes

        # create tmp and copy input
        self.path_tmp = self.create_temporary_path()
        conv.convert(self.input_image.absolutepath, self.path_tmp+raw_file_name)

        if self.vesselness_provided:
            sct.run('cp '+vesselness_file_name+' '+self.path_tmp+vesselness_file_name)
        os.chdir(self.path_tmp)

        # get input image information
        img = Image(raw_file_name)

        # save original orientation and change image to RPI
        self.raw_orientation = img.change_orientation()

        # get body symmetry
        if self.enable_symmetry:
            from msct_image import change_data_orientation
            sym = SymmetryDetector(raw_file_name, self.contrast, crop_xy=0)
            self.raw_symmetry = sym.execute()
            img.change_orientation(self.raw_orientation)
            self.output_debug_file(img, self.raw_symmetry, 'body_symmetry')
            img.change_orientation()

        if self.smooth_vesselness:
            from msct_image import change_data_orientation
            img.data = gaussian_filter(img.data, [10,10, 1])
            self.output_debug_file(img, img.data, "raw_smooth")
            normalised_symmetry = normalize_array_histogram(self.raw_symmetry)
            # normalized_data = normalize_array_histogram(img.data)
            img.data = np.multiply(img.data, change_data_orientation(normalised_symmetry, self.raw_orientation, "RPI"))
            img.file_name = "symmetry_x_rawsmoothed"
            raw_file_name = img.file_name + img.ext
            img.change_orientation(self.raw_orientation)
            img.save()
            self._contrast = "t1"

        # vesselness filter
        if not self.vesselness_provided:
            sct.run('isct_vesselness -i '+raw_file_name+' -t ' + self._contrast+' -radius '+str(self.spinalcord_radius))

        # load vesselness filter data and perform minimum path on it
        img = Image(vesselness_file_name)
        img.change_orientation()
        self.minimum_path_data, self.J1_min_path, self.J2_min_path = get_minimum_path(img.data, invert=1, debug=1)
        self.output_debug_file(img, self.minimum_path_data, 'minimal_path')
        self.output_debug_file(img, self.J1_min_path, 'J1_minimal_path')
        self.output_debug_file(img, self.J2_min_path, 'J2_minimal_path')

        # Apply an exponent to the minimum path
        self.minimum_path_powered = np.power(self.minimum_path_data, self.minimum_path_exponent)
        self.output_debug_file(img, self.minimum_path_powered, 'minimal_path_power_'+str(self.minimum_path_exponent))

        # Saving in Image since smooth_minimal_path needs pixel dimensions
        img.data = self.minimum_path_powered

        # smooth resulting minimal path
        self.smoothed_min_path = smooth_minimal_path(img)
        self.output_debug_file(img, self.smoothed_min_path.data, 'minimal_path_smooth')

        # normalise symmetry values between 0 and 1
        if self.enable_symmetry:
            normalised_symmetry = normalize_array_histogram(self.raw_symmetry)
            self.output_debug_file(img, self.smoothed_min_path.data, "normalized_symmetry")

        # multiply normalised symmetry data with the minimum path result
            from msct_image import change_data_orientation
            rpi_normalized_sym = change_data_orientation(np.power(normalised_symmetry, self.symmetry_exponent), self.raw_orientation, "RPI")
            self.spine_detect_data = np.multiply(self.smoothed_min_path.data, rpi_normalized_sym)
            self.output_debug_file(img, self.spine_detect_data, "symmetry_x_min_path")
            # extract the centerline from the minimal path image
            self.centerline_with_outliers = get_centerline(self.spine_detect_data, self.spine_detect_data.shape)
        else:
            # extract the centerline from the minimal path image
            self.centerline_with_outliers = get_centerline(self.smoothed_min_path.data, self.smoothed_min_path.data.shape)
        self.output_debug_file(img, self.centerline_with_outliers, 'centerline_with_outliers')

        # saving centerline with outliers to have
        img.data = self.centerline_with_outliers
        img.change_orientation()
        img.file_name = 'centerline_with_outliers'
        img.save()

        # use a b-spline to smooth out the centerline
        x, y, z, dx, dy, dz = smooth_centerline('centerline_with_outliers.nii.gz')

        # save the centerline
        nx, ny, nz, nt, px, py, pz, pt = img.dim
        img.data = np.zeros((nx, ny, nz))
        for i in range(0, np.size(x)):
            img.data[int(x[i]), int(y[i]), int(z[i])] = 1

        self.output_debug_file(img, img.data, 'centerline')
        img.change_orientation(self.raw_orientation)
        img.file_name = 'centerline'
        img.save()

        # copy back centerline
        os.chdir('../')
        conv.convert(self.path_tmp+img.file_name+img.ext, self.output_filename)
        if self.rm_tmp_file == 1:
            import shutil
            shutil.rmtree(self.path_tmp)

        print 'To view the output with FSL :'
        sct.printv('fslview '+self.input_image.absolutepath+' '+self.output_filename+' -l Red &', self.verbose, 'info')


def get_parser():
    """
    :return: Returns the parser with the command line documentation contained in it.
    """
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description("""This program is used to get the centerline of the spinal cord of a subject by using one of the three methods describe in the -method flag .""")
    parser.add_option(name='-i',
                      type_value='image_nifti',
                      description='Image to get centerline from.',
                      mandatory=True,
                      example='t2.nii.gz')
    parser.usage.addSection('Execution Option')
    parser.add_option(name='-p',
                      type_value='multiple_choice',
                      description='Method to get the centerline:\n'
'auto: Uses vesselness filtering + minimal path + body symmetry. Fully automatic.\n'
'point: Uses slice-by-slice registration. Requires point inside the cord. Requires FSL flirt.\n'
'labels: To use if you want to complete an existing centerline/segmentation : fit spline function across labels. Requires a couple of points along the cord. The -i file should be an uncomplete segmentation of SC or centerline, the -l file should be labels where there is a gap in the segentation/centerline.',
                      mandatory=True,
                      example=['auto', 'point', 'labels'])
    parser.add_option(name='-method',
                      type_value='multiple_choice',
                      description='Method to get the centerline:\n'
'auto: Uses vesselness filtering + minimal path + body symmetry. Fully automatic.\n'
'point: Uses slice-by-slice registration. Requires point inside the cord. Requires FSL flirt.\n'
'labels: Fit spline function across labels. Requires a couple of points along the cord.',
                      mandatory=False,
                      deprecated_by='-p',
                      example=['auto', 'point', 'labels'])
    parser.usage.addSection('General options')
    parser.add_option(name='-o',
                      type_value='str',
                      description='Centerline file name (result file name)',
                      mandatory=False,
                      example='out.nii.gz')
    parser.add_option(name='-r',
                      type_value='multiple_choice',
                      description= 'Removes the temporary folder and debug folder used for the algorithm at the end of execution',
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])
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
    parser.usage.addSection('Automatic method options')
    parser.add_option(name='-c',
                      type_value='multiple_choice',
                      description='type of image contrast, t2: cord dark / CSF bright ; t1: cord bright / CSF dark.\n'
                                  'For dMRI use t1, for T2* or MT use t2',
                      mandatory=False,
                      example=['t1', 't2'])
    parser.add_option(name='-contrast',
                      type_value='multiple_choice',
                      description='type of image contrast, t2: cord dark / CSF bright ; t1: cord bright / CSF dark.\n'
                                  'For dMRI use t1, for T2* or MT use t2',
                      mandatory=False,
                      deprecated_by='-c',
                      example=['t1', 't2'])
    parser.add_option(name='-t',
                      type_value='multiple_choice',
                      description='type of image contrast, t2: cord dark / CSF bright ; t1: cord bright / CSF dark.\n'
                                  'For dMRI use t1, for T2* or MT use t2',
                      deprecated_by='-c',
                      mandatory=False,
                      example=['t1', 't2'])
    parser.add_option(name="-radius",
                      type_value="int",
                      description="Approximate radius of spinal cord to help the algorithm",
                      mandatory=False,
                      default_value="4",
                      example="4")    
    parser.add_option(name="-smooth_vesselness",
                      type_value="multiple_choice",
                      description="Smoothing of the vesselness image",
                      mandatory=False,
                      default_value="0",
                      example=['0', '1'])
    parser.add_option(name='-sym_exp',
                      type_value='int',
                      description='Weight symmetry value (only use with flag -sym). Minimum weight: 0, maximum weight: 100.',
                      mandatory=False,
                      default_value=10)
    parser.add_option(name='-sym',
                      type_value='multiple_choice',
                      description='Uses right-left symmetry of the image to improve accuracy.',
                      mandatory=False,
                      default_value='0',
                      example=['0', '1'])

    parser.usage.addSection('Point method options')
    parser.add_option(name='-point',
                      type_value='file',
                      description='Binary image with a point inside the spinal cord.',
                      mandatory=False,
                      example='t2_point.nii.gz')
    parser.add_option(name="-g",
                      type_value="int",
                      description="Gap between slices for registration. Higher is faster but less robust.",
                      mandatory=False,
                      default_value=4,
                      example="4")    
    parser.add_option(name='-k',
                      type_value='int',
                      description='Kernel size for gaussian mask. Higher is more robust but less accurate.',
                      mandatory=False,
                      default_value=4,
                      example='4')

    parser.usage.addSection('Label method options')
    parser.add_option(name='-l',
                      type_value=[[','], 'file'],
                      description='Binary image with several points (5 to 10) along the spinal cord.',
                      mandatory=False,
                      example='t2_labels.nii.gz')

    return parser


if __name__ == '__main__':
    param = Param()
    param_default = Param()

    # init default params
    output_file_name = None
    verbose = param_default.verbose
    rm_tmp_files = param_default.remove_temp_files

    # get parser info
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    method = arguments['-p']
    fname_in = arguments['-i']
    if '-o' in arguments:
        output_file_name = arguments['-o']
    if '-v' in arguments:
        verbose = int(arguments['-v'])
    if '-r' in arguments:
        rm_tmp_files = int(arguments['-r'])

    if method == 'labels':
        if '-l' in arguments:
            list_fname_labels = arguments['-l']
        else:
            sct.printv('ERROR: Needs input label (option -l).', 1, 'error')
        get_centerline_from_labels(fname_in, list_fname_labels, param, output_file_name, rm_tmp_files)

    elif method == 'point':
        if '-point' in arguments:
            fname_point = arguments['-point']
        else:
            sct.printv('ERROR: Needs input point (option -point).', 1, 'error')
        if '-g' in arguments:
            gap = arguments['-g']
        if '-k' in arguments:
            gaussian_kernel = arguments['-k']
        get_centerline_from_point(fname_in, fname_point, gap, gaussian_kernel, rm_tmp_files)

    elif method == 'auto':
        try:
            contrast = arguments['-c']
        except Exception, e:
            sct.printv('The method automatic requires a contrast type to be defined', type='error')
        im = Image(fname_in)
        scad = SCAD(im, contrast=contrast)
        if '-o' in arguments:
            scad.output_filename = arguments['-o']
        if '-r' in arguments:
            scad.rm_tmp_file = int(arguments['-r'])
        if '-sym' in arguments:
            scad.enable_symmetry = int(arguments['-sym'])
        if '-sym_exp' in arguments:
            scad.symmetry_exponent = int(arguments['-sym_exp'])
        if '-radius' in arguments:
            scad.spinalcord_radius = int(arguments['-radius'])
        if '-smooth_vesselness' in arguments:
            scad.smooth_vesselness = int(arguments['-smooth_vesselness'])
        if '-v' in arguments:
            scad.verbose = int(arguments['-v'])
        scad.execute()





