#!/usr/bin/env python
#
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 NeuroPoly, Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Tanguy Magnan
#
# License: see the LICENSE.TXT
#=======================================================================================================================
#
import sys, commands
import sct_utils as sct
# Get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# Append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')
from sct_register_multimodal import Paramreg


def register_slicereg2d(fname_source,
                        fname_dest,
                        fname_mask='',
                        window_length='0',
                        warp_forward_out='step0Warp.nii.gz',
                        warp_inverse_out='step0InverseWarp.nii.gz',
                        paramreg=None,
                        ants_registration_params=None,
                        detect_outlier='0',
                        remove_temp_files=1,
                        verbose=0):
    from msct_register_regularized import register_seg, register_images, generate_warping_field
    from numpy import asarray, apply_along_axis, zeros
    from msct_smooth import smoothing_window, outliers_detection, outliers_completion
    # Calculate displacement
    current_algo = paramreg.algo
    if paramreg.type == 'seg':
        # calculate translation of center of mass between source and destination in voxel space
        res_reg = register_seg(fname_source, fname_dest, verbose)

    elif paramreg.type == 'im':
        if paramreg.algo == 'slicereg2d_pointwise':
            sct.printv('\nERROR: Algorithm slicereg2d_pointwise only operates for segmentation type.', verbose, 'error')
            sys.exit(2)
        algo_dic = {'slicereg2d_pointwise': 'Translation', 'slicereg2d_translation': 'Translation', 'slicereg2d_rigid': 'Rigid', 'slicereg2d_affine': 'Affine', 'slicereg2d_syn': 'SyN', 'slicereg2d_bsplinesyn': 'BSplineSyN'}
        paramreg.algo = algo_dic[current_algo]
        res_reg = register_images(fname_source, fname_dest, mask=fname_mask, paramreg=paramreg, remove_tmp_folder=remove_temp_files, ants_registration_params=ants_registration_params)

    else:
        sct.printv('\nERROR: wrong registration type inputed. pleas choose \'im\', or \'seg\'.', verbose, 'error')
        sys.exit(2)

    # if algo is slicereg2d _pointwise, -translation or _rigid: x_disp and y_disp are displacement fields
    # if algo is slicereg2d _affine, _syn or _bsplinesyn: x_disp and y_disp are warping fields names

    if current_algo in ['slicereg2d_pointwise', 'slicereg2d_translation', 'slicereg2d_rigid']:
        # Change to array
        x_disp, y_disp, theta_rot = res_reg
        x_disp_a = asarray(x_disp)
        y_disp_a = asarray(y_disp)
        # Detect outliers
        if not detect_outlier == '0':
            mask_x_a = outliers_detection(x_disp_a, type='median', factor=int(detect_outlier), return_filtered_signal='no', verbose=verbose)
            mask_y_a = outliers_detection(y_disp_a, type='median', factor=int(detect_outlier), return_filtered_signal='no', verbose=verbose)
            # Replace value of outliers by linear interpolation using closest non-outlier points
            x_disp_a_no_outliers = outliers_completion(mask_x_a, verbose=0)
            y_disp_a_no_outliers = outliers_completion(mask_y_a, verbose=0)
        else:
            x_disp_a_no_outliers = x_disp_a
            y_disp_a_no_outliers = y_disp_a
        # Smooth results
        if not window_length == '0':
            x_disp_smooth = smoothing_window(x_disp_a_no_outliers, window_len=int(window_length), window='hanning', verbose=verbose)
            y_disp_smooth = smoothing_window(y_disp_a_no_outliers, window_len=int(window_length), window='hanning', verbose=verbose)
        else:
            x_disp_smooth = x_disp_a_no_outliers
            y_disp_smooth = y_disp_a_no_outliers

        if theta_rot is not None:
            # same steps for theta_rot:
            theta_rot_a = asarray(theta_rot)
            mask_theta_a = outliers_detection(theta_rot_a, type='median', factor=int(detect_outlier), return_filtered_signal='no', verbose=verbose)
            theta_rot_a_no_outliers = outliers_completion(mask_theta_a, verbose=0)
            theta_rot_smooth = smoothing_window(theta_rot_a_no_outliers, window_len=int(window_length), window='hanning', verbose = verbose)
        else:
            theta_rot_smooth = None

        # Generate warping field
        generate_warping_field(fname_dest, x_disp_smooth, y_disp_smooth, theta_rot_smooth, fname=warp_forward_out)  #name_warp= 'step'+str(paramreg.step)
        # Inverse warping field
        generate_warping_field(fname_source, -x_disp_smooth, -y_disp_smooth, -theta_rot_smooth if theta_rot_smooth is not None else None, fname=warp_inverse_out)

    elif current_algo in ['slicereg2d_affine', 'slicereg2d_syn', 'slicereg2d_bsplinesyn']:
        from msct_image import Image
        warp_x, inv_warp_x, warp_y, inv_warp_y = res_reg
        im_warp_x = Image(warp_x)
        im_inv_warp_x = Image(inv_warp_x)
        im_warp_y = Image(warp_y)
        im_inv_warp_y = Image(inv_warp_y)

        data_warp_x = im_warp_x.data
        data_warp_x_inverse = im_inv_warp_x.data
        data_warp_y = im_warp_y.data
        data_warp_y_inverse = im_inv_warp_y.data

        hdr_warp = im_warp_x.hdr
        hdr_warp_inverse = im_inv_warp_x.hdr

        #Outliers deletion
        print'\n\tDeleting outliers...'
        mask_x_a = apply_along_axis(lambda m: outliers_detection(m, type='median', factor=int(detect_outlier), return_filtered_signal='no', verbose=0), axis=-1, arr=data_warp_x)
        mask_y_a = apply_along_axis(lambda m: outliers_detection(m, type='median', factor=int(detect_outlier), return_filtered_signal='no', verbose=0), axis=-1, arr=data_warp_y)
        mask_x_inverse_a = apply_along_axis(lambda m: outliers_detection(m, type='median', factor=int(detect_outlier), return_filtered_signal='no', verbose=0), axis=-1, arr=data_warp_x_inverse)
        mask_y_inverse_a = apply_along_axis(lambda m: outliers_detection(m, type='median', factor=int(detect_outlier), return_filtered_signal='no', verbose=0), axis=-1, arr=data_warp_y_inverse)
        #Outliers replacement by linear interpolation using closest non-outlier points
        data_warp_x_no_outliers = apply_along_axis(lambda m: outliers_completion(m, verbose=0), axis=-1, arr=mask_x_a)
        data_warp_y_no_outliers = apply_along_axis(lambda m: outliers_completion(m, verbose=0), axis=-1, arr=mask_y_a)
        data_warp_x_inverse_no_outliers = apply_along_axis(lambda m: outliers_completion(m, verbose=0), axis=-1, arr=mask_x_inverse_a)
        data_warp_y_inverse_no_outliers = apply_along_axis(lambda m: outliers_completion(m, verbose=0), axis=-1, arr=mask_y_inverse_a)
        #Smoothing of results along z
        print'\n\tSmoothing results...'
        data_warp_x_smooth = apply_along_axis(lambda m: smoothing_window(m, window_len=int(window_length), window='hanning', verbose=0), axis=-1, arr=data_warp_x_no_outliers)
        data_warp_x_smooth_inverse = apply_along_axis(lambda m: smoothing_window(m, window_len=int(window_length), window='hanning', verbose=0), axis=-1, arr=data_warp_x_inverse_no_outliers)
        data_warp_y_smooth = apply_along_axis(lambda m: smoothing_window(m, window_len=int(window_length), window='hanning', verbose=0), axis=-1, arr=data_warp_y_no_outliers)
        data_warp_y_smooth_inverse = apply_along_axis(lambda m: smoothing_window(m, window_len=int(window_length), window='hanning', verbose=0), axis=-1, arr=data_warp_y_inverse_no_outliers)

        print'\nSaving regularized warping fields...'
        # TODO: MODIFY NEXT PART
        #Get image dimensions of destination image
        from msct_image import Image
        from nibabel import load, Nifti1Image, save
        nx, ny, nz, nt, px, py, pz, pt = Image(fname_dest).dim
        data_warp_smooth = zeros(((((nx, ny, nz, 1, 3)))))
        data_warp_smooth[:,:,:,0,0] = data_warp_x_smooth
        data_warp_smooth[:,:,:,0,1] = data_warp_y_smooth
        data_warp_smooth_inverse = zeros(((((nx, ny, nz, 1, 3)))))
        data_warp_smooth_inverse[:,:,:,0,0] = data_warp_x_smooth_inverse
        data_warp_smooth_inverse[:,:,:,0,1] = data_warp_y_smooth_inverse
        # Force header's parameter to intent so that the file may be recognised as a warping field by ants
        hdr_warp.set_intent('vector', (), '')
        hdr_warp_inverse.set_intent('vector', (), '')
        img = Nifti1Image(data_warp_smooth, None, header=hdr_warp)
        img_inverse = Nifti1Image(data_warp_smooth_inverse, None, header=hdr_warp_inverse)
        save(img, filename=warp_forward_out)
        print'\tFile ' + warp_forward_out + ' saved.'
        save(img_inverse, filename=warp_inverse_out)
        print'\tFile ' + warp_inverse_out + ' saved.'
        return warp_forward_out, warp_inverse_out