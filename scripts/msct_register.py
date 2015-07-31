#!/usr/bin/env python
#
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 NeuroPoly, Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Tanguy Magnan
# Modified: 2015-07-29
#
# License: see the LICENSE.TXT
#=======================================================================================================================
#
import sys, commands

# Get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# Append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')
from sct_register_multimodal import Paramreg



def register_slicereg2d_pointwise(src, dest, window_length=31, paramreg=Paramreg(step='0', type='seg', algo='slicereg2d_pointwise', metric='MeanSquares', iter='10', shrink='1', smooth='0', gradStep='0.5'),
                                  warp_forward_out='step0Warp.nii.gz', warp_inverse_out='step0InverseWarp.nii.gz', factor=2, verbose=0):
    """Slice-by-slice regularized registration by translation of two segmentations.

    First we estimate for each slice the translation vector by calculating the difference of position of the two centers of
    mass of the two segmentations. Then we remove outliers using Median Absolute Deviation technique (MAD) and smooth
    the translation along x and y axis using moving average hanning window. Eventually, we generate two warping fields
    (forward and inverse) resulting from this regularized registration technique.
    The segmentations must be of same size (otherwise generate_warping_field will not work for forward or inverse
    creation).

    input:
        src: name of moving image (type: string)
        dest: name of fixed image (type: string)
        window_length: size of window for moving average smoothing (type: int)
        paramreg[optional]: parameters of antsRegistration (type: Paramreg class from sct_register_multimodal)
        warp_forward_out: name of output forward warp (type: string)
        warp_inverse_out: name of output inverse warp (type: string)
        factor: sensibility factor for outlier detection (higher the factor, smaller the detection) (type: int or float)
        verbose: display parameter (type: int, value: 0,1 or 2)

    output:
        creation of warping field files of name 'warp_forward_out' and 'warp_inverse_out'.

    """
    if paramreg.type != 'seg':
        print '\nERROR: Algorithm slicereg2d_pointwise only operates for segmentation type.'
        sys.exit(2)
    else:
        from msct_register_regularized import register_seg, generate_warping_field
        from numpy import asarray
        from msct_smooth import smoothing_window, outliers_detection, outliers_completion
        # Calculate displacement
        x_disp, y_disp = register_seg(src, dest)
        # Change to array
        x_disp_a = asarray(x_disp)
        y_disp_a = asarray(y_disp)
        # Detect outliers
        mask_x_a = outliers_detection(x_disp_a, type='median', factor=factor, return_filtered_signal='no', verbose=verbose)
        mask_y_a = outliers_detection(y_disp_a, type='median', factor=factor, return_filtered_signal='no', verbose=verbose)
        # Replace value of outliers by linear interpolation using closest non-outlier points
        x_disp_a_no_outliers = outliers_completion(mask_x_a, verbose=0)
        y_disp_a_no_outliers = outliers_completion(mask_y_a, verbose=0)
        # Smooth results
        x_disp_smooth = smoothing_window(x_disp_a_no_outliers, window_len=int(window_length), window='hanning', verbose=verbose)
        y_disp_smooth = smoothing_window(y_disp_a_no_outliers, window_len=int(window_length), window='hanning', verbose=verbose)
        # Generate warping field
        generate_warping_field(dest, x_disp_smooth, y_disp_smooth, fname=warp_forward_out)  #name_warp= 'step'+str(paramreg.step)
        # Inverse warping field
        generate_warping_field(src, -x_disp_smooth, -y_disp_smooth, fname=warp_inverse_out)


def register_slicereg2d_translation(src, dest, window_length=31, paramreg=Paramreg(step='0', type='im', algo='Translation', metric='MeanSquares', iter='10', shrink='1', smooth='0', gradStep='0.5'),
                                    fname_mask='', warp_forward_out='step0Warp.nii.gz', warp_inverse_out='step0InverseWarp.nii.gz', factor=2, remove_temp_files=1, verbose=0,
                                    ants_registration_params={'rigid': '', 'affine': '', 'compositeaffine': '', 'similarity': '', 'translation': '','bspline': ',10', 'gaussiandisplacementfield': ',3,0',
                                                              'bsplinedisplacementfield': ',5,10', 'syn': ',3,0', 'bsplinesyn': ',1,3'}):
    """Slice-by-slice regularized registration by translation of two images.

    We first register slice-by-slice the two images using antsRegistration in 2D. Then we remove outliers using
    Median Absolute Deviation technique (MAD) and smooth the translations along x and y axis using moving average
    hanning window. Eventually, we generate two warping fields (forward and inverse) resulting from this regularized
    registration technique.
    The images must be of same size (otherwise generate_warping_field will not work for forward or inverse
    creation).

    input:
        src: name of moving image (type: string)
        dest: name of fixed image (type: string)
        window_length[optional]: size of window for moving average smoothing (type: int)
        paramreg[optional]: parameters of antsRegistration (type: Paramreg class from sct_register_multimodal)
        fname_mask[optional]: name of mask file (type: string) (parameter -x of antsRegistration)
        warp_forward_out[optional]: name of output forward warp (type: string)
        warp_inverse_out[optional]: name of output inverse warp (type: string)
        factor[optional]: sensibility factor for outlier detection (higher the factor, smaller the detection)
            (type: int or float)
        remove_temp_files[optional]: 1 to remove, 0 to keep (type: int)
        verbose[optional]: display parameter (type: int, value: 0,1 or 2)
        ants_registration_params[optional]: specific algorithm's parameters for antsRegistration (type: dictionary)

    output:
        creation of warping field files of name 'warp_forward_out' and 'warp_inverse_out'.
    """
    from msct_register_regularized import register_images, generate_warping_field
    from numpy import asarray
    from msct_smooth import smoothing_window, outliers_detection, outliers_completion

    # Calculate displacement
    x_disp, y_disp = register_images(src, dest, mask=fname_mask, paramreg=paramreg, remove_tmp_folder=remove_temp_files, ants_registration_params=ants_registration_params)
    # Change to array
    x_disp_a = asarray(x_disp)
    y_disp_a = asarray(y_disp)
    # Detect outliers
    mask_x_a = outliers_detection(x_disp_a, type='median', factor=factor, return_filtered_signal='no', verbose=verbose)
    mask_y_a = outliers_detection(y_disp_a, type='median', factor=factor, return_filtered_signal='no', verbose=verbose)
    # Replace value of outliers by linear interpolation using closest non-outlier points
    x_disp_a_no_outliers = outliers_completion(mask_x_a, verbose=0)
    y_disp_a_no_outliers = outliers_completion(mask_y_a, verbose=0)
    # Smooth results
    x_disp_smooth = smoothing_window(x_disp_a_no_outliers, window_len=int(window_length), window='hanning', verbose = verbose)
    y_disp_smooth = smoothing_window(y_disp_a_no_outliers, window_len=int(window_length), window='hanning', verbose = verbose)
    # Generate warping field
    generate_warping_field(dest, x_disp_smooth, y_disp_smooth, fname=warp_forward_out)
    # Inverse warping field
    generate_warping_field(src, -x_disp_smooth, -y_disp_smooth, fname=warp_inverse_out)


def register_slicereg2d_rigid(src, dest, window_length=31, paramreg=Paramreg(step='0', type='im', algo='Rigid', metric='MeanSquares', iter='10', shrink='1', smooth='0', gradStep='0.5'),
                              fname_mask='', warp_forward_out='step0Warp.nii.gz', warp_inverse_out='step0InverseWarp.nii.gz', factor=2, remove_temp_files=1, verbose=0,
                              ants_registration_params={'rigid': '', 'affine': '', 'compositeaffine': '', 'similarity': '', 'translation': '','bspline': ',10', 'gaussiandisplacementfield': ',3,0',
                                                              'bsplinedisplacementfield': ',5,10', 'syn': ',3,0', 'bsplinesyn': ',1,3'}):
    """Slice-by-slice regularized registration (rigid) of two images.

    We first register slice-by-slice the two images using antsRegistration in 2D. Then we remove outliers using
    Median Absolute Deviation technique (MAD) and smooth the translations and angle of rotation along x and y axis using
    moving average hanning window. Eventually, we generate two warping fields (forward and inverse) resulting from this
    regularized registration technique.
    The images must be of same size (otherwise generate_warping_field will not work for forward or inverse
    creation).

    input:
        src: name of moving image (type: string)
        dest: name of fixed image (type: string)
        window_length[optional]: size of window for moving average smoothing (type: int)
        paramreg[optional]: parameters of antsRegistration (type: Paramreg class from sct_register_multimodal)
        fname_mask[optional]: name of mask file (type: string) (parameter -x of antsRegistration)
        warp_forward_out[optional]: name of output forward warp (type: string)
        warp_inverse_out[optional]: name of output inverse warp (type: string)
        factor[optional]: sensibility factor for outlier detection (higher the factor, smaller the detection)
            (type: int or float)
        remove_temp_files[optional]: 1 to remove, 0 to keep (type: int)
        verbose[optional]: display parameter (type: int, value: 0,1 or 2)
        ants_registration_params[optional]: specific algorithm's parameters for antsRegistration (type: dictionary)

    output:
        creation of warping field files of name 'warp_forward_out' and 'warp_inverse_out'.
    """
    from msct_register_regularized import register_images, generate_warping_field
    from numpy import asarray
    from msct_smooth import smoothing_window, outliers_detection, outliers_completion

    # Calculate displacement
    x_disp, y_disp, theta_rot = register_images(src, dest, mask=fname_mask, paramreg=paramreg, remove_tmp_folder=remove_temp_files, ants_registration_params=ants_registration_params)
    # Change to array
    x_disp_a = asarray(x_disp)
    y_disp_a = asarray(y_disp)
    theta_rot_a = asarray(theta_rot)
    # Detect outliers
    mask_x_a = outliers_detection(x_disp_a, type='median', factor=factor, return_filtered_signal='no', verbose=verbose)
    mask_y_a = outliers_detection(y_disp_a, type='median', factor=factor, return_filtered_signal='no', verbose=verbose)
    mask_theta_a = outliers_detection(theta_rot_a, type='median', factor=2, return_filtered_signal='no', verbose=verbose)
    # Replace value of outliers by linear interpolation using closest non-outlier points
    x_disp_a_no_outliers = outliers_completion(mask_x_a, verbose=0)
    y_disp_a_no_outliers = outliers_completion(mask_y_a, verbose=0)
    theta_rot_a_no_outliers = outliers_completion(mask_theta_a, verbose=0)
    # Smooth results
    x_disp_smooth = smoothing_window(x_disp_a_no_outliers, window_len=int(window_length), window='hanning', verbose = verbose)
    y_disp_smooth = smoothing_window(y_disp_a_no_outliers, window_len=int(window_length), window='hanning', verbose = verbose)
    theta_rot_smooth = smoothing_window(theta_rot_a_no_outliers, window_len=int(window_length), window='hanning', verbose = verbose)
    # Generate warping field
    generate_warping_field(dest, x_disp_smooth, y_disp_smooth, theta_rot_smooth, fname=warp_forward_out)
    # Inverse warping field
    generate_warping_field(src, -x_disp_smooth, -y_disp_smooth, -theta_rot_smooth, fname=warp_inverse_out)


def register_slicereg2d_affine(src, dest, window_length=31, paramreg=Paramreg(step='0', type='im', algo='Affine', metric='MeanSquares', iter='10', shrink='1', smooth='0', gradStep='0.5'),
                               fname_mask='', warp_forward_out='step0Warp.nii.gz', warp_inverse_out='step0InverseWarp.nii.gz', factor=2, remove_temp_files=1, verbose=0,
                                    ants_registration_params={'rigid': '', 'affine': '', 'compositeaffine': '', 'similarity': '', 'translation': '','bspline': ',10', 'gaussiandisplacementfield': ',3,0',
                                                              'bsplinedisplacementfield': ',5,10', 'syn': ',3,0', 'bsplinesyn': ',1,3'}):
    """Slice-by-slice regularized registration (affine) of two images.

    We first register slice-by-slice the two images using antsRegistration in 2D (algo: affine) and create 3D warping
    fields (forward and inverse) by merging the 2D warping fields along z. Then we directly detect outliers and smooth
    the 3d warping fields applying a moving average hanning window on each pixel of the plan xOy (i.e. we consider that
    for a position (x,y) in the plan xOy, the variation along z of the vector of displacement (xo, yo, zo) of the
    warping field should not be too abrupt). Eventually, we generate two warping fields (forward and inverse) resulting
    from this regularized registration technique.
    The images must be of same size (otherwise generate_warping_field will not work for forward or inverse
    creation).

    input:
        src: name of moving image (type: string)
        dest: name of fixed image (type: string)
        window_length[optional]: size of window for moving average smoothing (type: int)
        paramreg[optional]: parameters of antsRegistration (type: Paramreg class from sct_register_multimodal)
        fname_mask[optional]: name of mask file (type: string) (parameter -x of antsRegistration)
        warp_forward_out[optional]: name of output forward warp (type: string)
        warp_inverse_out[optional]: name of output inverse warp (type: string)
        factor[optional]: sensibility factor for outlier detection (higher the factor, smaller the detection)
            (type: int or float)
        remove_temp_files[optional]: 1 to remove, 0 to keep (type: int)
        verbose[optional]: display parameter (type: int, value: 0,1 or 2)
        ants_registration_params[optional]: specific algorithm's parameters for antsRegistration (type: dictionary)

    output:
        creation of warping field files of name 'warp_forward_out' and 'warp_inverse_out'.
    """
    from nibabel import load, Nifti1Image, save
    from msct_smooth import smoothing_window, outliers_detection, outliers_completion
    from msct_register_regularized import register_images
    from numpy import apply_along_axis, zeros
    import sct_utils as sct
    name_warp_syn = 'Warp_total'

    # Calculate displacement
    register_images(src, dest, mask=fname_mask, paramreg=paramreg, remove_tmp_folder=remove_temp_files, ants_registration_params=ants_registration_params)

    print'\nRegularizing warping fields along z axis...'
    print'\n\tSplitting warping fields ...'
    sct.run('isct_c3d -mcs ' + name_warp_syn + '.nii.gz -oo ' + name_warp_syn + '_x.nii.gz ' + name_warp_syn + '_y.nii.gz')
    sct.run('isct_c3d -mcs ' + name_warp_syn + '_inverse.nii.gz -oo ' + name_warp_syn + '_x_inverse.nii.gz ' + name_warp_syn + '_y_inverse.nii.gz')
    data_warp_x = load(name_warp_syn + '_x.nii.gz').get_data()
    data_warp_y = load(name_warp_syn + '_y.nii.gz').get_data()
    hdr_warp = load(name_warp_syn + '_x.nii.gz').get_header()
    data_warp_x_inverse = load(name_warp_syn + '_x_inverse.nii.gz').get_data()
    data_warp_y_inverse = load(name_warp_syn + '_y_inverse.nii.gz').get_data()
    hdr_warp_inverse = load(name_warp_syn + '_x_inverse.nii.gz').get_header()
    #Outliers deletion
    print'\n\tDeleting outliers...'
    mask_x_a = apply_along_axis(lambda m: outliers_detection(m, type='median', factor=factor, return_filtered_signal='no', verbose=0), axis=-1, arr=data_warp_x)
    mask_y_a = apply_along_axis(lambda m: outliers_detection(m, type='median', factor=factor, return_filtered_signal='no', verbose=0), axis=-1, arr=data_warp_y)
    mask_x_inverse_a = apply_along_axis(lambda m: outliers_detection(m, type='median', factor=factor, return_filtered_signal='no', verbose=0), axis=-1, arr=data_warp_x_inverse)
    mask_y_inverse_a = apply_along_axis(lambda m: outliers_detection(m, type='median', factor=factor, return_filtered_signal='no', verbose=0), axis=-1, arr=data_warp_y_inverse)
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
    #Get image dimensions of destination image
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(dest)
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

def register_slicereg2d_syn(src, dest, window_length=31, paramreg=Paramreg(step='0', type='im', algo='SyN', metric='MeanSquares', iter='10', shrink='1', smooth='0', gradStep='0.5'),
                            fname_mask='', warp_forward_out='step0Warp.nii.gz', warp_inverse_out='step0InverseWarp.nii.gz', factor=2, remove_temp_files=1, verbose=0,
                                    ants_registration_params={'rigid': '', 'affine': '', 'compositeaffine': '', 'similarity': '', 'translation': '','bspline': ',10', 'gaussiandisplacementfield': ',3,0',
                                                              'bsplinedisplacementfield': ',5,10', 'syn': ',3,0', 'bsplinesyn': ',1,3'}):
    """Slice-by-slice regularized registration (syn) of two images.

    We first register slice-by-slice the two images using antsRegistration in 2D (algo: syn) and create 3D warping
    fields (forward and inverse) by merging the 2D warping fields along z. Then we directly detect outliers and smooth
    the 3d warping fields applying a moving average hanning window on each pixel of the plan xOy (i.e. we consider that
    for a position (x,y) in the plan xOy, the variation along z of the vector of displacement (xo, yo, zo) of the
    warping field should not be too abrupt). Eventually, we generate two warping fields (forward and inverse) resulting
    from this regularized registration technique.
    The images must be of same size (otherwise generate_warping_field will not work for forward or inverse
    creation).

    input:
        src: name of moving image (type: string)
        dest: name of fixed image (type: string)
        window_length[optional]: size of window for moving average smoothing (type: int)
        paramreg[optional]: parameters of antsRegistration (type: Paramreg class from sct_register_multimodal)
        fname_mask[optional]: name of mask file (type: string) (parameter -x of antsRegistration)
        warp_forward_out[optional]: name of output forward warp (type: string)
        warp_inverse_out[optional]: name of output inverse warp (type: string)
        factor[optional]: sensibility factor for outlier detection (higher the factor, smaller the detection)
            (type: int or float)
        remove_temp_files[optional]: 1 to remove, 0 to keep (type: int)
        verbose[optional]: display parameter (type: int, value: 0,1 or 2)
        ants_registration_params[optional]: specific algorithm's parameters for antsRegistration (type: dictionary)

    output:
        creation of warping field files of name 'warp_forward_out' and 'warp_inverse_out'.
    """
    from nibabel import load, Nifti1Image, save
    from msct_smooth import smoothing_window, outliers_detection, outliers_completion
    from msct_register_regularized import register_images
    from numpy import apply_along_axis, zeros
    import sct_utils as sct
    name_warp_syn = 'Warp_total'
    # Registrating images
    register_images(src, dest, mask=fname_mask, paramreg=paramreg, remove_tmp_folder=remove_temp_files, ants_registration_params=ants_registration_params)
    print'\nRegularizing warping fields along z axis...'
    print'\n\tSplitting warping fields ...'
    sct.run('isct_c3d -mcs ' + name_warp_syn + '.nii.gz -oo ' + name_warp_syn + '_x.nii.gz ' + name_warp_syn + '_y.nii.gz')
    sct.run('isct_c3d -mcs ' + name_warp_syn + '_inverse.nii.gz -oo ' + name_warp_syn + '_x_inverse.nii.gz ' + name_warp_syn + '_y_inverse.nii.gz')
    data_warp_x = load(name_warp_syn + '_x.nii.gz').get_data()
    data_warp_y = load(name_warp_syn + '_y.nii.gz').get_data()
    hdr_warp = load(name_warp_syn + '_x.nii.gz').get_header()
    data_warp_x_inverse = load(name_warp_syn + '_x_inverse.nii.gz').get_data()
    data_warp_y_inverse = load(name_warp_syn + '_y_inverse.nii.gz').get_data()
    hdr_warp_inverse = load(name_warp_syn + '_x_inverse.nii.gz').get_header()
    #Outliers deletion
    print'\n\tDeleting outliers...'
    mask_x_a = apply_along_axis(lambda m: outliers_detection(m, type='median', factor=factor, return_filtered_signal='no', verbose=0), axis=-1, arr=data_warp_x)
    mask_y_a = apply_along_axis(lambda m: outliers_detection(m, type='median', factor=factor, return_filtered_signal='no', verbose=0), axis=-1, arr=data_warp_y)
    mask_x_inverse_a = apply_along_axis(lambda m: outliers_detection(m, type='median', factor=factor, return_filtered_signal='no', verbose=0), axis=-1, arr=data_warp_x_inverse)
    mask_y_inverse_a = apply_along_axis(lambda m: outliers_detection(m, type='median', factor=factor, return_filtered_signal='no', verbose=0), axis=-1, arr=data_warp_y_inverse)
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
    #Get image dimensions of destination image
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(dest)
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


def register_slicereg2d_bsplinesyn(src, dest, window_length=31, paramreg=Paramreg(step='0', type='im', algo='BSplineSyN', metric='MeanSquares', iter='10', shrink='1', smooth='0', gradStep='0.5'),
                                   fname_mask='', warp_forward_out='step0Warp.nii.gz', warp_inverse_out='step0InverseWarp.nii.gz', factor=2, remove_temp_files=1, verbose=0,
                                    ants_registration_params={'rigid': '', 'affine': '', 'compositeaffine': '', 'similarity': '', 'translation': '','bspline': ',10', 'gaussiandisplacementfield': ',3,0',
                                                              'bsplinedisplacementfield': ',5,10', 'syn': ',3,0', 'bsplinesyn': ',1,3'}):
    """Slice-by-slice regularized registration (bsplinesyn) of two images.

    We first register slice-by-slice the two images using antsRegistration in 2D (algo: bsplinesyn) and create 3D warping
    fields (forward and inverse) by merging the 2D warping fields along z. Then we directly detect outliers and smooth
    the 3d warping fields applying a moving average hanning window on each pixel of the plan xOy (i.e. we consider that
    for a position (x,y) in the plan xOy, the variation along z of the vector of displacement (xo, yo, zo) of the
    warping field should not be too abrupt). Eventually, we generate two warping fields (forward and inverse) resulting
    from this regularized registration technique.
    The images must be of same size (otherwise generate_warping_field will not work for forward or inverse
    creation).

    input:
        src: name of moving image (type: string)
        dest: name of fixed image (type: string)
        window_length[optional]: size of window for moving average smoothing (type: int)
        paramreg[optional]: parameters of antsRegistration (type: Paramreg class from sct_register_multimodal)
        fname_mask[optional]: name of mask file (type: string) (parameter -x of antsRegistration)
        warp_forward_out[optional]: name of output forward warp (type: string)
        warp_inverse_out[optional]: name of output inverse warp (type: string)
        factor[optional]: sensibility factor for outlier detection (higher the factor, smaller the detection)
            (type: int or float)
        remove_temp_files[optional]: 1 to remove, 0 to keep (type: int)
        verbose[optional]: display parameter (type: int, value: 0,1 or 2)
        ants_registration_params[optional]: specific algorithm's parameters for antsRegistration (type: dictionary)

    output:
        creation of warping field files of name 'warp_forward_out' and 'warp_inverse_out'.
    """
    from nibabel import load, Nifti1Image, save
    from msct_smooth import smoothing_window, outliers_detection, outliers_completion
    from msct_register_regularized import register_images
    from numpy import apply_along_axis, zeros
    import sct_utils as sct
    name_warp_syn = 'Warp_total'
    # Registrating images
    register_images(src, dest, mask=fname_mask, paramreg=paramreg, remove_tmp_folder=remove_temp_files, ants_registration_params=ants_registration_params)
    print'\nRegularizing warping fields along z axis...'
    print'\n\tSplitting warping fields ...'
    sct.run('isct_c3d -mcs ' + name_warp_syn + '.nii.gz -oo ' + name_warp_syn + '_x.nii.gz ' + name_warp_syn + '_y.nii.gz')
    sct.run('isct_c3d -mcs ' + name_warp_syn + '_inverse.nii.gz -oo ' + name_warp_syn + '_x_inverse.nii.gz ' + name_warp_syn + '_y_inverse.nii.gz')
    data_warp_x = load(name_warp_syn + '_x.nii.gz').get_data()
    data_warp_y = load(name_warp_syn + '_y.nii.gz').get_data()
    hdr_warp = load(name_warp_syn + '_x.nii.gz').get_header()
    data_warp_x_inverse = load(name_warp_syn + '_x_inverse.nii.gz').get_data()
    data_warp_y_inverse = load(name_warp_syn + '_y_inverse.nii.gz').get_data()
    hdr_warp_inverse = load(name_warp_syn + '_x_inverse.nii.gz').get_header()
    #Outliers deletion
    print'\n\tDeleting outliers...'
    mask_x_a = apply_along_axis(lambda m: outliers_detection(m, type='median', factor=factor, return_filtered_signal='no', verbose=0), axis=-1, arr=data_warp_x)
    mask_y_a = apply_along_axis(lambda m: outliers_detection(m, type='median', factor=factor, return_filtered_signal='no', verbose=0), axis=-1, arr=data_warp_y)
    mask_x_inverse_a = apply_along_axis(lambda m: outliers_detection(m, type='median', factor=factor, return_filtered_signal='no', verbose=0), axis=-1, arr=data_warp_x_inverse)
    mask_y_inverse_a = apply_along_axis(lambda m: outliers_detection(m, type='median', factor=factor, return_filtered_signal='no', verbose=0), axis=-1, arr=data_warp_y_inverse)
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
    #Get image dimensions of destination image
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(dest)
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
