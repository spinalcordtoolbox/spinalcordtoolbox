#!/usr/bin/env python

import sys, commands

# Get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# Append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')
from sct_register_multimodal import Paramreg



def register_slicereg2d_pointwise(src, dest, window_length=31, paramreg=Paramreg(step=0, type='seg', algo='slicereg2d_pointwise', metric='MeanSquares', iter= 10, shrink=1, smooth=0, gradStep=0.5), warp_forward_out='step0Warp.nii.gz', warp_inverse_out='step0InverseWarp.nii.gz', factor=2, verbose=0):
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


def register_slicereg2d_translation(src, dest, window_length=31, paramreg=Paramreg(step=0, type='im', algo='Translation', metric='MeanSquares', iter= 10, shrink=1, smooth=0, gradStep=0.5),
                                    fname_mask='', warp_forward_out='step0Warp.nii.gz', warp_inverse_out='step0InverseWarp.nii.gz', factor=2, remove_temp_files=1, verbose=0,
                                    ants_registration_params={'rigid': '', 'affine': '', 'compositeaffine': '', 'similarity': '', 'translation': '','bspline': ',10', 'gaussiandisplacementfield': ',3,0',
                                                              'bsplinedisplacementfield': ',5,10', 'syn': ',3,0', 'bsplinesyn': ',1,3'}):
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


def register_slicereg2d_rigid(src, dest, window_length=31, paramreg=Paramreg(step=0, type='im', algo='Rigid', metric='MeanSquares', iter= 10, shrink=1, smooth=0, gradStep=0.5),
                              fname_mask='', warp_forward_out='step0Warp.nii.gz', warp_inverse_out='step0InverseWarp.nii.gz', factor=2, remove_temp_files=1, verbose=0,
                              ants_registration_params={'rigid': '', 'affine': '', 'compositeaffine': '', 'similarity': '', 'translation': '','bspline': ',10', 'gaussiandisplacementfield': ',3,0',
                                                              'bsplinedisplacementfield': ',5,10', 'syn': ',3,0', 'bsplinesyn': ',1,3'}):
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


def register_slicereg2d_affine(src, dest, window_length=31, paramreg=Paramreg(step=0, type='im', algo='Affine', metric='MeanSquares', iter= 10, shrink=1, smooth=0, gradStep=0.5),
                               fname_mask='', warp_forward_out='step0Warp.nii.gz', warp_inverse_out='step0InverseWarp.nii.gz', factor=2, remove_temp_files=1, verbose=0,
                                    ants_registration_params={'rigid': '', 'affine': '', 'compositeaffine': '', 'similarity': '', 'translation': '','bspline': ',10', 'gaussiandisplacementfield': ',3,0',
                                                              'bsplinedisplacementfield': ',5,10', 'syn': ',3,0', 'bsplinesyn': ',1,3'}):
    from msct_register_regularized import register_images, generate_warping_field
    from numpy import asarray
    from numpy.linalg import inv
    from msct_smooth import smoothing_window, outliers_detection, outliers_completion

    # Calculate displacement
    x_disp, y_disp, matrix_def = register_images(src, dest, mask=fname_mask, paramreg=paramreg, remove_tmp_folder=remove_temp_files, ants_registration_params=ants_registration_params)
    # Change to array
    x_disp_a = asarray(x_disp)
    y_disp_a = asarray(y_disp)
    matrix_def_0_a = asarray([matrix_def[j][0][0] for j in range(len(matrix_def))])
    matrix_def_1_a = asarray([matrix_def[j][0][1] for j in range(len(matrix_def))])
    matrix_def_2_a = asarray([matrix_def[j][1][0] for j in range(len(matrix_def))])
    matrix_def_3_a = asarray([matrix_def[j][1][1] for j in range(len(matrix_def))])
    # Detect outliers
    mask_x_a = outliers_detection(x_disp_a, type='median', factor=factor, return_filtered_signal='no', verbose=verbose)
    mask_y_a = outliers_detection(y_disp_a, type='median', factor=factor, return_filtered_signal='no', verbose=verbose)
    mask_0_a = outliers_detection(matrix_def_0_a, type='median', factor=factor, return_filtered_signal='no', verbose=verbose)
    mask_1_a = outliers_detection(matrix_def_1_a, type='median', factor=factor, return_filtered_signal='no', verbose=verbose)
    mask_2_a = outliers_detection(matrix_def_2_a, type='median', factor=factor, return_filtered_signal='no', verbose=verbose)
    mask_3_a = outliers_detection(matrix_def_3_a, type='median', factor=factor, return_filtered_signal='no', verbose=verbose)
    # Replace value of outliers by linear interpolation using closest non-outlier points
    x_disp_a_no_outliers = outliers_completion(mask_x_a, verbose=0)
    y_disp_a_no_outliers = outliers_completion(mask_y_a, verbose=0)
    matrix_def_0_a_no_outliers = outliers_completion(mask_0_a, verbose=0)
    matrix_def_1_a_no_outliers = outliers_completion(mask_1_a, verbose=0)
    matrix_def_2_a_no_outliers = outliers_completion(mask_2_a, verbose=0)
    matrix_def_3_a_no_outliers = outliers_completion(mask_3_a, verbose=0)
    # Smooth results
    x_disp_smooth = smoothing_window(x_disp_a_no_outliers, window_len=int(window_length), window='hanning', verbose = verbose)
    y_disp_smooth = smoothing_window(y_disp_a_no_outliers, window_len=int(window_length), window='hanning', verbose = verbose)
    matrix_def_smooth_0 = smoothing_window(matrix_def_0_a_no_outliers, window_len=int(window_length), window='hanning', verbose = verbose)
    matrix_def_smooth_1 = smoothing_window(matrix_def_1_a_no_outliers, window_len=int(window_length), window='hanning', verbose = verbose)
    matrix_def_smooth_2 = smoothing_window(matrix_def_2_a_no_outliers, window_len=int(window_length), window='hanning', verbose = verbose)
    matrix_def_smooth_3 = smoothing_window(matrix_def_3_a_no_outliers, window_len=int(window_length), window='hanning', verbose = verbose)
    matrix_def_smooth = [[[matrix_def_smooth_0[iz], matrix_def_smooth_1[iz]], [matrix_def_smooth_2[iz], matrix_def_smooth_3[iz]]] for iz in range(len(matrix_def_smooth_0))]
    matrix_def_smooth_inv = inv(asarray(matrix_def_smooth)).tolist()
    # Generate warping field
    generate_warping_field(dest, x_disp_smooth, y_disp_smooth, matrix_def=matrix_def_smooth, fname=warp_forward_out)
    # Inverse warping field
    generate_warping_field(src, -x_disp_smooth, -y_disp_smooth, matrix_def=matrix_def_smooth_inv, fname=warp_inverse_out)


def register_slicereg2d_syn(src, dest, window_length=31, paramreg=Paramreg(step=0, type='im', algo='SyN', metric='MeanSquares', iter= 10, shrink=1, smooth=0, gradStep=0.5),
                            fname_mask='', warp_forward_out='step0Warp.nii.gz', warp_inverse_out='step0InverseWarp.nii.gz', factor=2, remove_temp_files=1,
                                    ants_registration_params={'rigid': '', 'affine': '', 'compositeaffine': '', 'similarity': '', 'translation': '','bspline': ',10', 'gaussiandisplacementfield': ',3,0',
                                                              'bsplinedisplacementfield': ',5,10', 'syn': ',3,0', 'bsplinesyn': ',1,3'}):
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


def register_slicereg2d_bsplinesyn(src, dest, window_length=31, paramreg=Paramreg(step=0, type='im', algo='BSplineSyN', metric='MeanSquares', iter= 10, shrink=1, smooth=0, gradStep=0.5),
                                   fname_mask='', warp_forward_out='step0Warp.nii.gz', warp_inverse_out='step0InverseWarp.nii.gz', factor=2, remove_temp_files=1,
                                    ants_registration_params={'rigid': '', 'affine': '', 'compositeaffine': '', 'similarity': '', 'translation': '','bspline': ',10', 'gaussiandisplacementfield': ',3,0',
                                                              'bsplinedisplacementfield': ',5,10', 'syn': ',3,0', 'bsplinesyn': ',1,3'}):
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

