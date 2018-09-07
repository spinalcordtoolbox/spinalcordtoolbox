#!/usr/bin/env python
#########################################################################################
#
# List of functions for moco.
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Karun Raju, Tanguy Duval, Julien Cohen-Adad
# Modified: 2014-10-04
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: check the status of spline()
# TODO: check the status of combine_matrix()
# TODO: add tests with sag and ax orientation, with -g 1 and 3
# TODO: make it a spinalcordtoolbox module with im as input
# TODO: params for ANTS: CC/MI, shrink fact, nb_it
# TODO: ants: explore optin  --float  for faster computation

from __future__ import absolute_import

import sys, os, glob

import numpy as np
import scipy.interpolate

import sct_utils as sct
from spinalcordtoolbox.image import Image
from sct_image import split_data, concat_data
import sct_apply_transfo

path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))

#=======================================================================================================================
# moco Function
#=======================================================================================================================
def moco(param):

    # retrieve parameters
    file_data = param.file_data
    file_target = param.file_target
    folder_mat = param.mat_moco  # output folder of mat file
    todo = param.todo
    suffix = param.suffix
    #file_schedule = param.file_schedule
    verbose = param.verbose
    ext = '.nii'

    # get path of the toolbox

    # sct.printv(arguments)
    sct.printv('\nInput parameters:', param.verbose)
    sct.printv('  Input file ............' + file_data, param.verbose)
    sct.printv('  Reference file ........' + file_target, param.verbose)
    sct.printv('  Polynomial degree .....' + param.poly, param.verbose)
    sct.printv('  Smoothing kernel ......' + param.smooth, param.verbose)
    sct.printv('  Gradient step .........' + param.gradStep, param.verbose)
    sct.printv('  Metric ................' + param.metric, param.verbose)
    sct.printv('  Sampling ..............' + param.sampling, param.verbose)
    sct.printv('  Todo ..................' + todo, param.verbose)
    sct.printv('  Mask  .................' + param.fname_mask, param.verbose)
    sct.printv('  Output mat folder .....' + folder_mat, param.verbose)

    # create folder for mat files
    sct.create_folder(folder_mat)

    # Get size of data
    sct.printv('\nData dimensions:', verbose)
    im_data = Image(file_data + ext)
    nx, ny, nz, nt, px, py, pz, pt = im_data.dim
    sct.printv(('  ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz) + ' x ' + str(nt)), verbose)

    # Get orientation
    sct.printv('\nData orientation: ' + im_data.orientation, verbose)
    if im_data.orientation[2] in 'LR':
        is_sagittal = True
        sct.printv('  Treated as sagittal')
    elif im_data.orientation[2] in 'IS':
        is_sagittal = False
        sct.printv('  Treated as axial')
    else:
        is_sagittal = False
        sct.printv('WARNING: Orientation seems to be neither axial nor sagittal.')

    # copy file_target to a temporary file
    sct.printv('\nCopy file_target to a temporary file...', verbose)
    sct.copy(file_target + ext, 'target.nii')
    file_target = 'target'

    # If scan is sagittal, split src and target along Z (slice)
    if is_sagittal:
        dim_sag = 2  # TODO: find it
        # z-split data (time series)
        im_z_list = split_data(im_data, dim=dim_sag, squeeze_data=False)
        file_data_splitZ = []
        for im_z in im_z_list:
            im_z.save()
            file_data_splitZ.append(im_z.absolutepath)
        # z-split target
        im_targetz_list = split_data(Image(file_target+ext), dim=dim_sag, squeeze_data=False)
        file_target_splitZ = []
        for im_targetz in im_targetz_list:
            im_targetz.save()
            file_target_splitZ.append(im_targetz.absolutepath)
        # initialize file list for output matrices
        file_mat = np.chararray([nz, nt],
                                itemsize=50)  # itemsize=50 is to accomodate relative path to matrix file name.

    else:
        file_data_splitZ = [file_data + ext]  # TODO: make it absolute like above
        file_target_splitZ = [file_target + ext]  # TODO: make it absolute like above
        # initialize file list for output matrices
        file_mat = np.chararray([1, nt],
                                itemsize=50)  # itemsize=50 is to accomodate relative path to matrix file name.

    # Loop across file list, where each file is either a 2D volume (if sagittal) or a 3D volume (otherwise)
    # file_mat = tuple([[[] for i in range(nt)] for i in range(nz)])
    file_mat[:] = ''  # init
    file_data_splitZ_moco = []
    for file in file_data_splitZ:
        iz = file_data_splitZ.index(file)
        # Split data along T dimension
        sct.printv('\nSplit data along T dimension...', verbose)
        im_z = Image(file)
        list_im_zt = split_data(im_z, dim=3)
        file_data_splitZ_splitT = []
        for im_zt in list_im_zt:
            im_zt.save()
            file_data_splitZ_splitT.append(im_zt.absolutepath)
        # file_data_splitT = file_data + '_T'

        # Motion correction: initialization
        index = np.arange(nt)
        file_data_splitT_num = []
        file_data_splitZ_splitT_moco = []
        failed_transfo = [0 for i in range(nt)]

        # Motion correction: Loop across T
        for indice_index in range(nt):

            # create indices and display stuff
            it = index[indice_index]
            # file_data_splitT_num.append(file_data_splitT + str(it).zfill(4))
            # file_data_splitZ_splitT_moco.append(file_data + suffix + '_T' + str(it).zfill(4))
            sct.printv(('\nVolume ' + str((it)) + '/' + str(nt - 1) + ':'), verbose)
            file_mat[iz][it] = os.path.join(folder_mat, "mat.Z") + str(iz).zfill(4) + 'T' + str(it).zfill(4)
            file_data_splitZ_splitT_moco.append(sct.add_suffix(file_data_splitZ_splitT[it], '_moco'))
            # run 3D registration
            failed_transfo[it] = register(param, file_data_splitZ_splitT[it], file_target_splitZ[iz], file_mat[iz][it], file_data_splitZ_splitT_moco[it])

            # average registered volume with target image
            # N.B. use weighted averaging: (target * nb_it + moco) / (nb_it + 1)
            if param.iterAvg and indice_index < 10 and failed_transfo[it] == 0 and not param.todo == 'apply':
                im_targetz = Image(file_target_splitZ[iz])
                data_targetz = im_targetz.data
                data_mocoz = Image(file_data_splitZ_splitT_moco[it]).data
                data_targetz = (data_targetz * (indice_index + 1) + data_mocoz) / (indice_index + 2)
                im_targetz.data = data_targetz
                im_targetz.save(verbose=0)
                # sct.run(["sct_maths", "-i", file_target_splitZ[iz], "-mul", str(indice_index + 1), "-o", file_target_splitZ[iz]])
                # sct.run(["sct_maths", "-i", file_target_splitZ[iz], "-add", file_data_splitZ_splitT_moco[it], "-o", file_target_splitZ[iz]])
                # sct.run(["sct_maths", "-i", file_target_splitZ[iz], "-div", str(indice_index + 2), "-o", file_target_splitZ[iz]])

        # Replace failed transformation with the closest good one
        sct.printv(('\nReplace failed transformations...'), verbose)
        fT = [i for i, j in enumerate(failed_transfo) if j == 1]
        gT = [i for i, j in enumerate(failed_transfo) if j == 0]
        for it in range(len(fT)):
            abs_dist = [abs(gT[i] - fT[it]) for i in range(len(gT))]
            if not abs_dist == []:
                index_good = abs_dist.index(min(abs_dist))
                sct.printv('  transfo #' + str(fT[it]) + ' --> use transfo #' + str(gT[index_good]), verbose)
                # copy transformation
                sct.copy(file_mat[iz][gT[index_good]] + 'Warp.nii.gz', file_mat[iz][fT[it]] + 'Warp.nii.gz')
                # apply transformation
                sct.run(["sct_apply_transfo",
                 "-i", file_data_splitT_num[fT[it]] + ".nii",
                 "-d", file_target + ".nii",
                 "-w", file_mat[iz][fT[it]] + 'Warp.nii.gz',
                 "-o", file_data_splitZ_splitT_moco[fT[it]] + '.nii',
                 "-x", param.interp], verbose)
            else:
                # exit program if no transformation exists.
                sct.printv('\nERROR in ' + os.path.basename(__file__) + ': No good transformation exist. Exit program.\n', verbose, 'error')
                sys.exit(2)

        # Merge data along T
        file_data_splitZ_moco.append(sct.add_suffix(file, suffix))
        if todo != 'estimate':
            sct.printv('\nMerge data back along T...', verbose)
            # im_list = []
            # fname_list = []
            # for indice_index in range(len(index)):
                # im_list.append(Image(file_data_splitZ_splitT_moco[indice_index] + ext))
                # fname_list.append(file_data_splitZ_splitT_moco[indice_index] + ext)
            im_out = concat_data(file_data_splitZ_splitT_moco, 3)
            im_out.save(file_data_splitZ_moco[iz])

    # If sagittal, merge along Z
    if is_sagittal:
        sct.printv('\nMerge data back along Z...', verbose)
        im_out = concat_data(file_data_splitZ_moco, 2)
        im_out.save(file_data + suffix + ext)

    return file_mat


def register(param, file_src, file_dest, file_mat, file_out):
    """
    Register two images by estimating slice-wise Tx and Ty transformations, which are regularized along Z. This function
    uses ANTs' isct_antsSliceRegularizedRegistration.
    :param param:
    :param file_src:
    :param file_dest:
    :param file_mat:
    :param file_out:
    :return:
    """

    # TODO: deal with mask

    # initialization
    failed_transfo = 0  # by default, failed matrix is 0 (i.e., no failure)
    file_mask = param.fname_mask

    # get metric radius (if MeanSquares, CC) or nb bins (if MI)
    if param.metric == 'MI':
        metric_radius = '16'
    else:
        metric_radius = '4'

    # If orientation is sagittal, we need to do a couple of things...
    im_data = Image(file_src)
    if im_data.orientation[2] in 'LR':
        im = Image(file_src)
        # reorient to RPI because ANTs algo will assume that the 3rd dim is along the S-I axis (where we want the
        # regularization)
        native_orientation = im.orientation
        im.change_orientation('RPI')
        # since we are dealing with a 2D slice, we need to pad (by copying the same slice) because this ANTs function
        # only accepts 3D input
        im_concat = concat_data([im, im, im, im, im], 0, squeeze_data=False)  # TODO: do it more elegantly inside the list
        file_src_concat = sct.add_suffix(file_src, '_rpi_concat')
        im_concat.save(file_src_concat)
        # and we need to do the same thing with the target (if not already done at the previous iteration)
        file_dest_concat = sct.add_suffix(file_dest, '_rpi_concat')
        if not os.path.isfile(file_dest_concat):
            im_dest = Image(file_dest)
            im_dest.change_orientation('RPI')
            im_dest_concat = concat_data([im_dest, im_dest, im_dest, im_dest, im_dest], 0, squeeze_data=False)
            im_dest_concat.save(file_dest_concat)
        # and the same thing with the mask (if there is one)
        if not param.fname_mask == '':
            file_mask_concat = 'mask_rpi_concat.nii.gz'
            if not os.path.isfile(file_mask_concat):
                im_mask = Image(param.fname_mask)
                im_mask.change_orientation('RPI')
                im_mask_concat = concat_data([im_mask, im_mask, im_mask, im_mask, im_mask], 0, squeeze_data=False)
                im_mask_concat.save(file_mask_concat)
        # update variables
        file_src = file_src_concat
        file_dest = file_dest_concat
        file_out_concat = sct.add_suffix(file_src, '_moco')
    else:
        file_out_concat = file_out
        file_mask_concat = file_mask

    # register file_src to file_dest
    if param.todo == 'estimate' or param.todo == 'estimate_and_apply':
        cmd = ['isct_antsSliceRegularizedRegistration',
               '--polydegree', param.poly,
               '--transform', 'Translation[%s]' %param.gradStep,
               '--metric', param.metric + '[' + file_dest + ',' + file_src + ',1,' + metric_radius + ',Regular,' + param.sampling + ']',
               '--iterations', param.iter,
               '--shrinkFactors', '1',
               '--smoothingSigmas', param.smooth,
               '--verbose', '1',
               '--output', '[' + file_mat + ',' + file_out_concat + ']']
        cmd += sct.get_interpolation('isct_antsSliceRegularizedRegistration', param.interp)
        if not file_mask_concat == '':
            cmd += ['--mask', file_mask_concat]
        status, output = sct.run(cmd, param.verbose)

    if param.todo == 'apply':
        sct_apply_transfo.main(args=['-i', file_src,
                                     '-d', file_dest,
                                     '-w', file_mat + 'Warp.nii.gz',
                                     '-o', file_out_concat,
                                     '-x', param.interp])
    #     cmd = ['sct_apply_transfo',
    #      '-i', file_src,
    #      '-d', file_dest,
    #      '-w', file_mat + 'Warp.nii.gz',
    #      '-o', file_out,
    #      '-x', param.interp]
    # # run the stuff
    # status, output = sct.run(cmd, param.verbose)

    # check if output file exists
    if not os.path.isfile(file_out_concat):
        # sct.printv(output, verbose, 'error')
        sct.printv('WARNING in ' + os.path.basename(__file__) + ': No output. Maybe related to improper calculation of '
                                                                'mutual information. Either the mask you provided is '
                                                                'too small, or the subject moved a lot. If you see too '
                                                                'many messages like this try with a bigger mask. '
                                                                'Using previous transformation for this volume (if it'
                                                                'exists).', param.verbose, 'warning')
        failed_transfo = 1

    # TODO: if sagittal, remove x values from mat, remove concat and put back in original orientation
    if im_data.orientation[2] in 'LR':
        im_out = Image(file_out_concat)
        im_out.change_orientation(native_orientation)
        im_out.data = im_out.data[:, :, 3]
        im_out.data = np.expand_dims(im_out.data, 2)  # need to have 3D data because target is also 3D (even though last dim is a singleton)
        im_out.save(file_out)

    # return status of failure
    return failed_transfo


# #=======================================================================================================================
# # check_transformation_absurdity:  find outliers
# #=======================================================================================================================
# def check_transformation_absurdity(file_mat):
#
#     # init param
#     failed_transfo = 0
#
#     file = open(file_mat)
#     M_transform = np.loadtxt(file)
#     file.close()
#
#     if abs(M_transform[0, 3]) > 10 or abs(M_transform[1, 3]) > 10 or abs(M_transform[2, 3]) > 10 or abs(M_transform[3, 3]) > 10:
#         failed_transfo = 1
#         sct.printv('  WARNING: This tranformation matrix is absurd, try others parameters (Gaussian mask, group size, ...)', 1, 'warning')
#
#     return failed_transfo


#=======================================================================================================================
# spline
#=======================================================================================================================
def spline(folder_mat, nt, nz, verbose, index_b0 = [], graph=0):

    sct.printv('\n\n\n------------------------------------------------------------------------------', verbose)
    sct.printv('Spline Regularization along T: Smoothing Patient Motion...', verbose)

    file_mat = [[[] for i in range(nz)] for i in range(nt)]
    for it in range(nt):
        for iz in range(nz):
            file_mat[it][iz] = os.path.join(folder_mat, "mat.T") + str(it) + '_Z' + str(iz) + '.txt'

    # Copying the existing Matrices to another folder
    old_mat = os.path.join(folder_mat, "old")
    if not os.path.exists(old_mat):
        os.makedirs(old_mat)
    # TODO
    for mat in glob.glob(os.path.join(folder_mat, '*.txt')):
        sct.copy(mat, old_mat)

    sct.printv('\nloading matrices...', verbose)
    X = [[[] for i in range(nt)] for i in range(nz)]
    Y = [[[] for i in range(nt)] for i in range(nz)]
    X_smooth = [[[] for i in range(nt)] for i in range(nz)]
    Y_smooth = [[[] for i in range(nt)] for i in range(nz)]
    for iz in range(nz):
        for it in range(nt):
            file =  open(file_mat[it][iz])
            Matrix = np.loadtxt(file)
            file.close()

            X[iz][it] = Matrix[0, 3]
            Y[iz][it] = Matrix[1, 3]

    # Generate motion splines
    sct.printv('\nGenerate motion splines...', verbose)
    T = np.arange(nt)
    if graph:
        import pylab as pl

    for iz in range(nz):

#        frequency = scipy.fftpack.fftfreq(len(X[iz][:]), d=1)
#        spectrum = np.abs(scipy.fftpack.fft(X[iz][:], n=None, axis=-1, overwrite_x=False))
#        Wn = np.amax(frequency)/10
#        N = 5              #Order of the filter
#        b, a = scipy.signal.iirfilter(N, Wn, rp=None, rs=None, btype='low', analog=False, ftype='butter', output='ba')
#        X_smooth[iz][:] = scipy.signal.filtfilt(b, a, X[iz][:], axis=-1, padtype=None)

        spline = scipy.interpolate.UnivariateSpline(T, X[iz][:], w=None, bbox=[None, None], k=3, s=None)
        X_smooth[iz][:] = spline(T)

        if graph:
            pl.plot(T, X_smooth[iz][:], label='spline_smoothing')
            pl.plot(T, X[iz][:], marker='*', linestyle='None', label='original_val')
            if len(index_b0) != 0:
                T_b0 = [T[i_b0] for i_b0 in index_b0]
                X_b0 = [X[iz][i_b0] for i_b0 in index_b0]
                pl.plot(T_b0, X_b0, marker='D', linestyle='None', color='k', label='b=0')
            pl.title('X')
            pl.grid()
            pl.legend()
            pl.show()

#        frequency = scipy.fftpack.fftfreq(len(Y[iz][:]), d=1)
#        spectrum = np.abs(scipy.fftpack.fft(Y[iz][:], n=None, axis=-1, overwrite_x=False))
#        Wn = np.amax(frequency)/10
#        N = 5              #Order of the filter
#        b, a = scipy.signal.iirfilter(N, Wn, rp=None, rs=None, btype='low', analog=False, ftype='butter', output='ba')
#        Y_smooth[iz][:] = scipy.signal.filtfilt(b, a, Y[iz][:], axis=-1, padtype=None)

        spline = scipy.interpolate.UnivariateSpline(T, Y[iz][:], w=None, bbox=[None, None], k=3, s=None)
        Y_smooth[iz][:] = spline(T)

        if graph:
            pl.plot(T, Y_smooth[iz][:], label='spline_smoothing')
            pl.plot(T, Y[iz][:], marker='*', linestyle='None', label='original_val')
            if len(index_b0) != 0:
                T_b0 = [T[i_b0] for i_b0 in index_b0]
                Y_b0 = [Y[iz][i_b0] for i_b0 in index_b0]
                pl.plot(T_b0, Y_b0, marker='D', linestyle='None', color='k', label='b=0')
            pl.title('Y')
            pl.grid()
            pl.legend()
            pl.show()

    # Storing the final Matrices
    sct.printv('\nStoring the final Matrices...', verbose)
    for iz in range(nz):
        for it in range(nt):
            file =  open(file_mat[it][iz])
            Matrix = np.loadtxt(file)
            file.close()

            Matrix[0, 3] = X_smooth[iz][it]
            Matrix[1, 3] = Y_smooth[iz][it]

            file =  open(file_mat[it][iz], 'w')
            np.savetxt(file_mat[it][iz], Matrix, fmt="%s", delimiter='  ', newline='\n')
            file.close()

    sct.printv('\n...Done. Patient motion has been smoothed', verbose)
    sct.printv('------------------------------------------------------------------------------\n', verbose)


#=======================================================================================================================
# combine_matrix
#=======================================================================================================================
def combine_matrix(param):

    # required fields
    # param.mat_2_combine
    # param.mat_final
    # param.verbose

    sct.printv('\nCombine matrices...', param.verbose)
    # list all mat files in source mat folder
    m2c_fnames = [fname for fname in os.listdir(param.mat_2_combine) if os.path.isfile(os.path.join(param.mat_2_combine, fname))]
    # loop across files
    for fname in m2c_fnames:
        if os.path.isfile(os.path.join(param.mat_final, fname)):
            # read source matrix
            file = open(os.path.join(param.mat_2_combine, fname))
            Matrix_m2c = np.loadtxt(file)
            file.close()
            # read destination matrix
            file = open(os.path.join(param.mat_final, fname))
            Matrix_f = np.loadtxt(file)
            file.close()
            # initialize final matrix
            Matrix_final = np.identity(4)
            # multiplies rotation matrix (3x3)
            Matrix_final[0:3, 0:3] = Matrix_f[0:3, 0:3] * Matrix_m2c[0:3, 0:3]
            # add translations matrix (3x1)
            Matrix_final[0, 3] = Matrix_f[0, 3] + Matrix_m2c[0, 3]
            Matrix_final[1, 3] = Matrix_f[1, 3] + Matrix_m2c[1, 3]
            Matrix_final[2, 3] = Matrix_f[2, 3] + Matrix_m2c[2, 3]
            # write final matrix (overwrite destination)
            file = open(os.path.join(param.mat_final, fname), 'w')
            np.savetxt(os.path.join(param.mat_final, fname), Matrix_final, fmt="%s", delimiter='  ', newline='\n')
            file.close()

#
# #=======================================================================================================================
# # gauss2d: creates a 2D Gaussian Function
# #=======================================================================================================================
# def gauss2d(dims, sigma, center):
#     x = np.zeros((dims[0],dims[1]))
#     y = np.zeros((dims[0],dims[1]))
#
#     for i in range(dims[0]):
#         x[i,:] = i+1
#     for i in range(dims[1]):
#         y[:,i] = i+1
#
#     xc = center[0]
#     yc = center[1]
#
#     return np.exp(-(((x-xc)**2)/(2*(sigma[0]**2)) + ((y-yc)**2)/(2*(sigma[1]**2))))
