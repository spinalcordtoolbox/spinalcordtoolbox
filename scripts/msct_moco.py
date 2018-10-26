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
# TODO: add tests with sag and ax orientation, with -g 1 and 3, with mask (not covering all slices)
# TODO: make it a spinalcordtoolbox module with im as input
# TODO: params for ANTS: CC/MI, shrink fact, nb_it
# TODO: ants: explore optin  --float  for faster computation

from __future__ import absolute_import

import sys, os, glob
from tqdm import tqdm
import numpy as np
import scipy.interpolate

import sct_utils as sct
from sct_convert import convert
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
    verbose = param.verbose

    # other parameters
    ext = '.nii'
    file_mask = 'mask.nii'

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

    # copy file_target to a temporary file
    sct.printv('\nCopy file_target to a temporary file...', verbose)
    sct.copy(file_target + ext, 'target.nii')
    file_target = 'target'

    # If scan is sagittal, split src and target along Z (slice)
    if param.is_sagittal:
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
        # z-split mask (if exists)
        if not param.fname_mask == '':
            im_maskz_list = split_data(Image(file_mask), dim=dim_sag, squeeze_data=False)
            file_mask_splitZ = []
            for im_maskz in im_maskz_list:
                im_maskz.save()
                file_mask_splitZ.append(im_maskz.absolutepath)
        # initialize file list for output matrices
        file_mat = np.chararray([nz, nt],
                                itemsize=50)  # itemsize=50 is to accomodate relative path to matrix file name.

    # axial orientation
    else:
        file_data_splitZ = [file_data + ext]  # TODO: make it absolute like above
        file_target_splitZ = [file_target + ext]  # TODO: make it absolute like above
        # initialize file list for output matrices
        file_mat = np.chararray([1, nt],
                                itemsize=50)  # itemsize=50 is to accomodate relative path to matrix file name.
        # deal with mask
        if not param.fname_mask == '':
            convert(param.fname_mask, file_mask, squeeze_data=False)
            im_maskz_list = [Image(file_mask)]  # use a list with single element

    # Loop across file list, where each file is either a 2D volume (if sagittal) or a 3D volume (otherwise)
    # file_mat = tuple([[[] for i in range(nt)] for i in range(nz)])
    file_mat[:] = ''  # init
    file_data_splitZ_moco = []
    sct.printv('\nRegister. Loop across Z (note: there is only one Z if orientation is axial')
    for file in file_data_splitZ:
        iz = file_data_splitZ.index(file)
        # Split data along T dimension
        # sct.printv('\nSplit data along T dimension.', verbose)
        im_z = Image(file)
        list_im_zt = split_data(im_z, dim=3)
        file_data_splitZ_splitT = []
        for im_zt in list_im_zt:
            im_zt.save(verbose=0)
            file_data_splitZ_splitT.append(im_zt.absolutepath)
        # file_data_splitT = file_data + '_T'

        # Motion correction: initialization
        index = np.arange(nt)
        file_data_splitT_num = []
        file_data_splitZ_splitT_moco = []
        failed_transfo = [0 for i in range(nt)]

        # Motion correction: Loop across T
        for indice_index in tqdm(range(nt), unit='iter', unit_scale=False,
                                 desc="Z=" + str(iz) + "/" + str(len(file_data_splitZ)-1), ascii=True, ncols=80):

            # create indices and display stuff
            it = index[indice_index]
            file_mat[iz][it] = os.path.join(folder_mat, "mat.Z") + str(iz).zfill(4) + 'T' + str(it).zfill(4)
            file_data_splitZ_splitT_moco.append(sct.add_suffix(file_data_splitZ_splitT[it], '_moco'))
            # deal with masking
            if not param.fname_mask == '':
                input_mask = im_maskz_list[iz]
            else:
                input_mask = None
            # run 3D registration
            failed_transfo[it] = register(param, file_data_splitZ_splitT[it], file_target_splitZ[iz], file_mat[iz][it],
                                          file_data_splitZ_splitT_moco[it], im_mask=input_mask)

            # average registered volume with target image
            # N.B. use weighted averaging: (target * nb_it + moco) / (nb_it + 1)
            if param.iterAvg and indice_index < 10 and failed_transfo[it] == 0 and not param.todo == 'apply':
                im_targetz = Image(file_target_splitZ[iz])
                data_targetz = im_targetz.data
                data_mocoz = Image(file_data_splitZ_splitT_moco[it]).data
                data_targetz = (data_targetz * (indice_index + 1) + data_mocoz) / (indice_index + 2)
                im_targetz.data = data_targetz
                im_targetz.save(verbose=0)

        # Replace failed transformation with the closest good one
        fT = [i for i, j in enumerate(failed_transfo) if j == 1]
        gT = [i for i, j in enumerate(failed_transfo) if j == 0]
        for it in range(len(fT)):
            abs_dist = [np.abs(gT[i] - fT[it]) for i in range(len(gT))]
            if not abs_dist == []:
                index_good = abs_dist.index(min(abs_dist))
                sct.printv('  transfo #' + str(fT[it]) + ' --> use transfo #' + str(gT[index_good]), verbose)
                # copy transformation
                sct.copy(file_mat[iz][gT[index_good]] + 'Warp.nii.gz', file_mat[iz][fT[it]] + 'Warp.nii.gz')
                # apply transformation
                sct.run(["sct_apply_transfo",
                 "-i", file_data_splitZ_splitT[fT[it]],
                 "-d", file_target + ".nii",
                 "-w", file_mat[iz][fT[it]] + 'Warp.nii.gz',
                 "-o", file_data_splitZ_splitT_moco[fT[it]],
                 "-x", param.interp], verbose=0)
            else:
                # exit program if no transformation exists.
                sct.printv('\nERROR in ' + os.path.basename(__file__) + ': No good transformation exist. Exit program.\n', verbose, 'error')
                sys.exit(2)

        # Merge data along T
        file_data_splitZ_moco.append(sct.add_suffix(file, suffix))
        if todo != 'estimate':
            im_out = concat_data(file_data_splitZ_splitT_moco, 3)
            im_out.save(file_data_splitZ_moco[iz])

    # If sagittal, merge along Z
    if param.is_sagittal:
        im_out = concat_data(file_data_splitZ_moco, 2)
        im_out.save(file_data + suffix + ext)

    return file_mat


def register(param, file_src, file_dest, file_mat, file_out, im_mask=None):
    """
    Register two images by estimating slice-wise Tx and Ty transformations, which are regularized along Z. This function
    uses ANTs' isct_antsSliceRegularizedRegistration.
    :param param:
    :param file_src:
    :param file_dest:
    :param file_mat:
    :param file_out:
    :param im_mask: Image of mask, could be 2D or 3D
    :return:
    """

    # TODO: deal with mask

    # initialization
    failed_transfo = 0  # by default, failed matrix is 0 (i.e., no failure)
    do_registration = True

    # get metric radius (if MeanSquares, CC) or nb bins (if MI)
    if param.metric == 'MI':
        metric_radius = '16'
    else:
        metric_radius = '4'
    file_out_concat = file_out

    im_data = Image(file_src)  # TODO: pass argument to use antsReg instead of opening Image each time

    # register file_src to file_dest
    if param.todo == 'estimate' or param.todo == 'estimate_and_apply':
        # If orientation is sagittal, use antsRegistration in 2D mode
        # Note: the parameter --restrict-deformation is irrelevant with affine transfo
        if im_data.orientation[2] in 'LR':
            cmd = ['isct_antsRegistration',
                   '-d', '2',
                   '--transform', 'Affine[%s]' %param.gradStep,
                   '--metric', param.metric + '[' + file_dest + ',' + file_src + ',1,' + metric_radius + ',Regular,' + param.sampling + ']',
                   '--convergence', param.iter,
                   '--shrink-factors', '1',
                   '--smoothing-sigmas', param.smooth,
                   '--verbose', '1',
                   '--output', '[' + file_mat + ',' + file_out_concat + ']']
            cmd += sct.get_interpolation('isct_antsRegistration', param.interp)
            if im_mask is not None:
                # if user specified a mask, make sure there are non-null voxels in the image before running the registration
                if np.count_nonzero(im_mask.data):
                    cmd += ['--masks', im_mask.absolutepath]
                else:
                    # Mask only contains zeros. Copying the image instead of estimating registration.
                    sct.copy(file_src, file_out_concat, verbose=0)
                    do_registration = False
                    # TODO: create affine mat file with identity, in case used by -g 2
        # 3D mode
        else:
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
            if im_mask is not None:
                cmd += ['--mask', im_mask.absolutepath]
        # run command
        if do_registration:
            status, output = sct.run(cmd, verbose=0)

    elif param.todo == 'apply':
        sct_apply_transfo.main(args=['-i', file_src,
                                     '-d', file_dest,
                                     '-w', file_mat + 'Warp.nii.gz',
                                     '-o', file_out_concat,
                                     '-x', param.interp,
                                     '-v', '0'])

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

    # TODO: if sagittal, copy header (because ANTs screws it) and add singleton in 3rd dimension (for z-concatenation)
    if im_data.orientation[2] in 'LR' and do_registration:
        im_out = Image(file_out_concat)
        im_out.header = im_data.header
        im_out.data = np.expand_dims(im_out.data, 2)
        im_out.save(file_out, verbose=0)

    # return status of failure
    return failed_transfo


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
