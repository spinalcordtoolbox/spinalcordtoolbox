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

# TODO: params for ANTS: CC/MI, shrink fact, nb_it
# TODO: use mask
# TODO: unpad after applying transfo
# TODO: do not output inverse warp for ants
# TODO: ants: explore optin  --float  for faster computation

import os
import sys
import commands
import numpy as np
import sct_utils as sct
from msct_image import Image
from sct_image import split_data


#=======================================================================================================================
# moco Function
#=======================================================================================================================
def moco(param):

    # retrieve parameters
    fsloutput = 'export FSLOUTPUTTYPE=NIFTI; '  # for faster processing, all outputs are in NIFTI
    file_data = param.file_data
    file_target = param.file_target
    folder_mat = sct.slash_at_the_end(param.mat_moco, 1)  # output folder of mat file
    todo = param.todo
    suffix = param.suffix
    #file_schedule = param.file_schedule
    verbose = param.verbose
    ext = '.nii'

    # get path of the toolbox
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')

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
    sct.printv('\nGet dimensions data...', verbose)
    data_im = Image(file_data + ext)
    nx, ny, nz, nt, px, py, pz, pt = data_im.dim
    sct.printv(('.. ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz) + ' x ' + str(nt)), verbose)

    # copy file_target to a temporary file
    sct.printv('\nCopy file_target to a temporary file...', verbose)
    sct.run('cp ' + file_target + ext + ' target.nii')
    file_target = 'target'

    # Split data along T dimension
    sct.printv('\nSplit data along T dimension...', verbose)
    data_split_list = split_data(data_im, dim=3)
    for im in data_split_list:
        im.save()
    file_data_splitT = file_data + '_T'

    # Motion correction: initialization
    index = np.arange(nt)
    file_data_splitT_num = []
    file_data_splitT_moco_num = []
    failed_transfo = [0 for i in range(nt)]
    file_mat = [[] for i in range(nt)]

    # Motion correction: Loop across T
    for indice_index in range(nt):

        # create indices and display stuff
        it = index[indice_index]
        file_data_splitT_num.append(file_data_splitT + str(it).zfill(4))
        file_data_splitT_moco_num.append(file_data + suffix + '_T' + str(it).zfill(4))
        sct.printv(('\nVolume ' + str((it)) + '/' + str(nt - 1) + ':'), verbose)
        file_mat[it] = folder_mat + 'mat.T' + str(it)

        # run 3D registration
        failed_transfo[it] = register(param, file_data_splitT_num[it], file_target, file_mat[it], file_data_splitT_moco_num[it])

        # average registered volume with target image
        # N.B. use weighted averaging: (target * nb_it + moco) / (nb_it + 1)
        if param.iterative_averaging and indice_index < 10 and failed_transfo[it] == 0:
            sct.run('sct_maths -i ' + file_target + ext + ' -mul ' + str(indice_index + 1) + ' -o ' + file_target + ext)
            sct.run('sct_maths -i ' + file_target + ext + ' -add ' + file_data_splitT_moco_num[it] + ext + ' -o ' + file_target + ext)
            sct.run('sct_maths -i ' + file_target + ext + ' -div ' + str(indice_index + 2) + ' -o ' + file_target + ext)

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
            sct.run('cp ' + file_mat[gT[index_good]] + 'Warp.nii.gz' + ' ' + file_mat[fT[it]] + 'Warp.nii.gz')
            # apply transformation
            sct.run('sct_apply_transfo -i ' + file_data_splitT_num[fT[it]] + '.nii -d ' + file_target + '.nii -w ' + file_mat[fT[it]] + 'Warp.nii.gz' + ' -o ' + file_data_splitT_moco_num[fT[it]] + '.nii' + ' -x ' + param.interp, verbose)
        else:
            # exit program if no transformation exists.
            sct.printv('\nERROR in ' + os.path.basename(__file__) + ': No good transformation exist. Exit program.\n', verbose, 'error')
            sys.exit(2)

    # Merge data along T
    file_data_moco = file_data + suffix
    if todo != 'estimate':
        sct.printv('\nMerge data back along T...', verbose)
        from sct_image import concat_data
        # im_list = []
        fname_list = []
        for indice_index in range(len(index)):
            # im_list.append(Image(file_data_splitT_moco_num[indice_index] + ext))
            fname_list.append(file_data_splitT_moco_num[indice_index] + ext)
        im_out = concat_data(fname_list, 3)
        im_out.setFileName(file_data_moco + ext)
        im_out.save()

    # delete file target.nii (to avoid conflict if this function is run another time)
    sct.printv('\nRemove temporary file...', verbose)
    # os.remove('target.nii')
    sct.run('rm target.nii')


#=======================================================================================================================
# register:  registration of two volumes (or two images)
#=======================================================================================================================
def register(param, file_src, file_dest, file_mat, file_out):

    # initialization
    failed_transfo = 0  # by default, failed matrix is 0 (i.e., no failure)

    # get metric radius (if MeanSquares, CC) or nb bins (if MI)
    if param.metric == 'MI':
        metric_radius = '16'
    else:
        metric_radius = '4'

    # register file_src to file_dest
    if param.todo == 'estimate' or param.todo == 'estimate_and_apply':
        cmd = 'isct_antsSliceRegularizedRegistration' \
              ' -p ' + param.poly + \
              ' --transform Translation[' + param.gradStep + ']' \
              ' --metric ' + param.metric + '[' + file_dest + '.nii, ' + file_src + '.nii, 1, ' + metric_radius + ', Regular, ' + param.sampling + ']' \
              ' --iterations 5' \
              ' --shrinkFactors 1' \
              ' --smoothingSigmas ' + param.smooth + \
              ' --output [' + file_mat + ',' + file_out + '.nii]' \
              + sct.get_interpolation('isct_antsSliceRegularizedRegistration', param.interp)
        if not param.fname_mask == '':
            cmd += ' -x ' + param.fname_mask
    if param.todo == 'apply':
        cmd = 'sct_apply_transfo -i ' + file_src + '.nii -d ' + file_dest + '.nii -w ' + file_mat + 'Warp.nii.gz' + ' -o ' + file_out + '.nii' + ' -x ' + param.interp
    status, output = sct.run(cmd, param.verbose)

    # check if output file exists
    if not os.path.isfile(file_out + '.nii'):
        # sct.printv(output, verbose, 'error')
        sct.printv('WARNING in ' + os.path.basename(__file__) + ': Improper calculation of mutual information. Either the mask you provided is too small, or the subject moved a lot. If you see too many messages like this try with a bigger mask. Using previous transformation for this volume.', param.verbose, 'warning')
        failed_transfo = 1

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
    # get path of the toolbox
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
    # append path that contains scripts, to be able to load modules
    sys.path.append(path_sct + '/scripts')
    import sct_utils as sct

    sct.printv('\n\n\n------------------------------------------------------------------------------', verbose)
    sct.printv('Spline Regularization along T: Smoothing Patient Motion...', verbose)

    file_mat = [[[] for i in range(nz)] for i in range(nt)]
    for it in range(nt):
        for iz in range(nz):
            file_mat[it][iz] = folder_mat + 'mat.T' + str(it) + '_Z' + str(iz) + '.txt'

    # Copying the existing Matrices to another folder
    old_mat = folder_mat + 'old/'
    if not os.path.exists(old_mat):
        os.makedirs(old_mat)
    cmd = 'cp ' + folder_mat + '*.txt ' + old_mat
    status, output = sct.run(cmd, verbose)

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
