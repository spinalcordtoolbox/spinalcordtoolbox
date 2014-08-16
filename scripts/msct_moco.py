#!/usr/bin/env python
#########################################################################################
#
# List of functions for moco.
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Karun Raju, Tanguy Duval, Julien Cohen-Adad
# Modified: 2014-08-14
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os
import sys
import commands
import numpy as np
import sct_utils as sct


#=======================================================================================================================
# sct_moco Function
#=======================================================================================================================
def moco(param):

    # Initialization
    file_schedule = '/flirtsch/schedule_TxTy_2mmScale.sch'
    fsloutput = 'export FSLOUTPUTTYPE=NIFTI; ' # for faster processing, all outputs are in NIFTI

    # get path of the toolbox
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')

    #Different parameters
    fname_data = param.fname_data
    fname_target = param.fname_target
    mat_final = param.mat_final
    todo = param.todo
    suffix = param.suffix
    mask_size = param.mask_size
    #program = param.program
    cost_function_flirt = param.cost_function_flirt
    interp = param.interp
    merge_back = param.merge_back
    verbose = param.verbose
    slicewise = 1  # TODO: in the future, enables non-slicewise

    # print arguments
    sct.printv('\nInput parameters:', param.verbose)
    sct.printv('  Input file ............'+param.fname_data, param.verbose)
    sct.printv('  Reference file ........'+param.fname_target, param.verbose)
    sct.printv('  Centerline file .......'+param.fname_centerline, param.verbose)
    sct.printv('  Schedule file .........'+file_schedule, param.verbose)
    sct.printv('  Method: ...............'+param.todo, param.verbose)

    # check existence of input files
    sct.printv('\nCheck file existence...', verbose)
    sct.check_file_exist(fname_data,verbose)
    if todo != 'apply':
        sct.check_file_exist(fname_target, verbose)

    # Extract path, file and extension
    path_data, file_data, ext_data = sct.extract_fname(fname_data)
    target_path_data, target_file_data, target_ext_data = sct.extract_fname(fname_target)

    #Schedule file for FLIRT
    schedule_file = path_sct+file_schedule

    if todo == 'estimate':
        if param.mat_moco == '':
            folder_mat = 'mat_moco/'
        else:
            folder_mat = param.mat_moco + '/'
    elif todo == 'estimate_and_apply':
        if param.mat_moco == '':
            folder_mat = 'mat_tmp/'
        else:
            folder_mat = param.mat_moco + '/'
    else:
        folder_mat = mat_final

    # create folder for mat files
    if not os.path.exists(folder_mat):
        os.makedirs(folder_mat)

    # Get size of data
    sct.printv('\nGet dimensions data...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_data)
    sct.printv(('.. '+str(nx)+' x '+str(ny)+' x '+str(nz)+' x '+str(nt)), verbose)

    # split along T dimension
    fname_data_splitT = file_data + '_T'
    sct.run(fsloutput + 'fslsplit ' + fname_data + ' ' + fname_data_splitT, verbose)

    #SLICE-by-SLICE MOTION CORRECTION
    sct.printv('\nMotion correction...', verbose)
    #split target data along Z
    fname_data_ref_splitZ = target_file_data + '_Z'

    sct.run(fsloutput + 'fslsplit ' + fname_target + ' ' + fname_data_ref_splitZ + ' -z', verbose)

    #Generate Gaussian Mask
    fslmask = []
    if mask_size > 0:
        import nibabel
        sigma = np.array([mask_size/px, mask_size/py])
        dims = np.array([nx, ny, nz, nt])
        data = nibabel.load((fname_data_ref_splitZ + '0000.nii'))
        hdr = data.get_header()
        hdr.set_data_dtype('uint8')  # set imagetype to uint8

        if param.fname_centerline == '':
            import math
            center = np.array([math.ceil(nx/2), math.ceil(ny/2), math.ceil(nz/2), math.ceil(nt/2)])
            fname_mask = 'gaussian_mask_in'
            M_mask = gauss2d(dims, sigma, center)
            # Write NIFTI volumes
            img = nibabel.Nifti1Image(M_mask, None, hdr)
            nibabel.save(img,(fname_mask+'.nii'))
            for iZ in range(nz):
                fslmask.append(' -inweight ' + fname_mask + ' -refweight ' + fname_mask)
            # sct.printv(('\n.. File created: '+fname_mask),verbose)
        else:
            centerline = nibabel.load(param.fname_centerline)
            data_centerline = centerline.get_data()
            cx, cy, cz = np.where(data_centerline > 0)
            arg = np.argsort(cz)
            cz = cz[arg]
            cx = cx[arg]
            cy = cy[arg]
            fname_mask = 'gaussian_mask_in'
            for iZ in range(nz):
                center = np.array([cx[iZ], cy[iZ]])
                M_mask = gauss2d(dims, sigma, center)
                # Write NIFTI volumes
                img = nibabel.Nifti1Image(M_mask, None, hdr)
                nibabel.save(img,(fname_mask+str(iZ)+'.nii'))
                fslmask.append(' -inweight ' + fname_mask+str(iZ) + ' -refweight ' + fname_mask+str(iZ))
                # sct.printv(('\n.. File created: '+(fname_mask+str(iZ))),verbose)

            #Merging all masks
            cmd = 'fslmerge -z ' + path_data + 'mask '
            for iZ in range(nz):
                cmd = cmd + fname_mask+str(iZ)+' '
            status, output = sct.run(cmd, verbose)
    else:
        for iZ in range(nz):
            fslmask.append('')
    index = np.arange(nt)

    # MOTION CORRECTION
    nb_fails = 0
    fail_mat = np.zeros((nt,nz))
    fname_data_splitT_num = []
    fname_data_splitT_moco_num = []
    fname_data_splitT_splitZ_num = [[[] for i in range(nz)] for i in range(nt)]
    fname_data_splitT_splitZ_moco_num = [[[] for i in range(nz)] for i in range(nt)]
    fname_mat = [[[] for i in range(nz)] for i in range(nt)]

    for indice_index in range(nt):
        iT = index[indice_index]
        sct.printv(('\nVolume '+str((iT+1))+'/'+str(nt)+':'), verbose)
        sct.printv('--------------------', verbose)

        fname_data_splitT_num.append(fname_data_splitT + str(iT).zfill(4))
        fname_data_splitT_moco_num.append(file_data + suffix + '_T' + str(iT).zfill(4))

        if slicewise:
            # split data along Z
            sct.printv('Split data along Z...', verbose)
            fname_data_splitT_splitZ = fname_data_splitT_num[iT] + '_Z'
            cmd = fsloutput + 'fslsplit ' + fname_data_splitT_num[iT] + ' ' + fname_data_splitT_splitZ + ' -z'
            status, output = sct.run(cmd, verbose)

            fname_data_ref_splitZ_num = []
            for iZ in range(nz):
                fname_data_splitT_splitZ_num[iT][iZ] = fname_data_splitT_splitZ + str(iZ).zfill(4)
                fname_data_splitT_splitZ_moco_num[iT][iZ] = fname_data_splitT_splitZ_num[iT][iZ] + suffix
                fname_data_ref_splitZ_num.append(fname_data_ref_splitZ + str(iZ).zfill(4))
                fname_mat[iT][iZ] = folder_mat + 'mat.T' + str(iT) + '_Z' + str(iZ) + '.txt'

                if todo == 'estimate':
                    cmd = fsloutput+'flirt -schedule '+schedule_file+' -in '+fname_data_splitT_splitZ_num[iT][iZ]+' -ref '+fname_data_ref_splitZ_num[iZ]+' -omat '+fname_mat[iT][iZ]+' -out '+fname_data_splitT_splitZ_moco_num[iT][iZ]+' -cost '+cost_function_flirt+fslmask[iZ]+' -interp '+interp

                if todo == 'apply':
                    cmd = fsloutput + 'flirt -in ' + fname_data_splitT_splitZ_num[iT][iZ] + ' -ref ' + fname_data_ref_splitZ_num[iZ] + ' -applyxfm -init ' + fname_mat[iT][iZ] + ' -out ' + fname_data_splitT_splitZ_moco_num[iT][iZ] + ' -interp ' + interp

                if todo == 'estimate_and_apply':
                    cmd = fsloutput+'flirt -schedule '+schedule_file+ ' -in '+fname_data_splitT_splitZ_num[iT][iZ]+' -ref '+ fname_data_ref_splitZ_num[iZ] +' -out '+fname_data_splitT_splitZ_moco_num[iT][iZ]+' -omat '+fname_mat[iT][iZ]+' -cost '+cost_function_flirt+fslmask[iZ]+' -interp '+interp

                sct.run(cmd, verbose)

                #Check transformation absurdity
                file = open(fname_mat[iT][iZ])
                M_transform = np.loadtxt(file)
                file.close()

                if abs(M_transform[0, 3]) > 10 or abs(M_transform[1, 3]) > 10 or abs(M_transform[2, 3]) > 10 or abs(M_transform[3, 3]) > 10:
                    nb_fails = nb_fails + 1
                    fail_mat[iT, iZ] = 1
                    sct.printv('  WARNING: This tranformation matrix is absurd, try others parameters (Gaussian mask, cost_function, group size, ...)', verbose, 'warning')

            # Merge data along Z
            if todo != 'estimate':
                if merge_back == 1:
                    sct.printv('Concatenate along Z...', verbose)
                    cmd = fsloutput + 'fslmerge -z ' + fname_data_splitT_moco_num[iT]
                    for iZ in range(nz):
                        cmd = cmd + ' ' + fname_data_splitT_splitZ_moco_num[iT][iZ]
                    sct.run(cmd,verbose)


    #Replace failed transformation matrix to the closest good one

    fT, fZ = np.where(fail_mat==1)
    gT, gZ = np.where(fail_mat==0)

    for iT in range(len(fT)):
        sct.printv(('\nReplace failed matrix T'+str(fT[iT])+' Z'+str(fZ[iT])+'...'),verbose)

        # rename failed matrix
        cmd = 'mv ' + fname_mat[fT[iT]][fZ[iT]] + ' ' + fname_mat[fT[iT]][fZ[iT]] + '_failed'
        status, output = sct.run(cmd,verbose)

        good_Zindex = np.where(gZ == fZ[iT])
        good_index = gT[good_Zindex]

        I = np.amin(abs(good_index-fT[iT]))
        cmd = 'cp ' + fname_mat[good_index[I]][fZ[iT]] + ' ' + fname_mat[fT[iT]][fZ[iT]]
        status, output = sct.run(cmd,verbose)

    # Merge data along T
    fname_data_moco = file_data + suffix
    if todo != 'estimate':
        if merge_back == 1:
            sct.printv('\nMerge data back along T...', verbose)
            cmd = fsloutput + 'fslmerge -t ' + fname_data_moco
            for indice_index in range(len(index)):
                cmd = cmd + ' ' + fname_data_splitT_moco_num[indice_index]
            sct.run(cmd, verbose)

    if todo == 'estimate_and_apply':
        if param.mat_moco == '':
            sct.printv('\nDelete temporary files...', verbose)
            sct.run('rm -rf '+folder_mat, verbose)


#=======================================================================================================================
# spline
#=======================================================================================================================
def spline(folder_mat,nt,nz,verbose,index_b0 = [],graph=0):
    # get path of the toolbox
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
    # append path that contains scripts, to be able to load modules
    sys.path.append(path_sct + '/scripts')
    import sct_utils as sct

    sct.printv('\n\n\n------------------------------------------------------------------------------',verbose)
    sct.printv('Spline Regularization along T: Smoothing Patient Motion...',verbose)

    fname_mat = [[[] for i in range(nz)] for i in range(nt)]
    for iT in range(nt):
        for iZ in range(nz):
            fname_mat[iT][iZ] = folder_mat + 'mat.T' + str(iT) + '_Z' + str(iZ) + '.txt'

    #Copying the existing Matrices to another folder
    old_mat = folder_mat + 'old/'
    if not os.path.exists(old_mat): os.makedirs(old_mat)
    cmd = 'cp ' + folder_mat + '*.txt ' + old_mat
    status, output = sct.run(cmd, verbose)

    sct.printv('\nloading matrices...',verbose)
    X = [[[] for i in range(nt)] for i in range(nz)]
    Y = [[[] for i in range(nt)] for i in range(nz)]
    X_smooth = [[[] for i in range(nt)] for i in range(nz)]
    Y_smooth = [[[] for i in range(nt)] for i in range(nz)]
    for iZ in range(nz):
        for iT in range(nt):
            file =  open(fname_mat[iT][iZ])
            Matrix = np.loadtxt(file)
            file.close()

            X[iZ][iT] = Matrix[0,3]
            Y[iZ][iT] = Matrix[1,3]

    # Generate motion splines
    sct.printv('\nGenerate motion splines...',verbose)
    T = np.arange(nt)
    if graph:
        import pylab as pl

    for iZ in range(nz):

#        frequency = scipy.fftpack.fftfreq(len(X[iZ][:]), d=1)
#        spectrum = np.abs(scipy.fftpack.fft(X[iZ][:], n=None, axis=-1, overwrite_x=False))
#        Wn = np.amax(frequency)/10
#        N = 5              #Order of the filter
#        b, a = scipy.signal.iirfilter(N, Wn, rp=None, rs=None, btype='low', analog=False, ftype='butter', output='ba')
#        X_smooth[iZ][:] = scipy.signal.filtfilt(b, a, X[iZ][:], axis=-1, padtype=None)

        spline = scipy.interpolate.UnivariateSpline(T, X[iZ][:], w=None, bbox=[None, None], k=3, s=None)
        X_smooth[iZ][:] = spline(T)

        if graph:
            pl.plot(T,X_smooth[iZ][:],label='spline_smoothing')
            pl.plot(T,X[iZ][:],marker='*',linestyle='None',label='original_val')
            if len(index_b0)!=0:
                T_b0 = [T[i_b0] for i_b0 in index_b0]
                X_b0 = [X[iZ][i_b0] for i_b0 in index_b0]
                pl.plot(T_b0,X_b0,marker='D',linestyle='None',color='k',label='b=0')
            pl.title('X')
            pl.grid()
            pl.legend()
            pl.show()

#        frequency = scipy.fftpack.fftfreq(len(Y[iZ][:]), d=1)
#        spectrum = np.abs(scipy.fftpack.fft(Y[iZ][:], n=None, axis=-1, overwrite_x=False))
#        Wn = np.amax(frequency)/10
#        N = 5              #Order of the filter
#        b, a = scipy.signal.iirfilter(N, Wn, rp=None, rs=None, btype='low', analog=False, ftype='butter', output='ba')
#        Y_smooth[iZ][:] = scipy.signal.filtfilt(b, a, Y[iZ][:], axis=-1, padtype=None)

        spline = scipy.interpolate.UnivariateSpline(T, Y[iZ][:], w=None, bbox=[None, None], k=3, s=None)
        Y_smooth[iZ][:] = spline(T)

        if graph:
            pl.plot(T,Y_smooth[iZ][:],label='spline_smoothing')
            pl.plot(T,Y[iZ][:],marker='*', linestyle='None',label='original_val')
            if len(index_b0)!=0:
                T_b0 = [T[i_b0] for i_b0 in index_b0]
                Y_b0 = [Y[iZ][i_b0] for i_b0 in index_b0]
                pl.plot(T_b0,Y_b0,marker='D',linestyle='None',color='k',label='b=0')
            pl.title('Y')
            pl.grid()
            pl.legend()
            pl.show()

    #Storing the final Matrices
    sct.printv('\nStoring the final Matrices...',verbose)
    for iZ in range(nz):
        for iT in range(nt):
            file =  open(fname_mat[iT][iZ])
            Matrix = np.loadtxt(file)
            file.close()

            Matrix[0,3] = X_smooth[iZ][iT]
            Matrix[1,3] = Y_smooth[iZ][iT]

            file =  open(fname_mat[iT][iZ],'w')
            np.savetxt(fname_mat[iT][iZ], Matrix, fmt="%s", delimiter='  ', newline='\n')
            file.close()

    sct.printv('\n...Done. Patient motion has been smoothed',verbose)
    sct.printv('------------------------------------------------------------------------------\n',verbose)


#=======================================================================================================================
# moco_combine_matrix
#=======================================================================================================================
def combine_matrix(param):

    sct.printv('\nCombine matrices...', param.verbose)
    m2c_fnames = [ fname for fname in os.listdir(param.mat_2_combine) if os.path.isfile(os.path.join(param.mat_2_combine,fname)) ]
    for fname in m2c_fnames:
        if os.path.isfile(os.path.join(param.mat_final, fname)):
            file =  open(os.path.join(param.mat_2_combine, fname))
            Matrix_m2c = np.loadtxt(file)
            file.close()

            file =  open(os.path.join(param.mat_final, fname))
            Matrix_f = np.loadtxt(file)
            file.close()
            Matrix_final = np.identity(4)
            Matrix_final[0:3,0:3] = Matrix_f[0:3,0:3]*Matrix_m2c[0:3,0:3]
            Matrix_final[0,3] = Matrix_f[0,3] + Matrix_m2c[0,3]
            Matrix_final[1,3] = Matrix_f[1,3] + Matrix_m2c[1,3]

            file =  open(os.path.join(param.mat_final,fname),'w')
            np.savetxt(os.path.join(param.mat_final, fname), Matrix_final, fmt="%s", delimiter='  ', newline='\n')
            file.close()


#=======================================================================================================================
# gauss2d: creates a 2D Gaussian Function
#=======================================================================================================================
def gauss2d(dims, sigma, center):
    x = np.zeros((dims[0],dims[1]))
    y = np.zeros((dims[0],dims[1]))

    for i in range(dims[0]):
        x[i,:] = i+1
    for i in range(dims[1]):
        y[:,i] = i+1

    xc = center[0]
    yc = center[1]

    return np.exp(-(((x-xc)**2)/(2*(sigma[0]**2)) + ((y-yc)**2)/(2*(sigma[1]**2))))
