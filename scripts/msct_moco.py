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

# TODO: with ants, pad at top and bottom
# TODO: make mask work with slicewise=0
# TODO: do not output inverse warp for ants
# TODO: ants: explore optin  --float  for faster computation

import os
import sys
import commands
import numpy as np
import sct_utils as sct


#=======================================================================================================================
# sct_moco Function
#=======================================================================================================================
def moco(param):

    # retrieve parameters
    fsloutput = 'export FSLOUTPUTTYPE=NIFTI; '  # for faster processing, all outputs are in NIFTI
    fname_data = param.fname_data
    fname_target = param.fname_target
    folder_mat = sct.slash_at_the_end(param.mat_moco, 1)  # output folder of mat file
    # mat_final = param.mat_final
    todo = param.todo
    suffix = param.suffix
    mask_size = param.mask_size
    program = param.program  # flirt, ants
    cost_function_flirt = param.cost_function_flirt
    file_schedule = param.file_schedule
    verbose = param.verbose
    slicewise = param.slicewise
    # ANTs parameters
    restrict_deformation = '1x1x0'

    # get path of the toolbox
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')

    # print arguments
    sct.printv('\nInput parameters:', param.verbose)
    sct.printv('  Input file ............'+fname_data, param.verbose)
    sct.printv('  Reference file ........'+fname_target, param.verbose)
    sct.printv('  Centerline file .......'+param.fname_centerline, param.verbose)
    sct.printv('  Program ...............'+program, param.verbose)
    sct.printv('  Slicewise .............'+str(slicewise), param.verbose)
    sct.printv('  Schedule file .........'+file_schedule, param.verbose)
    sct.printv('  Method ................'+todo, param.verbose)
    sct.printv('  Mask size .............'+str(mask_size), param.verbose)
    sct.printv('  Output mat folder .....'+folder_mat, param.verbose)

    # check existence of input files
    sct.printv('\nCheck file existence...', verbose)
    sct.check_file_exist(fname_data, verbose)
    sct.check_file_exist(fname_target, verbose)

    # Schedule file for FLIRT
    schedule_file = path_sct+file_schedule

    # create folder for mat files
    sct.create_folder(folder_mat)

    # get the right matrix extension depending on method
    if param.program == 'flirt':
        ext_mat = '.txt'  # affine matrix
    elif param.program == 'ants':
        ext_mat = '0Warp.nii.gz'  # warping field
    elif param.program == 'ants_affine':
        ext_mat = '0GenericAffine.mat'  # ITK affine matrix

    # get the right interpolation field depending on method
    interp = get_interpolation(param.program, param.interp)

    # Get size of data
    sct.printv('\nGet dimensions data...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_data)
    sct.printv(('.. '+str(nx)+' x '+str(ny)+' x '+str(nz)+' x '+str(nt)), verbose)

    # pad data (for ANTs)
    # TODO: check if need to pad also for the estimate_and_apply
    if program == 'ants' and todo == 'estimate':
        sct.printv('\nPad data (for ANTs)...', verbose)
        sct.run('sct_c3d '+fname_target+' -pad 0x0x3vox 0x0x3vox 0 -o '+fname_target+'_pad.nii')
        fname_target = fname_target+'_pad'

    # Split data along T dimension
    sct.printv('\nSplit data along T dimension...', verbose)
    fname_data_splitT = fname_data + '_T'
    sct.run(fsloutput + 'fslsplit ' + fname_data + ' ' + fname_data_splitT, verbose)

    # split target data along Z
    if slicewise:
        file_data_ref_splitZ = fname_target + '_Z'
        sct.run(fsloutput + 'fslsplit ' + fname_target + ' ' + file_data_ref_splitZ + ' -z', verbose)

    # Generate Gaussian Mask
    fslmask = []
    # TODO: make case that works for slicewise=0
    if mask_size > 0 and slicewise:
        import nibabel
        sigma = np.array([mask_size/px, mask_size/py])
        dims = np.array([nx, ny, nz, nt])
        data = nibabel.load((file_data_ref_splitZ + '0000.nii'))
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
            for iz in range(nz):
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
            for iz in range(nz):
                center = np.array([cx[iz], cy[iz]])
                M_mask = gauss2d(dims, sigma, center)
                # Write NIFTI volumes
                img = nibabel.Nifti1Image(M_mask, None, hdr)
                nibabel.save(img,(fname_mask+str(iz)+'.nii'))
                fslmask.append(' -inweight ' + fname_mask+str(iz) + ' -refweight ' + fname_mask+str(iz))

            #Merging all masks
            cmd = 'fslmerge -z mask '
            for iz in range(nz):
                cmd = cmd + fname_mask+str(iz)+' '
            status, output = sct.run(cmd, verbose)
    else:
        for iz in range(nz):
            fslmask.append('')  # TODO: adapt if volume-based moco
    index = np.arange(nt)

    # Motion correction: initialization
    file_data_splitT_num = []
    file_data_splitT_moco_num = []
    if slicewise:
        fail_mat = np.zeros((nt, nz))
        file_data_splitT_splitZ_num = [[[] for i in range(nz)] for i in range(nt)]
        file_data_splitT_splitZ_moco_num = [[[] for i in range(nz)] for i in range(nt)]
        file_mat = [[[] for i in range(nz)] for i in range(nt)]
    else:
        fail_mat = np.zeros((nt))
        file_mat = [[] for i in range(nt)]

    # Motion correction: Loop across T
    for indice_index in range(nt):

        # create indices and display stuff
        it = index[indice_index]
        file_data_splitT_num.append(fname_data_splitT + str(it).zfill(4))
        file_data_splitT_moco_num.append(fname_data + suffix + '_T' + str(it).zfill(4))
        sct.printv(('\nVolume '+str((it+1))+'/'+str(nt)+':'), verbose)

        # pad data (for ANTs)
        # TODO: check if need to pad also for the estimate_and_apply
        if program == 'ants' and todo == 'estimate':
            sct.run('sct_c3d '+file_data_splitT_num[it]+' -pad 0x0x3vox 0x0x3vox 0 -o '+file_data_splitT_num[it]+'_pad.nii')
            file_data_splitT_num[it] = file_data_splitT_num[it]+'_pad'

        # Slice-by-slice moco
        if slicewise:
            # split data along Z
            sct.printv('Split data along Z...', verbose)
            fname_data_splitT_splitZ = file_data_splitT_num[it] + '_Z'
            cmd = fsloutput + 'fslsplit ' + file_data_splitT_num[it] + ' ' + fname_data_splitT_splitZ + ' -z'
            status, output = sct.run(cmd, verbose)
            file_data_ref_splitZ_num = []

            # loop across Z
            sct.printv('Loop across Z ('+todo+')...', verbose)
            for iz in range(nz):
                file_data_splitT_splitZ_num[it][iz] = fname_data_splitT_splitZ + str(iz).zfill(4)
                file_data_splitT_splitZ_moco_num[it][iz] = file_data_splitT_splitZ_num[it][iz] + suffix
                file_data_ref_splitZ_num.append(file_data_ref_splitZ + str(iz).zfill(4))
                file_mat[it][iz] = folder_mat + 'mat.T' + str(it) + '_Z' + str(iz) + '.txt'

                # run moco
                # TODO: simplifies this as in volume-based
                if todo == 'estimate':
                    cmd = fsloutput+'flirt -schedule '+schedule_file+' -in '+file_data_splitT_splitZ_num[it][iz]+' -ref '+file_data_ref_splitZ_num[iz]+' -omat '+file_mat[it][iz]+' -cost '+cost_function_flirt+fslmask[iz]
                if todo == 'apply':
                    cmd = fsloutput + 'flirt -in ' + file_data_splitT_splitZ_num[it][iz] + ' -ref ' + file_data_ref_splitZ_num[iz] + ' -applyxfm -init ' + file_mat[it][iz] + ' -out ' + file_data_splitT_splitZ_moco_num[it][iz] + ' -interp ' + interp
                if todo == 'estimate_and_apply':
                    cmd = fsloutput+'flirt -schedule '+schedule_file+ ' -in '+file_data_splitT_splitZ_num[it][iz]+' -ref '+ file_data_ref_splitZ_num[iz] +' -out '+file_data_splitT_splitZ_moco_num[it][iz]+' -omat '+file_mat[it][iz]+' -cost '+cost_function_flirt+fslmask[iz]+' -interp '+interp
                sct.run(cmd, verbose)

                #Check transformation absurdity
                fail_mat[it, iz] = check_transformation_absurdity(file_mat[it][iz], param)

            # Merge data along Z
            if todo != 'estimate':
                sct.printv('Concatenate along Z...', verbose)
                cmd = fsloutput + 'fslmerge -z ' + file_data_splitT_moco_num[it]
                for iz in range(nz):
                    cmd = cmd + ' ' + file_data_splitT_splitZ_moco_num[it][iz]
                sct.run(cmd, verbose)

        # volume-based moco
        else:
            # use FSL
            if program == 'flirt':
                file_mat[it] = folder_mat + 'mat.T' + str(it) + '.txt'
                cmd = fsloutput + 'flirt -schedule ' + schedule_file + ' -in ' + file_data_splitT_num[it] + ' -ref ' + fname_target
                if todo == 'estimate' or todo == 'estimate_and_apply':
                    cmd = cmd + ' -omat '+file_mat[it] + ' -cost ' + cost_function_flirt
                if todo == 'apply' or todo == 'estimate_and_apply':
                    cmd = cmd + ' -out ' + file_data_splitT_moco_num[it] + ' -interp ' + interp
                    if todo == 'apply':
                        cmd = cmd + ' -applyxfm -init ' + file_mat[it]
                sct.run(cmd, verbose)
                #Check transformation absurdity
                fail_mat[it] = check_transformation_absurdity(file_mat[it], param)
            # use ANTs
            elif program == 'ants':
                file_mat[it] = folder_mat + 'mat.T' + str(it)
                # TODO: figure out orientation
                if todo == 'estimate' or todo == 'estimate_and_apply':
                    cmd = 'sct_antsRegistration' \
                          ' --dimensionality 3' \
                          ' --transform BSplineSyN[1, 1x1x5, 0x0x0, 2]' \
                          ' --metric MI['+fname_target+'.nii, '+file_data_splitT_num[it]+'.nii, 1, 32]' \
                          ' --convergence 10x5' \
                          ' --shrink-factors 2x1' \
                          ' --smoothing-sigmas 1x1mm' \
                          ' --Restrict-Deformation '+restrict_deformation+'' \
                          ' --output ['+file_mat[it]+','+file_data_splitT_moco_num[it]+'.nii]' \
                          ' --interpolation '+interp
                if todo == 'apply':
                    cmd = 'sct_apply_transfo.py -i '+file_data_splitT_num[it]+'.nii -d '+fname_target+'.nii -w '+file_mat[it]+ext_mat+' -o '+file_data_splitT_moco_num[it]+'.nii'
                sct.run(cmd, verbose)
            elif program == 'ants_affine':
                file_mat[it] = folder_mat + 'mat.T' + str(it)
                # TODO: figure out orientation
                if todo == 'estimate' or todo == 'estimate_and_apply':
                    cmd = 'sct_antsRegistration' \
                          ' --dimensionality 3' \
                          ' --transform Affine[0.5]' \
                          ' --metric MI['+fname_target+'.nii, '+file_data_splitT_num[it]+'.nii, 1, 32]' \
                          ' --convergence 10x5' \
                          ' --shrink-factors 2x1' \
                          ' --smoothing-sigmas 2x1mm' \
                          ' --Restrict-Deformation '+restrict_deformation+'' \
                          ' --output ['+file_mat[it]+','+file_data_splitT_moco_num[it]+'.nii]' \
                          ' --interpolation '+interp
                if todo == 'apply':
                    # TODO: consider interpolation!
                    cmd = 'sct_apply_transfo.py -i '+file_data_splitT_num[it]+'.nii -d '+fname_target+'.nii -w '+file_mat[it]+ext_mat+' -o '+file_data_splitT_moco_num[it]+'.nii'
                sct.run(cmd, verbose)

    # Replace failed transformation matrix to the closest good one
    # TODO: do it also for the volume-based (not urgent...)
    if slicewise:
        fT, fZ = np.where(fail_mat == 1)
        gT, gZ = np.where(fail_mat == 0)
        for it in range(len(fT)):
            sct.printv(('\nReplace failed matrix T'+str(fT[it])+' Z'+str(fZ[it])+'...'),verbose)

            # rename failed matrix
            cmd = 'mv ' + file_mat[fT[it]][fZ[it]] + ' ' + file_mat[fT[it]][fZ[it]] + '_failed'
            status, output = sct.run(cmd, verbose)

            good_Zindex = np.where(gZ == fZ[it])
            good_index = gT[good_Zindex]

            I = np.amin(abs(good_index-fT[it]))
            cmd = 'cp ' + file_mat[good_index[I]][fZ[it]] + ' ' + file_mat[fT[it]][fZ[it]]
            status, output = sct.run(cmd, verbose)

    # Merge data along T
    fname_data_moco = fname_data+suffix
    if todo != 'estimate':
        sct.printv('\nMerge data back along T...', verbose)
        cmd = fsloutput + 'fslmerge -t ' + fname_data_moco
        for indice_index in range(len(index)):
            cmd = cmd + ' ' + file_data_splitT_moco_num[indice_index]
        sct.run(cmd, verbose)

    # if todo == 'estimate_and_apply':
    #     if param.mat_moco == '':
    #         sct.printv('\nDelete temporary files...', verbose)
    #         sct.run('rm -rf '+folder_mat, verbose)


#=======================================================================================================================
# check_transformation_absurdity:  find outliers
#=======================================================================================================================
def check_transformation_absurdity(file_mat, param):

    # init param
    failed_transfo = 0

    file = open(file_mat)
    M_transform = np.loadtxt(file)
    file.close()

    if abs(M_transform[0, 3]) > 10 or abs(M_transform[1, 3]) > 10 or abs(M_transform[2, 3]) > 10 or abs(M_transform[3, 3]) > 10:
        failed_transfo = 1
        sct.printv('  WARNING: This tranformation matrix is absurd, try others parameters (Gaussian mask, cost_function, group size, ...)', param.verbose, 'warning')

    return failed_transfo


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

    file_mat = [[[] for i in range(nz)] for i in range(nt)]
    for it in range(nt):
        for iz in range(nz):
            file_mat[it][iz] = folder_mat + 'mat.T' + str(it) + '_Z' + str(iz) + '.txt'

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
    for iz in range(nz):
        for it in range(nt):
            file =  open(file_mat[it][iz])
            Matrix = np.loadtxt(file)
            file.close()

            X[iz][it] = Matrix[0,3]
            Y[iz][it] = Matrix[1,3]

    # Generate motion splines
    sct.printv('\nGenerate motion splines...',verbose)
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
            pl.plot(T,X_smooth[iz][:],label='spline_smoothing')
            pl.plot(T,X[iz][:],marker='*',linestyle='None',label='original_val')
            if len(index_b0)!=0:
                T_b0 = [T[i_b0] for i_b0 in index_b0]
                X_b0 = [X[iz][i_b0] for i_b0 in index_b0]
                pl.plot(T_b0,X_b0,marker='D',linestyle='None',color='k',label='b=0')
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
            pl.plot(T,Y_smooth[iz][:],label='spline_smoothing')
            pl.plot(T,Y[iz][:],marker='*', linestyle='None',label='original_val')
            if len(index_b0)!=0:
                T_b0 = [T[i_b0] for i_b0 in index_b0]
                Y_b0 = [Y[iz][i_b0] for i_b0 in index_b0]
                pl.plot(T_b0,Y_b0,marker='D',linestyle='None',color='k',label='b=0')
            pl.title('Y')
            pl.grid()
            pl.legend()
            pl.show()

    #Storing the final Matrices
    sct.printv('\nStoring the final Matrices...',verbose)
    for iz in range(nz):
        for it in range(nt):
            file =  open(file_mat[it][iz])
            Matrix = np.loadtxt(file)
            file.close()

            Matrix[0,3] = X_smooth[iz][it]
            Matrix[1,3] = Y_smooth[iz][it]

            file =  open(file_mat[it][iz],'w')
            np.savetxt(file_mat[it][iz], Matrix, fmt="%s", delimiter='  ', newline='\n')
            file.close()

    sct.printv('\n...Done. Patient motion has been smoothed', verbose)
    sct.printv('------------------------------------------------------------------------------\n',verbose)


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
    m2c_fnames = [ fname for fname in os.listdir(param.mat_2_combine) if os.path.isfile(os.path.join(param.mat_2_combine, fname)) ]
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


#=======================================================================================================================
# get_interpolation: get correct interpolation field depending on program used
#=======================================================================================================================
def get_interpolation(program, interp):
    interp_program = ''
    if program == 'flirt':
        if interp == 'nn':
            interp_program = 'nearestneighbour'
        elif interp == 'trilinear':
            interp_program = 'trilinear'
        elif interp == 'spline':
            interp_program = 'spline'
    elif program == 'ants' or program == 'ants_affine':
        if interp == 'nn':
            interp_program = 'NearestNeighbor'
        elif interp == 'trilinear':
            interp_program = 'Linear'
        elif interp == 'spline':
            interp_program = 'BSpline[3]'
    # check if not assigned
    if interp_program == '':
        sct.printv('WARNING: interp_program not assigned. Using trilinear for ants_affine.', 1, 'warning')
        interp_program = 'Linear'
    return interp_program
