#!/usr/bin/env python
#########################################################################################
#
# Motion correction of MRI data. Module called by sct_dmri_moco()
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Karun Raju, Tanguy Duval, Julien Cohen-Adad
# Modified: 2014-06-14
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# check if needed Python libraries are already installed or not
import sys
import os
import commands
import getopt
import time
import math
try:
    import nibabel
except ImportError:
    print '--- nibabel not installed! Exit program. ---'
    sys.exit(2)
try:
    import numpy as np
except ImportError:
    print '--- numpy not installed! Exit program. ---'
    sys.exit(2)

class param_class:
    def __init__(self):
        
        #============================================
        #Different Parameters
        #============================================
        self.fname_data                = ''
        self.fname_target              = ''
        self.fname_centerline          = ''
        self.output_path               = ''
        self.mat_final                 = ''
        self.mat_moco                  = ''
        self.todo                      = ''              # 'estimate' || 'apply' || 'estimate_and_apply'. NB: 'apply' requires input matrix. Default is 'estimate_and_apply'.
        self.suffix                    = '_moco'
        self.mask_size                 = 0               # sigma of gaussian mask in mm --> std of the kernel. Default is 0
        self.program                   = 'FLIRT'
        self.cost_function_flirt       = 'normcorr'      # 'mutualinfo' | 'woods' | 'corratio' | 'normcorr' | 'normmi' | 'leastsquares'. Default is 'normcorr'.
        self.interp                    = 'trilinear'              #  Default is 'trilinear'. Additional options: trilinear,nearestneighbour,sinc,spline
        self.delete_tmp_files          = 1
        self.merge_back                = 1

#=======================================================================================================================
# main
#=======================================================================================================================

def main():
    
    start_time = time.time()
    param = param_class()
    
    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],'hi:t:c:f:g:l:m:o:p:r:s:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ('-i'):
            param.fname_data = arg
        elif opt in ('-t'):
            param.fname_target = arg
        elif opt in ('-c'):
            param.cost_function_flirt = arg
        elif opt in ('-f'):
            param.mat_final = arg
        elif opt in ('-g'):
            param.mat_moco = arg
        elif opt in ('-l'):
            param.fname_centerline = arg
        elif opt in ('-m'):
            param.todo = arg
        elif opt in ('-o'):
            param.output_path = arg
        elif opt in ('-p'):
            param.interp = arg
        elif opt in ('-r'):
            param.delete_tmp_files = int(arg)
        elif opt in ('-s'):
            param.mask_size = float(arg)

    # display usage if a mandatory argument is not provided
    if param.fname_data == '':
        print '\n\nAll mandatory arguments are not provided \n'
        usage()
    else:
        if param.todo != 'apply':
            if param.fname_target=='':
                print '\n\nAll mandatory arguments are not provided \n'
                usage()
        else:
            if param.mat_final=='':
                print '\n\nFinal matrix folder is not provided \n'
                usage()
            param.fname_target = param.fname_data

    if param.todo=='':  param.todo = 'estimate_and_apply'                         #Default Value
    if param.cost_function_flirt=='':  param.cost_function_flirt = 'normcorr'     #Default Value
    if param.output_path=='': param.output_path = os.getcwd() + '/'

    # get path of the toolbox
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
    # append path that contains scripts, to be able to load modules
    sys.path.append(path_sct + '/scripts')
    import sct_utils as sct
    
    # create temporary folder
    path_tmp = 'tmp.'+time.strftime("%y%m%d%H%M%S")
    sct.run('mkdir '+ path_tmp)
    
    # go to tmp folder
    os.chdir(path_tmp)

    sct_moco(param)

    # come back to parent folder
    os.chdir('..')

    # Delete temporary files
    if param.delete_tmp_files == 1:
        print '\nDelete temporary files...'
        sct.run('rm -rf '+ path_tmp)

    # display elapsed time
    elapsed_time = time.time() - start_time
    print '\nFinished! Elapsed time: '+str(int(round(elapsed_time)))+'s'

#=======================================================================================================================
#sct_moco Function
#=======================================================================================================================
def sct_moco(param):
    print '\n\n\n\n==================================================='
    print '                Running: sct_moco'
    print '===================================================\n'
    
    print 'Input File:',param.fname_data
    print 'Reference File:',param.fname_target
    if param.fname_centerline!='': print 'Centreline File:',param.fname_centerline
    print 'Method:',param.todo
    
    # Initialization
    fsloutput = 'export FSLOUTPUTTYPE=NIFTI; ' # for faster processing, all outputs are in NIFTI
    
    #Different parameters
    fname_data          = param.fname_data
    fname_target        = param.fname_target
    mat_final           = param.mat_final
    todo                = param.todo
    suffix              = param.suffix
    mask_size           = param.mask_size
    program             = param.program
    cost_function_flirt = param.cost_function_flirt
    interp              = param.interp
    merge_back          = param.merge_back
    
    # get path of the toolbox
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
    # append path that contains scripts, to be able to load modules
    sys.path.append(path_sct + '/scripts')
    import sct_utils as sct
    
    # check existence of input files
    sct.check_file_exist(fname_data)
    if todo!='apply':
        sct.check_file_exist(fname_target)
    
    # Extract path, file and extension
    path_data, file_data, ext_data = sct.extract_fname(fname_data)
    
    #Schedule file for FLIRT
    schedule_file = path_sct + '/flirtsch/schedule_TxTy_2mmScale.sch'
    print '\n.. Schedule file: ',schedule_file
    
    if todo=='estimate':
        if param.mat_moco=='':
            folder_mat = 'mat_moco/'
        else:
            folder_mat = param.mat_moco + '/'
    elif todo=='estimate_and_apply':
        if param.mat_moco=='':
            folder_mat = 'mat_tmp/'
        else:
            folder_mat = param.mat_moco + '/'
    else:
        folder_mat = mat_final
    if not os.path.exists(folder_mat): os.makedirs(folder_mat)
    
    # Get size of data
    print '\nGet dimensions data...'
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_data)
    print '.. '+str(nx)+' x '+str(ny)+' x '+str(nz)+' x '+str(nt)
    
    # split along T dimension
    fname_data_splitT = 'data_splitT'
    cmd = fsloutput + 'fslsplit ' + fname_data + ' ' + fname_data_splitT
    status, output = sct.run(cmd)
    
    #SLICE-by-SLICE MOTION CORRECTION
    print '\n   Motion correction...'
    #split target data along Z
    fname_data_ref_splitZ = 'target_splitZ'
    cmd = fsloutput + 'fslsplit ' + fname_target + ' ' + fname_data_ref_splitZ + ' -z'
    status, output = sct.run(cmd)
    
    #Generate Gaussian Mask
    fslmask = []
    if mask_size>0:
        sigma = np.array([mask_size/px, mask_size/py])
        dims = np.array([nx, ny, nz, nt])
        data = nibabel.load((fname_data_ref_splitZ + '0000.nii'))
        hdr = data.get_header()
        hdr.set_data_dtype('uint8') # set imagetype to uint8
        
        if param.fname_centerline=='':
            center = np.array([math.ceil(nx/2), math.ceil(ny/2), math.ceil(nz/2), math.ceil(nt/2)])
            fname_mask = 'gaussian_mask_in'
            M_mask = gauss2d(dims, sigma, center)
            # Write NIFTI volumes
            img = nibabel.Nifti1Image(M_mask, None, hdr)
            nibabel.save(img,(fname_mask+'.nii'))
            for iZ in range(nz):
                fslmask.append(' -inweight ' + fname_mask + ' -refweight ' + fname_mask)
            print '\n.. File created: ',fname_mask
        else:
            centerline = nibabel.load(param.fname_centerline)
            data_centerline = centerline.get_data()
            cx,cy,cz = np.where(data_centerline>0)
            arg = np.argsort(cz)
            cz = cz[arg]
            cx = cx[arg]
            cy = cy[arg]
            fname_mask = 'gaussian_mask_in'
            for iZ in range(nz):
                center = np.array([cx[iZ],cy[iZ]])
                M_mask = gauss2d(dims, sigma, center)
                # Write NIFTI volumes
                img = nibabel.Nifti1Image(M_mask, None, hdr)
                nibabel.save(img,(fname_mask+str(iZ)+'.nii'))
                fslmask.append(' -inweight ' + fname_mask+str(iZ) + ' -refweight ' + fname_mask+str(iZ))
                print '\n.. File created: ',(fname_mask+str(iZ))

            #Merging all masks
            cmd = 'fslmerge -z ' + path_data + 'mask '
            for iZ in range(nz):
                cmd = cmd + fname_mask+str(iZ)+' '
            status, output = sct.run(cmd)
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

    print 'Loop on iT...'
    for indice_index in range(nt):
        
        iT = index[indice_index]
        print '\nVolume ',str((iT+1)),'/',str(nt),':'
        print '--------------------'
        
        fname_data_splitT_num.append(fname_data_splitT + str(iT).zfill(4)) 
        fname_data_splitT_moco_num.append(fname_data_splitT + suffix + str(iT).zfill(4)) 
        
        # SLICE-WISE MOTION CORRECTION
        print 'Slicewise motion correction...'
        
        # split data along Z
        print 'Split data along Z...\n'
        
        fname_data_splitT_splitZ = fname_data_splitT_num[iT] + '_splitZ'
        cmd = fsloutput + 'fslsplit ' + fname_data_splitT_num[iT] + ' ' + fname_data_splitT_splitZ + ' -z'
        status, output = sct.run(cmd)
        
        # Loop On Z
        print 'Loop on Z...'
        
        fname_data_ref_splitZ_num = []
        for iZ in range(nz):
            fname_data_splitT_splitZ_num[iT][iZ] = fname_data_splitT_splitZ + str(iZ).zfill(4) 
            fname_data_splitT_splitZ_moco_num[iT][iZ] = fname_data_splitT_splitZ_num[iT][iZ] + suffix
            fname_data_ref_splitZ_num.append(fname_data_ref_splitZ + str(iZ).zfill(4)) 
            fname_mat[iT][iZ] = folder_mat + 'mat.T' + str(iT) + '_Z' + str(iZ) + '.txt'
            
            if todo == 'estimate':
                if program == 'FLIRT':
                    print '\nProcess with FLIRT...\n'
                    cmd = fsloutput+'flirt -schedule '+schedule_file+' -in '+fname_data_splitT_splitZ_num[iT][iZ]+' -ref '+fname_data_ref_splitZ_num[iZ]+' -omat '+fname_mat[iT][iZ]+' -cost '+cost_function_flirt+fslmask[iZ]+' -interp '+interp
        
            if todo == 'apply':
                if program == 'FLIRT':
                    print '\nProcess with FLIRT...\n'
                    cmd = fsloutput + 'flirt -in ' + fname_data_splitT_splitZ_num[iT][iZ] + ' -ref ' + fname_data_ref_splitZ_num[iZ] + ' -applyxfm -init ' + fname_mat[iT][iZ] + ' -out ' + fname_data_splitT_splitZ_moco_num[iT][iZ] + ' -interp ' + interp

            if todo == 'estimate_and_apply':
                if program == 'FLIRT':
                    print '\nProcess with FLIRT...\n'
                    cmd = fsloutput+'flirt -schedule '+schedule_file+ ' -in '+fname_data_splitT_splitZ_num[iT][iZ]+' -ref '+ fname_data_ref_splitZ_num[iZ] +' -out '+fname_data_splitT_splitZ_moco_num[iT][iZ]+' -omat '+fname_mat[iT][iZ]+' -cost '+cost_function_flirt+fslmask[iZ]+' -interp '+interp

            if program =='FLIRT':
                status, output = sct.run(cmd)

            #Check transformation absurdity

            file =  open(fname_mat[iT][iZ])
            M_transform = np.loadtxt(file)
            file.close()

            if abs(M_transform[0,3]) > 10 or abs(M_transform[1,3]) > 10 or abs(M_transform[2,3]) > 10 or abs(M_transform[3,3]) > 10 :
                nb_fails = nb_fails + 1
                fail_mat[iT,iZ] = 1
                print 'failure... this tranformation matrix is absurd, try others parameters (SPM, cost_function...) '

        # Merge data along Z
        if todo!='estimate':
            if merge_back==1:
                print '\n\nConcatenate along Z...\n'
                
                cmd = fsloutput + 'fslmerge -z ' + fname_data_splitT_moco_num[iT]
                for iZ in range(nz):
                    cmd = cmd + ' ' + fname_data_splitT_splitZ_moco_num[iT][iZ]
                status, output = sct.run(cmd)
                print '.. File created: ',fname_data_splitT_moco_num[iT]
    
    
    #Replace failed transformation matrix to the closest good one
    
    fT, fZ = np.where(fail_mat==1)
    gT, gZ = np.where(fail_mat==0)
    
    for iT in range(len(fT)):
        print '\nReplace failed matrix T',str(fT[iT]), ' Z', str(fZ[iT]),'...'
        
        # rename failed matrix
        cmd = 'mv ' + fname_mat[fT[iT]][fZ[iT]] + ' ' + fname_mat[fT[iT]][fZ[iT]] + '_failed'
        status, output = sct.run(cmd)
        
        good_Zindex = np.where(gZ == fZ[iT])
        good_index = gT[good_Zindex]
        
        I = np.amin(abs(good_index-fT[iT]))
        cmd = 'cp ' + fname_mat[good_index[I]][fZ[iT]] + ' ' + fname_mat[fT[iT]][fZ[iT]]
        status, output = sct.run(cmd)

    fname_data_moco = param.output_path + file_data + suffix + '.nii'
    # Merge data along T
    if todo!='estimate':
        if merge_back==1:
            print '\n\nMerge data back along T...\n'
            
            cmd = fsloutput + 'fslmerge -t ' + fname_data_moco
            
            for indice_index in range(len(index)):
                cmd = cmd + ' ' + fname_data_splitT_moco_num[indice_index]
            status, output = sct.run(cmd)
            print '.. File created: ',fname_data_moco

    if todo=='estimate_and_apply':
        if param.mat_moco=='':
            print '\nDelete temporary files...'
            sct.run('rm -rf '+ folder_mat)
    print '\n\n==================================================='
    print '                Completed: sct_moco'
    print '===================================================\n\n\n'

#=======================================================================================================================
# usage
#=======================================================================================================================
def usage():
    print '\n' \
        ''+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        'Register a 4D volume (data) on a 3D volume (ref) slice by slice. \n' \
        '\nUSAGE: \n' \
        '  '+os.path.basename(__file__)+' -i <filename> -r <reference_file>\n' \
        '\n'\
        'MANDATORY ARGUMENTS\n' \
        '  -i           input_file \n' \
        '  -t           reference file - if -m !=apply \n' \
        '\n'\
        'OPTIONAL ARGUMENTS\n' \
        '  -o           Specify Output path.\n' \
        '  -l           Centerline file can be given to specify the centre of Gaussian Mask. \n' \
        '  -m           method - estimate | apply | estimate_and_apply. NB: <apply> requires -f. Default is estimate_and_apply \n' \
        '  -s           Gaussian Mask_size - Specify mask_size in millimeters. Default value of mask_size is 0.\n' \
        '  -f           Input Matrix folder. Used if -m apply.\n' \
        '  -g           Output Matrix Folder name. (Used if -m is not <apply>)\n' \
        '  -c           Cost function FLIRT - mutualinfo | woods | corratio | normcorr | normmi | leastsquares. Default is <normcorr>..\n' \
        '  -p           Interpolation - Default is trilinear. Additional options: nearestneighbour,sinc,spline.\n' \
        '  -r           Set value to 0 for not deleting temp files. Default value is 1 \n' \
        '  -h           help. Show this message.\n' \
        '\n'\
        'EXAMPLE:\n' \
        '  '+os.path.basename(__file__)+' -i dwi_averaged_groups.nii -r dwi_mean.nii \n'
    
    #Exit Program
    sys.exit(2)

#=======================================================================================================================
# 2D Gaussian Function
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
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # call main function
    main()