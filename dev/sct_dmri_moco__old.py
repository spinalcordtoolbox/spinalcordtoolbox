#!/usr/bin/env python
#########################################################################################
#
# Motion correction of dMRI data.
#
# Inspired by Xu et al. Neuroimage 2013.
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
# Modified: 2014-06-03
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: slice-wise
# TODO: fit spline
# TODO: only merge first X DWI data as target.
# TODO: moco b=0 on first b=0
# TODO: restrict deformation to TxTy -- copy the schedule file
# TODO: masking
# TODO: add flag for selecting output images

import sys
import os
import commands
import getopt
import nibabel
import time
import math
import sct_utils as sct


# DEFAULT PARAMETERS
class param:
    ## The constructor
    def __init__(self):
        self.debug              = 0
        self.interp = 'sinc' # final interpolation
        self.verbose            = 0 # verbose
        self.remove_temp_files  = 1 # remove temporary files



# MAIN
# ==========================================================================================
def main():

    # get path of the toolbox
    path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
    print path_sct

    # Initialization
    fsloutput = 'export FSLOUTPUTTYPE=NIFTI; ' # for faster processing, all outputs are in NIFTI
    fname_data = ''
    fname_bvecs = ''
    fname_schedule = path_sct+'/flirtsch/schedule_TxTy.sch'
    interp = param.interp
    remove_temp_files = param.remove_temp_files
    verbose = param.verbose
    start_time = time.time()

    # Parameters for debug mode
    if param.debug:
        fname_data = path_sct+'/testing/data/errsm_23/dmri/dmri.nii.gz'
        fname_bvecs = path_sct+'/testing/data/errsm_23/dmri/bvecs.txt'
        interp = 'trilinear'
        remove_temp_files = 0
        verbose = 1

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],'hb:i:v:s:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ("-b"):
            fname_bvecs = arg
        elif opt in ("-i"):
            fname_data = arg
        elif opt in ('-s'):
            interp = str(arg)
        elif opt in ('-v'):
            verbose = int(arg)

    # display usage if a mandatory argument is not provided
    if fname_data == '' or fname_bvecs == '':
        usage()

    # check existence of input files
    sct.check_file_exist(fname_data)
    sct.check_file_exist(fname_bvecs)

    # print arguments
    print '\nCheck parameters:'
    print '.. DWI data:             '+fname_data
    print '.. bvecs file:           '+fname_bvecs
    print ''

    # Get full path
    fname_data = os.path.abspath(fname_data)
    fname_bvecs = os.path.abspath(fname_bvecs)

    # Extract path, file and extension
    path_data, file_data, ext_data = sct.extract_fname(fname_data)

    path_tmp = sct.tmp_create(basename="dmri_moco__old", verbose=verbose)

    # go to tmp folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    # Get size of data
    print '\nGet dimensions data...'
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_data)
    print '.. '+str(nx)+' x '+str(ny)+' x '+str(nz)+' x '+str(nt)

    # Open bvecs file
    print '\nOpen bvecs file...'
    bvecs = []
    with open(fname_bvecs) as f:
        for line in f:
            bvecs_new = map(float, line.split())
            bvecs.append(bvecs_new)

    # Check if bvecs file is nx3
    if not len(bvecs[0][:]) == 3:
        print '.. WARNING: bvecs file is 3xn instead of nx3. Consider using sct_dmri_transpose_bvecs.'
        print 'Transpose bvecs...'
        # transpose bvecs
        bvecs = zip(*bvecs)

    # Identify b=0 and DW images
    print '\nIdentify b=0 and DW images...'
    index_b0 = []
    index_dwi = []
    for it in xrange(0,nt):
        if math.sqrt(math.fsum([i**2 for i in bvecs[it]])) < 0.01:
            index_b0.append(it)
        else:
            index_dwi.append(it)
    n_b0 = len(index_b0)
    n_dwi = len(index_dwi)
    print '.. Index of b=0:'+str(index_b0)
    print '.. Index of DWI:'+str(index_dwi)

    #TODO: check if number of bvecs and nt match

    # Split into T dimension
    print '\nSplit along T dimension...'
    #cmd = fsloutput+'fslsplit tmp.data tmp.data_splitT'
    status, output = sct.run(fsloutput+'fslsplit '+fname_data+' tmp.data_splitT')

    # retrieve output names
    status, output = sct.run('ls tmp.data_splitT*.*')
    file_data_split = output.split()
    # Remove .nii extension
    file_data_split = [file_data_split[i].replace('.nii','') for i in xrange (0,len(file_data_split))]

    # Merge b=0 images
    print '\nMerge b=0...'
    file_b0 = 'tmp.b0'
    cmd = fsloutput+'fslmerge -t '+file_b0
    for it in xrange(0,n_b0):
        cmd += ' '+file_data_split[index_b0[it]]
    #print('>> '+cmd)
    status, output = sct.run(cmd)

    # Merge DWI images
    print '\nMerge DWI...'
    file_dwi = 'tmp.dwi'
    cmd = fsloutput+'fslmerge -t '+file_dwi
    for it in xrange(0,n_dwi):
        cmd += ' '+file_data_split[index_dwi[it]]
    status, output = sct.run(cmd)

    # Average b=0 images
    print '\nAverage b=0...'
    file_b0_mean = 'tmp.b0_mean'
    cmd = fsloutput+'fslmaths '+file_b0+' -Tmean '+file_b0_mean
    status, output = sct.run(cmd)

    # Average DWI images
    print '\nAverage DWI...'
    file_dwi_mean = 'tmp.dwi_mean'
    cmd = fsloutput+'fslmaths '+file_dwi+' -Tmean '+file_dwi_mean
    status, output = sct.run(cmd)



    # REGISTER DWI TO THE MEAN DWI  -->  output transfo Tdwi
    # ---------------------------------------------------------------------------------------

    # loop across DWI data
    print '\nRegister DWI data to '+file_dwi_mean+'...'
    for it in xrange(0,n_dwi):
        # estimate transformation matrix
        file_target = file_dwi_mean
        file_mat = 'tmp.mat_'+str(index_dwi[it]).zfill(4)
        cmd = fsloutput+'flirt -in '+file_data_split[index_dwi[it]]+' -ref '+file_target+' -omat '+file_mat+' -cost normcorr -schedule '+fname_schedule+' -interp trilinear -out '+file_data_split[index_dwi[it]]+'_moco'
        status, output = sct.run(cmd)

    # Merge corrected DWI images
    print '\nMerge corrected DWI...'
    file_dwi = 'tmp.dwi_moco'
    cmd = fsloutput+'fslmerge -t '+file_dwi
    for it in xrange(0,n_dwi):
        cmd += ' '+file_data_split[index_dwi[it]]+'_moco'
    status, output = sct.run(cmd)

    # Average corrected DWI
    print '\nAverage corrected DWI...'
    file_dwi_mean = 'tmp.dwi_moco_mean'
    cmd = fsloutput+'fslmaths '+file_dwi+' -Tmean '+file_dwi_mean
    status, output = sct.run(cmd)


    # REGISTER B=0 DATA TO THE FIRST B=0  --> output transfo Tb0
    # ---------------------------------------------------------------------------------------
    print '\nRegister b=0 data to the first b=0...'
    for it in xrange(0,n_b0):
        # estimate transformation matrix
        file_target = file_data_split[int(index_b0[0])]
        file_mat = 'tmp.mat_'+str(index_b0[it]).zfill(4)
        cmd = fsloutput+'flirt -in '+file_data_split[index_b0[it]]+' -ref '+file_target+' -omat '+file_mat+' -cost normcorr -forcescaling -2D -out '+file_data_split[index_b0[it]]+'_moco'
        status, output = sct.run(cmd)

    # Merge corrected b=0 images
    print '\nMerge corrected b=0...'
    cmd = fsloutput+'fslmerge -t tmp.b0_moco'
    for it in xrange(0,n_b0):
        cmd += ' '+file_data_split[index_b0[it]]+'_moco'
    status, output = sct.run(cmd)

    # Average corrected b=0
    print '\nAverage corrected b=0...'
    cmd = fsloutput+'fslmaths tmp.b0_moco -Tmean tmp.b0_moco_mean'
    status, output = sct.run(cmd)


    # REGISTER MEAN DWI TO THE MEAN B=0  --> output transfo Tdwi2b0
    # ---------------------------------------------------------------------------------------
    print '\nRegister mean DWI to the mean b=0...'
    cmd = fsloutput+'flirt -in tmp.dwi_moco_mean -ref tmp.b0_moco_mean -omat tmp.mat_dwi2b0 -cost mutualinfo -forcescaling -dof 12 -2D -out tmp.dwi_mean_moco_reg2b0'
    status, output = sct.run(cmd)


    # COMBINE TRANSFORMATIONS
    # ---------------------------------------------------------------------------------------
    print '\nCombine all transformations...'
    # USE FSL convert_xfm: convert_xfm -omat AtoC.mat -concat BtoC.mat AtoB.mat
    # For DWI
    print '\n.. For DWI:'
    for it in xrange(0,n_dwi):
        cmd = 'convert_xfm -omat tmp.mat_final_'+str(index_dwi[it]).zfill(4)+' -concat tmp.mat_dwi2b0 tmp.mat_'+str(index_dwi[it]).zfill(4)
        status, output = sct.run(cmd)
    # For b=0 (don't concat because there is just one mat file -- just rename it)
    print '\n.. For b=0:'
    for it in xrange(0,n_b0):
        cmd = 'cp tmp.mat_'+str(index_b0[it]).zfill(4)+' tmp.mat_final_'+str(index_b0[it]).zfill(4)
        status, output = sct.run(cmd)


    # APPLY TRANSFORMATIONS
    # ---------------------------------------------------------------------------------------
    ## Split original data into T dimension
    #print '\nSplit original data along T dimension...'
    #cmd = fsloutput+'fslsplit '+fname_data+' tmp.data_raw_splitT'
    #print('>> '+cmd)
    #status, output = commands.getstatusoutput(cmd)

    #print '\nApply transformations to original data...'
    #for it in xrange(0,nt):
    #    cmd = fsloutput+'flirt -in tmp.data_raw_splitT'+str(it).zfill(4)+' -ref tmp.data_raw_splitT'+index_b0[0].zfill(4)+' -applyxfm -init tmp.mat_final_'+str(it).zfill(4)+' -out tmp.data_raw_splitT'+str(it).zfill(4)+'_moco'
    #    print('>> '+cmd)
    #    status, output = commands.getstatusoutput(cmd)
    #
    ## Merge corrected data
    #print '\nMerge corrected data...'
    #cmd = fsloutput+'fslmerge -t tmp.data_raw_moco'
    #for it in xrange(0,it):
    #    cmd += ' tmp.data_raw_splitT'+str(it).zfill(4)+'_moco'
    #print('>> '+cmd)
    #status, output = commands.getstatusoutput(cmd)

    print '\nApply transformations...'
    for it in xrange(0,nt):
        # -paddingsize 3 prevents from having missing slices at the edge
        cmd = fsloutput+'flirt -in tmp.data_splitT'+str(it).zfill(4)+' -ref tmp.data_splitT'+str(index_b0[0]).zfill(4)+' -applyxfm -init tmp.mat_final_'+str(it).zfill(4)+' -out tmp.data_splitT'+str(it).zfill(4)+'_moco -paddingsize 3'+' -interp '+interp
        status, output = sct.run(cmd)

    # Merge corrected data
    print '\nMerge all corrected data...'
    cmd = fsloutput+'fslmerge -t tmp.data_moco'
    for it in xrange(0,nt):
        cmd += ' tmp.data_splitT'+str(it).zfill(4)+'_moco'
    status, output = sct.run(cmd)

    # Merge corrected DWI images
    print '\nMerge corrected DWI...'
    cmd = fsloutput+'fslmerge -t tmp.dwi_moco'
    for it in xrange(0,n_dwi):
        cmd += ' tmp.data_splitT'+str(index_dwi[it]).zfill(4)+'_moco'
    status, output = sct.run(cmd)

    # Average corrected DWI
    print '\nAverage corrected DWI...'
    cmd = fsloutput+'fslmaths tmp.dwi_moco -Tmean tmp.dwi_moco_mean'
    status, output = sct.run(cmd)

    # Merge corrected b=0 images
    print '\nMerge corrected b=0...'
    cmd = fsloutput+'fslmerge -t tmp.b0_moco'
    for it in xrange(0,n_b0):
        cmd += ' tmp.data_splitT'+str(index_b0[it]).zfill(4)+'_moco'
    status, output = sct.run(cmd)

    # Average corrected b=0
    print '\nAverage corrected b=0...'
    cmd = fsloutput+'fslmaths tmp.b0_moco -Tmean tmp.b0_moco_mean'
    status, output = sct.run(cmd)

    # Generate output files
    print('\nGenerate output files...')
    sct.generate_output_file('tmp.data_moco.nii',path_data,file_data+'_moco',ext_data)
    sct.generate_output_file('tmp.dwi_moco_mean.nii',path_data,'dwi_moco_mean',ext_data)
    sct.generate_output_file('tmp.b0_moco_mean.nii',path_data,'b0_moco_mean',ext_data)

    # come back
    os.chdir(curdir)

    # Delete temporary files
    if remove_temp_files == 1:
        print '\nDelete temporary files...'
        sct.run('rm -rf '+path_tmp)

    # display elapsed time
    elapsed_time = time.time() - start_time
    print '\nFinished! Elapsed time: '+str(int(round(elapsed_time)))+'s'

    # to view results
    print '\nTo view results, type:'
    print 'fslview '+file_data+' '+file_data+'_moco &\n'



# Print usage
# ==========================================================================================
def usage():
    print '\n' \
        ''+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        '  Slice-by-slice motion correction of DWI data.\n' \
        '\n' \
        'USAGE\n' \
        '  '+os.path.basename(__file__)+' -i <dmri> -b <bvecs>\n' \
        '\n' \
        'MANDATORY ARGUMENTS\n' \
        '  -i <dmri>                  diffusion data\n' \
        '  -b <bvecs>                 bvecs file\n' \
        '\n' \
        'OPTIONAL ARGUMENTS\n' \
        '  -s {nearestneighbour, trilinear, sinc}       final interpolation. Default='+str(param.interp)+'\n' \
        '  -v {0, 1}                   verbose. Default='+str(param.verbose)+'.\n'

    # exit program
    sys.exit(2)



# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = param()
    # call main function
    main()
