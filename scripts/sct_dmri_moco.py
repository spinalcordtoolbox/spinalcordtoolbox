#!/usr/bin/env python
#########################################################################################
# Motion correction of dMRI data.
#
# Uses the method of Xu et al. Neuroimage 2013.
#
#
# USAGE
# ---------------------------------------------------------------------------------------
#   sct_dmri_moco.py <dmri> <bvecs>
#
#
# INPUT
# ---------------------------------------------------------------------------------------
# dmri              dmri data to correct. Can be nii or nii.gz
# bvecs             bvecs ASCII file (FSL format).
#
#
# OUTPUT
# ---------------------------------------------------------------------------------------
# dmri_reg          motion corrected dmri data
#
#
# DEPENDENCIES
# ---------------------------------------------------------------------------------------
# EXTERNAL PYTHON PACKAGES
# - nibabel <http://nipy.sourceforge.net/nibabel/>
#
# EXTERNAL SOFTWARE
# - FSL <http://fsl.fmrib.ox.ac.uk/fsl/> 
#
#
# ---------------------------------------------------------------------------------------
# TODO: restrict deformation to TxTy -- copy the schedule file
# TODO: masking
# TODO: add flag for selecting output images
#
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuropoly.info>
# Author: Julien Cohen-Adad
# Modified: 2013-10-20
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#########################################################################################

import sys
import os
import commands
import nibabel
import math

# PARAMETERS
#crop_data           = 1 # Crop data for faster processing
debugging           = 0  # automatic file names for debugging


# INITIALIZATIONS
fsloutput = 'export FSLOUTPUTTYPE=NIFTI; ' # for faster processing, all outputs are in NIFTI




# MAIN
#########################################################################################
def main():

    # Check inputs
    if debugging:
        #fname_data = '/Users/julien/MRI/david_cadotte/2013-10-19_multiparametric/dmri/dmri_crop.nii.gz'
        #fname_bvecs = '/Users/julien/MRI/david_cadotte/2013-10-19_multiparametric/dmri/bvecs_t_concat.txt'
        fname_data = '/Users/julien/MRI/multisite_DTI/data_montreal/subject1/nonzoom1/dmri.nii.gz'
        fname_bvecs = '/Users/julien/MRI/multisite_DTI/data_montreal/subject1/nonzoom1/bvecs_t.txt'
    else:
        path_func, file_func, ext_func = extract_fname(sys.argv[0])
        if len(sys.argv) < 3:
            usage()
        fname_data = sys.argv[1]
        fname_bvecs = sys.argv[2]

        #TODO: check existence of input files

    # Extract path, file and extension
    path_data, file_data, ext_data = extract_fname(fname_data)


    # PREPARE DATA
    # ---------------------------------------------------------------------------------------


    # Crop data
    #if crop_data:
    #    print '\nCrop data...'
    #    cmd = fsloutput+'fslroi '+fname_data+' tmp.data 90 70 100 60 0 -1'
    #else:
    #    print '\nDo not crop data (just rename them and convert to .nii)...'
    #    cmd = 'fslchfiletype '+fname_data+' tmp.data'
    #print('>> '+cmd)
    #status, output = commands.getstatusoutput(cmd)

    print '\nCopy data locally...'
    cmd = 'fslchfiletype NIFTI '+fname_data+' tmp.data'
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)

    # Get size of data
    print '\nGet size data...'
    nx, ny, nz, nt = get_size_data('tmp.data.nii')
    print '.. '+str(nx)+' x '+str(ny)+' x '+str(nz)+' x '+str(nt)

    # Open bvecs file
    bvecs = []
    with open(fname_bvecs) as f:
        for line in f:
            bvecs_new = map(float, line.split())
            bvecs.append(bvecs_new)

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
    cmd = fsloutput+'fslsplit tmp.data tmp.data_splitT'
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)

    # retrieve output names
    status, output = commands.getstatusoutput('ls tmp.data_splitT*.*')
    file_data_split = output.split()
    # Remove .nii extension
    file_data_split = [file_data_split[i].replace('.nii','') for i in xrange (0,len(file_data_split))]

    # Merge b=0 images
    print '\nMerge b=0...'
    file_b0 = 'tmp.b0'
    cmd = fsloutput+'fslmerge -t '+file_b0
    for it in xrange(0,n_b0):
        cmd += ' '+file_data_split[index_b0[it]]
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)

    # Merge DWI images
    print '\nMerge DWI...'
    file_dwi = 'tmp.dwi'
    cmd = fsloutput+'fslmerge -t '+file_dwi
    for it in xrange(0,n_dwi):
        cmd += ' '+file_data_split[index_dwi[it]]
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)

    # Average b=0 images
    print '\nAverage b=0...'
    file_b0_mean = 'tmp.b0_mean'
    cmd = fsloutput+'fslmaths '+file_b0+' -Tmean '+file_b0_mean
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)

    # Average DWI images
    print '\nAverage DWI...'
    file_dwi_mean = 'tmp.dwi_mean'
    cmd = fsloutput+'fslmaths '+file_dwi+' -Tmean '+file_dwi_mean
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)



    # REGISTER DWI TO THE MEAN DWI  -->  output transfo Tdwi
    # ---------------------------------------------------------------------------------------

    # loop across DWI data
    print '\nRegister DWI data to '+file_dwi_mean+'...'
    for it in xrange(0,n_dwi):
        # estimate transformation matrix
        file_target = file_dwi_mean
        file_mat = 'tmp.mat_'+str(index_dwi[it]).zfill(4)
        cmd = fsloutput+'flirt -in '+file_data_split[index_dwi[it]]+' -ref '+file_target+' -omat '+file_mat+' -cost normcorr -forcescaling -2D -nosearch -interp trilinear -out '+file_data_split[index_dwi[it]]+'_moco'
        #cmd = fsloutput+'flirt -in '+file_data_split[index_dwi[it]]+' -ref '+file_target+' -omat '+file_mat+' -cost normcorr -forcescaling -dof 12 -2D -interp trilinear -out '+file_data_split[index_dwi[it]]+'_moco'
        #cmd = fsloutput+'flirt -in '+file_data_split[index_dwi[it]]+' -ref '+file_target+' -omat '+file_mat+' -cost normcorr -forcescaling -schedule sct_schedule_TxTy.txt -interp trilinear -out '+file_data_split[index_dwi[it]]+'_moco'
        #cmd = fsloutput+'flirt -in '+file_data_split[index_dwi[it]]+' -ref '+file_target+' -omat '+file_mat+' -cost normcorr -forcescaling -schedule sct_schedule_TxTy.txt -nosearch -interp trilinear -out '+file_data_split[index_dwi[it]]+'_moco'+' -inweight mask_dmri -refweight mask_dmri'
        print('>> '+cmd)
        status, output = commands.getstatusoutput(cmd)

    # Merge corrected DWI images
    print '\nMerge corrected DWI...'
    file_dwi = 'tmp.dwi_moco'
    cmd = fsloutput+'fslmerge -t '+file_dwi
    for it in xrange(0,n_dwi):
        cmd += ' '+file_data_split[index_dwi[it]]+'_moco'
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)

    # Average corrected DWI
    print '\nAverage corrected DWI...'
    file_dwi_mean = 'tmp.dwi_moco_mean'
    cmd = fsloutput+'fslmaths '+file_dwi+' -Tmean '+file_dwi_mean
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)


    # REGISTER B=0 DATA TO THE FIRST B=0  --> output transfo Tb0
    # ---------------------------------------------------------------------------------------
    print '\nRegister b=0 data to the first b=0...'
    for it in xrange(0,n_b0):
        # estimate transformation matrix
        file_target = file_data_split[int(index_b0[0])]
        file_mat = 'tmp.mat_'+str(index_b0[it]).zfill(4)
        cmd = fsloutput+'flirt -in '+file_data_split[index_b0[it]]+' -ref '+file_target+' -omat '+file_mat+' -cost normcorr -forcescaling -2D -out '+file_data_split[index_b0[it]]+'_moco'
        print('>> '+cmd)
        status, output = commands.getstatusoutput(cmd)

    # Merge corrected b=0 images
    print '\nMerge corrected b=0...'
    cmd = fsloutput+'fslmerge -t tmp.b0_moco'
    for it in xrange(0,n_b0):
        cmd += ' '+file_data_split[index_b0[it]]+'_moco'
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)

    # Average corrected b=0
    print '\nAverage corrected b=0...'
    cmd = fsloutput+'fslmaths tmp.b0_moco -Tmean tmp.b0_moco_mean'
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)


    # REGISTER MEAN DWI TO THE MEAN B=0  --> output transfo Tdwi2b0
    # ---------------------------------------------------------------------------------------
    print '\nRegister mean DWI to the mean b=0...'
    cmd = fsloutput+'flirt -in tmp.dwi_moco_mean -ref tmp.b0_moco_mean -omat tmp.mat_dwi2b0 -cost mutualinfo -forcescaling -dof 12 -2D -out tmp.dwi_mean_moco_reg2b0'
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)


    # COMBINE TRANSFORMATIONS
    # ---------------------------------------------------------------------------------------
    print '\nCombine all transformations...'
    # USE FSL convert_xfm: convert_xfm -omat AtoC.mat -concat BtoC.mat AtoB.mat
    # For DWI
    print '\n.. For DWI:'
    for it in xrange(0,n_dwi):
        cmd = 'convert_xfm -omat tmp.mat_final_'+str(index_dwi[it]).zfill(4)+' -concat tmp.mat_dwi2b0 tmp.mat_'+str(index_dwi[it]).zfill(4)
        print('>> '+cmd)
        status, output = commands.getstatusoutput(cmd)
    # For b=0 (don't concat because there is just one mat file -- just rename it)
    print '\n.. For b=0:'
    for it in xrange(0,n_b0):
        cmd = 'cp tmp.mat_'+str(index_b0[it]).zfill(4)+' tmp.mat_final_'+str(index_b0[it]).zfill(4)
        print('>> '+cmd)
        status, output = commands.getstatusoutput(cmd)


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
        cmd = fsloutput+'flirt -in tmp.data_splitT'+str(it).zfill(4)+' -ref tmp.data_splitT'+str(index_b0[0]).zfill(4)+' -applyxfm -init tmp.mat_final_'+str(it).zfill(4)+' -out tmp.data_splitT'+str(it).zfill(4)+'_moco -paddingsize 3'
        print('>> '+cmd)
        status, output = commands.getstatusoutput(cmd)

    # Merge corrected data
    print '\nMerge all corrected data...'
    cmd = fsloutput+'fslmerge -t tmp.data_moco'
    for it in xrange(0,nt):
        cmd += ' tmp.data_splitT'+str(it).zfill(4)+'_moco'
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)

    # Merge corrected DWI images
    print '\nMerge corrected DWI...'
    cmd = fsloutput+'fslmerge -t tmp.dwi_moco'
    for it in xrange(0,n_dwi):
        cmd += ' tmp.data_splitT'+str(index_dwi[it]).zfill(4)+'_moco'
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)

    # Average corrected DWI
    print '\nAverage corrected DWI...'
    cmd = fsloutput+'fslmaths tmp.dwi_moco -Tmean tmp.dwi_moco_mean'
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)

    # Merge corrected b=0 images
    print '\nMerge corrected b=0...'
    cmd = fsloutput+'fslmerge -t tmp.b0_moco'
    for it in xrange(0,n_b0):
        cmd += ' tmp.data_splitT'+str(index_b0[it]).zfill(4)+'_moco'
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)

    # Average corrected b=0
    print '\nAverage corrected b=0...'
    cmd = fsloutput+'fslmaths tmp.b0_moco -Tmean tmp.b0_moco_mean'
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)

    # Generate output files
    print('\nGenerate output files...')
    generate_output_file('tmp.data_moco',path_data,file_data+'_moco',ext_data)
    #generate_output_file('tmp.dwi_mean',path_data,'dwi_mean',ext_data)
    #generate_output_file('tmp.dwi_moco',path_data,'dwi_moco',ext_data)
    generate_output_file('tmp.dwi_moco_mean',path_data,'dwi_moco_mean',ext_data)
    #generate_output_file('tmp.b0',path_data,'b0',ext_data)
    #generate_output_file('tmp.b0_mean',path_data,'b0_mean',ext_data)
    #generate_output_file('tmp.b0_moco',path_data,'b0_moco',ext_data)
    generate_output_file('tmp.b0_moco_mean',path_data,'b0_moco_mean',ext_data)

    # Remove temporary files
    print('\nRemove temporary files...')
    cmd = 'rm tmp.*'
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)

    # End of script
    print('\nCreated files:')
    print '.. '+file_data+'_moco'+ext_data
    print '.. '+'b0_moco_mean'+ext_data
    print '.. '+'dwi_moco_mean'+ext_data+'\n'


# Extracts path, file and extension
#########################################################################################
def extract_fname(fname):
    # extract path
    path_fname = os.path.dirname(fname)+'/'
    # check if only single file was entered (without path)
    if path_fname == '/':
        path_fname = ''
    # extract file and extension
    file_fname = fname
    file_fname = file_fname.replace(path_fname,'')
    file_fname, ext_fname = os.path.splitext(file_fname)
    # check if .nii.gz file
    if ext_fname == '.gz':
        file_fname = file_fname[0:len(file_fname)-4]
        ext_fname = ".nii.gz"
    return path_fname, file_fname, ext_fname
#########################################################################################


# get size data
#########################################################################################
def get_size_data(fname_in):
    # get file size with nibabel
    hdr = nibabel.load(fname_in)
    nx, ny, nz, nt = hdr.get_header().get_data_shape()
    return nx, ny, nz, nt
#########################################################################################


# generate output file
#########################################################################################
def generate_output_file(file_in,path_out,file_out,ext_out):
    # Change output extension to be the same as input extension
    print('\nChange output extension to be the same as input extension...')
    if ext_out == '.nii':
        cmd = 'fslchfiletype NIFTI '+file_in
    elif ext_out == '.nii.gz':
        cmd = 'fslchfiletype NIFTI_GZ '+file_in
    print(">> "+cmd)
    os.system(cmd)
    # Move to output folder
    cmd = 'mv '+file_in+ext_out+' '+path_out+file_out+ext_out
    print(">> "+cmd)
    os.system(cmd)
    return
#########################################################################################



#=======================================================================================================================
# usage
#=======================================================================================================================
def usage():
    print '\n' \
        'sct_dmri_moco\n' \
        '--------------------------------------------------------------------------------------------------------------\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        '  This program takes as inputs diffusion MRI data and its bvecs and returns the dMRI data where the motion of ' \
        'the subject was corrected using the method of Xu et al. Neuroimage 2013.\n' \
        '\n'\
        'USAGE\n' \
        '  sct_dmri_moco.py <dmri> <bvecs>\n' \
        '\n'
    sys.exit(2)

#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # call main function
    main()
