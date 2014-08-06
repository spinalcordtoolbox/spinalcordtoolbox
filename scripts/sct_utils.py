#!/usr/bin/env python
#########################################################################################
#
# Module containing several useful functions.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
# Modified: 2014-07-01
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os
import sys
import commands

# TODO: under run(): add a flag "ignore error" for ComposeMultiTransform
# TODO: check if user has bash or t-schell for fsloutput definition

fsloutput = 'export FSLOUTPUTTYPE=NIFTI; ' # for faster processing, all outputs are in NIFTI'



#==============e=========================================================================================================
# run
#=======================================================================================================================
# Run UNIX command
def run(cmd, verbose=1):
    if verbose:
        print('>> ' + cmd)
    status, output = commands.getstatusoutput(cmd)
    if status != 0:
        print('\nERROR!!! \n'+output+'\nExit program.\n')
        sys.exit(2)
    else:
        return status, output


#=======================================================================================================================
# extract_fname
#=======================================================================================================================
# Extract path, file and extension
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



#=======================================================================================================================
# check_file_exist
#=======================================================================================================================
# Check existence of a file
def check_file_exist(fname, verbose=1):

    if os.path.isfile(fname) or os.path.isfile(fname + '.nii') or os.path.isfile(fname + '.nii.gz'):
        if verbose:
            print('  OK: '+fname)
        pass
    else:
        print('  ERROR: ' + fname + ' does not exist. Exit program.\n')
        sys.exit(2)



#=======================================================================================================================
# get_dimension
#=======================================================================================================================
# Get dimensions of a nifti file using FSL
def get_dimension(fname):
    # apply fslsize on data
    cmd = 'fslsize '+fname
    status, output = commands.getstatusoutput(cmd)
    # split output according to \n field
    output_split = output.split()
    # extract dimensions as integer
    nx = int(output_split[1])
    ny = int(output_split[3])
    nz = int(output_split[5])
    nt = int(output_split[7])
    px = float(output_split[9])
    py = float(output_split[11])
    pz = float(output_split[13])
    pt = float(output_split[15])
    return nx, ny, nz, nt, px, py, pz, pt



#=======================================================================================================================
# get_orientation
#=======================================================================================================================
# Get orientation of a nifti file
def get_orientation(fname):
    status, output = commands.getstatusoutput('sct_orientation -get -i '+fname)
    orientation = output.replace('Input image orientation : ','')
    return orientation



#=======================================================================================================================
# generate_output_file
#=======================================================================================================================
# Generate output file (put the extension for input file!!!)
def generate_output_file(fname_in, path_out, file_out, ext_out):
    # import stuff
    import shutil  # for moving files
    # extract input file extension
    path_in, file_in, ext_in = extract_fname(fname_in)
    # if (i) output path is not local and (ii) output file already exists in nii or nii.gz format, delete it and move file
    if not path_out == '' and not path_in == path_out:
        if os.path.isfile(path_out+file_out+'.nii'):
            os.system('rm '+path_out+file_out+'.nii')
        if os.path.isfile(path_out+file_out+'.nii.gz'):
            os.system('rm '+path_out+file_out+'.nii.gz')
    # Move file to output folder (keep the same extension as input)
    shutil.move(fname_in, path_out+file_out+ext_in)
    # convert to nii (only if necessary)
    if ext_out == '.nii' and ext_in != '.nii':
        os.system('fslchfiletype NIFTI '+path_out+file_out)
    # convert to nii.gz (only if necessary)
    if ext_out == '.nii.gz' and ext_in != '.nii.gz':
        os.system('fslchfiletype NIFTI_GZ '+path_out+file_out)
    # display message
    print '.. File created: '+path_out+file_out+ext_out
    return path_out+file_out+ext_out


#=======================================================================================================================
# sign
#=======================================================================================================================
# Get the sign of a number. Returns 1 if x>=0 and -1 if x<0
def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


#=======================================================================================================================
# check_if_installed
#=======================================================================================================================
# check if dependant software is installed
def check_if_installed(cmd, name_software):
    status, output = commands.getstatusoutput(cmd)
    if status != 0:
        print('\nERROR: '+name_software+' is not installed.\nExit program.\n')
        sys.exit(2)


#=======================================================================================================================
# printv: enables to print or not, depending on verbose status
#=======================================================================================================================
def printv(string, verbose=1):
    if verbose:
        print(string)



#=======================================================================================================================
# slash_at_the_end: make sure there is (or not) a slash at the end of path name
#=======================================================================================================================
def slash_at_the_end(path, slash=0):
    if slash == 0:
        if path[-1:] == '/':
            path = path[:-1]
    if slash == 1:
        if not path[-1:] == '/':
            path = path+'/'
    return path