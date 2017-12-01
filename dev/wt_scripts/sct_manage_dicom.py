#!/usr/bin/env python
#########################################################################################
# Convert dicom images into nifti format with dcm2nii
#
# Only for montreal data (anatomical, zoom, dezoom)
# For nonzoom, merge b0 and dwi
#
# USAGE
# ---------------------------------------------------------------------------------------
#   sct_manage_dicom.py <dicom_dir> <nifti_dir>
#
#
# INPUT
# ---------------------------------------------------------------------------------------
# dicom_dir         input directory with dicom files
# nifti_dir         directory where nifti files will be created
#
#
# OUTPUT
# ---------------------------------------------------------------------------------------
# none
#
#
# DEPENDENCIES
# ---------------------------------------------------------------------------------------
#
# EXTERNAL SOFTWARE
# - dcm2nii <http://www.mccauslandcenter.sc.edu/mricro/mricron/dcm2nii.html>
# - FSL (fslmerge) <http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Fslutils>
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuropoly.info>
# Author: William Thong
# Modified: 2013-11-07 16:50
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os, glob, sys, shutil

#dicom_path = '/Volumes/data_shared/montreal/errsm_22/'
#output_path = '/home/django/williamthong/data/multisite_dti/data_montreal/subject_test/'
dicom_path = sys.argv[1]
output_path = sys.argv[2]

#check existence of directories (if exists, removes subdirectories; if not, creates directory)
if not os.path.exists(output_path):
    os.makedirs(output_path)
else:
    shutil.rmtree(output_path)
    os.makedirs(output_path)

if not os.path.exists(dicom_path):
    print ('Directory '+dicom_path+' does not exist')
    sys.exit()

#normalize paths (if required remove trailing slash)
dicom_path = os.path.normpath(dicom_path)
output_path = os.path.normpath(output_path)

#specific string pattern
str_zoom = 'ep2d_diff_2drf_1mm'
str_nonzoom = 'ep2d_diff_p2'
str_anat = 'tse_spc_1mm_p2_FOV384__top'

def main():
    
    #initialization
    zoom=[]
    nonzoom_b0=[]
    nonzoom_dwi=[]
    anat=[]
    
    listfolders = os.listdir(dicom_path)
    
    #search for the forders of interest
    for (i,folder) in enumerate(listfolders):
        if folder.find(str_zoom) > -1 and folder.find('descending') == -1:
            zoom.append(folder)
        if folder.find(str_nonzoom) > -1 and folder.find('b0') > -1:
            nonzoom_b0.append(folder)
        if folder.find(str_nonzoom) > -1 and folder.find('b0') == -1:
            nonzoom_dwi.append(folder)
        if folder.find(str_anat) > -1:
            anat.append(folder)
    
    #convert anat
    output_anat = output_path+os.path.sep+'anat'
    os.mkdir(output_anat)
    cmd = 'dcm2nii -o '+output_anat+' ' + os.path.join(dicom_path, anat[0])
    print(">> "+cmd)
    os.system(cmd)
    #rename nifti
    rename_file(output_anat, '.nii.gz', 'anat.nii.gz')
    
    #convert zoom
    for (i, folder) in enumerate(zoom):
        output_zoom = os.path.join(output_path, 'zoom'+str(i+1))
        os.mkdir(output_zoom)
        #create nifti
        cmd = 'dcm2nii -o '+output_zoom+' ' + os.path.join(dicom_path, zoom[i])
        print(">> "+cmd)
        os.system(cmd)
        #rename nifti
        rename_file(output_zoom, '.nii.gz', 'dmri.nii.gz')
        #rename bvecs bvals
        rename_file(output_zoom, '.bvec', 'bvecs.txt')
        rename_file(output_zoom, '.bval', 'bvals.txt')
        
    
    #convert nonzoom
    output_nonzoom = output_path+os.path.sep+'nonzoom1'
    os.mkdir(output_nonzoom)
    
    merge_paths=[]
    
    #b0
    for (i, folder) in enumerate(nonzoom_b0):
        cmd = 'dcm2nii -o '+output_nonzoom+' '+dicom_path+os.path.sep+nonzoom_b0[i]
        print(">> "+cmd)
        os.system(cmd)
        #rename nifti
        rename_file(output_nonzoom, '.nii.gz','tmp.'+str(i)+'.nii.gz')
        merge_paths.append(output_nonzoom+os.path.sep+'tmp.'+str(i)+'.nii.gz')
    
    #dwi
    cmd = 'dcm2nii -o '+output_nonzoom+' '+dicom_path+os.path.sep+nonzoom_dwi[0]
    print(">> "+cmd)
    os.system(cmd)
    #rename nifti
    rename_file(output_nonzoom, '.nii.gz', 'tmp.dwi.nii.gz')
    merge_paths.append(output_nonzoom+os.path.sep+'tmp.dwi.nii.gz')
    #rename bvecs bvals
    rename_file(output_nonzoom, '.bvec', 'bvecs.txt')
    rename_file(output_nonzoom, '.bval', 'bvals.txt')
    
    #merge
    cmd = 'fslmerge -t '+output_nonzoom+os.path.sep+'dmri.nii.gz '+' '.join(merge_paths)
    print(">> "+cmd)
    os.system(cmd)
    
    # Delete temporary files
    print('\nDelete temporary files...')
    cmd = 'rm '+output_nonzoom+os.path.sep+'tmp.*'
    print(">> "+cmd)
    os.system(cmd)

def rename_file(path,name_in, name_out):
    tmp_name =  glob.glob(path+os.path.sep+'*'+name_in)
    print('Rename '+tmp_name[0]+' to '+path+os.path.sep+name_out)
    os.rename(tmp_name[0], path+os.path.sep+name_out)

if __name__ == "__main__":
    main()
