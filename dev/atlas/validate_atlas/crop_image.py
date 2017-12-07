#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Crop images
#This program extracts zslices from all of the images in a folder(folder_in) and outputs the cropped images in an output folder(folder_out)
# It can be used to extract certain zslices from the atlas, or the template

import os
import getopt
import sys
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, "scripts"))
import sct_utils as sct

def main():
    # params
    # Old atlas created from the registration of all slices to the reference slice
    # folder_in = '/home/django/cnaaman/data/data_marc/WMtracts_outputstest/final_results/' # path of atlas
    # New atlas created from the registration of all slices to the adjacent slice
    #folder_in = '/home/django/cnaaman/data/data_marc/WMtracts_outputsc_julien/final_results/'
    #folder_out = '/home/django/cnaaman/code/stage/cropped_atlas/'
    verbose = 1
    #zind = 10,110,210,310,410
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hf:o:z:') # define flag
    except getopt.GetoptError as err: # check if the arguments are defined
        print str(err) # error
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        if opt == '-f':
            folder_in = str(arg)
        if opt == '-o':
            folder_out = str(arg)
        if opt == '-z':
            zind = arg
            zind = zind.split(',')
    
    # create output folder
    if os.path.exists(folder_out):
        sct.printv('WARNING: Output folder already exists. Deleting it...', verbose)
        sct.run('rm -rf '+folder_out)
    sct.run('mkdir '+folder_out)

    # get atlas files
    status, output = sct.run('ls '+os.path.join(folder_in, '*.nii.gz') verbose)
    fname_list = output.split()


    # loop across atlas
    for i in xrange(0, len(fname_list)):
        path_list, file_list, ext_list = sct.extract_fname(fname_list[i])
        crop_file(fname_list[i], folder_out, zind)
    
def crop_file(fname_data, folder_out, zind):
    # extract file name
    path_list, file_list, ext_list = sct.extract_fname(fname_data)
   
   # crop file with fsl, and then merge back
    cmd = 'fslmerge -z '+os.path.join(folder_out, file_list)
    for i in zind:
        sct.run('fslroi '+fname_data+' z'+str(zind.index(i))+'_'+file_list+' 0 -1 0 -1 '+str(i)+' 1')
        cmd = cmd+' z'+str(zind.index(i))+'_'+file_list
    sct.run(cmd)

def usage():
    print '\n' \
        'crop_image\n' \
        '----------------------------------------------------------------------------------------------------------\n'\
        'DESCRIPTION\n' \
        ' This program extracts zslices from all of the images in a folder(folder_in) and outputs the cropped images in an output folder(folder_out) \n' \
        ' USAGE\n' \
        ' crop_image_folder -f input_folder -o output_folder -z z_ind  \n' \
        ' Example : crop_image.py -f folder_img/ -o cropped_img1/ -z 10,110,210,310,410  \n' 
    sys.exit(2)

if __name__ == "__main__":
    # call main function
    main()
