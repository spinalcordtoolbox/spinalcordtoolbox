#!/usr/bin/env python
__author__ = 'slevy'
# Simon LEVY, Julien Cohen-Adad
# Created: 2014-10-30

import getopt
import sys
import commands
import time
import os
# Get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# Append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')
import sct_utils as sct
from msct_parser import *

class Param:
    def __init__(self):
        self.debug = 0
        self.file_fname_output = 'b1_smoothed'

#=======================================================================================================================
# main
#=======================================================================================================================
def main():



    # Check input parameters
    parser = Parser(__file__)
    parser.usage.set_description('compute Ialpha/I2*alpha')
    parser.add_option("-d", "file", "image you want to crop", True, "t2.nii.gz")
    parser.add_option("-i", "str", "Two NIFTI : flip angle alpha and 2*alpha", True, "ep_fa60.nii.gz,ep_fa120.nii.gz")
    usage = parser.usage.generate()

    if param.debug:
        # Parameters for debug mode
        sct.printv('\n*** WARNING: DEBUG MODE ON ***\n', type='warning')
        os.chdir('/Volumes/users_hd2/tanguy/data/Boston/2014-07/Connectome/MS_SC_002/MTV')
        fname_spgr10 = 'spgr10.nii.gz'
        epi_fnames = 'b1/ep60.nii.gz,b1/ep120.nii.gz'
    else:
        arguments = parser.parse(sys.argv[1:])
        # Initialization of variables
        fname_spgr10 = arguments["-d"]
        epi_fnames   = arguments["-i"]



    # Parse inputs to get the actual data
    epi_fname_list = epi_fnames.split(',')

    # Extract path, file names and extensions
    path_epi, fname_epi, ext_epi = sct.extract_fname(epi_fname_list[0])

    # Create temporary folders and go in it
    sct.printv('Create temporary folder...')
    path_tmp = 'tmp_'+time.strftime("%y%m%d%H%M%S")
    sct.create_folder(path_tmp)
    os.chdir(path_tmp)
    fname_spgr10='../'+fname_spgr10

    # Compute the half ratio of the 2 epi (Saturated Double-Angle Method for Rapid B1 Mapping - Cunningham)
    fname_half_ratio = '../'+path_epi+'epi_half_ratio'
    sct.run('fslmaths -dt double ../'+epi_fname_list[0]+' -div 2 -div ../'+epi_fname_list[1]+' '+fname_half_ratio)



    # Smooth this half ratio slice-by-slice
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension('../'+epi_fname_list[0])
    # split slices
    sct.run('fslsplit '+fname_half_ratio+' -z')
    # 2D median filtering of each slice
    vol_list=''
    for slice in range(0, nz):
        sct.run('fslmaths vol'+str(slice).zfill(4)+' -kernel boxv 7x7x1 -fmedian vol'+str(slice).zfill(4)+'_median_smoothed')
        vol_list += 'vol'+str(slice).zfill(4)+'_median_smoothed '

    # merge volumes
    fname_half_ratio_smoothed = fname_half_ratio+'_smooth'
    sct.run('fslmerge -z '+fname_half_ratio_smoothed+' '+vol_list)

    # compute angle
    sct.run('fslmaths '+fname_half_ratio_smoothed+' -acos ../'+path_epi+'B1angle')

    sct.printv('\tDone.')

    # Remove temporary folder
    os.chdir('..')
    sct.run('rm -rf '+path_tmp)

#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # Construct object fro class 'param'
    param = Param()
    # Call main function
    main()

