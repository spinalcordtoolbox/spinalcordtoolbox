#!/usr/bin/env python
#########################################################################################
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Touati
# Created: 2014-08-11
#
# About the license: see the file LICENSE.TXT
#########################################################################################

#DEFAULT PARAMETERS
class param:
    ## The constructor
    def __init__(self):
        self.verbose = 1
# check if needed Python libraries are already installed or not
import sys
import getopt
import sct_utils as sct
import nibabel
import os
import time
from math import floor

def main():
    
    #Initialization
    fname = ''
    verbose = param.verbose
    start = ''
    end = ''
        
    try:
         opts, args = getopt.getopt(sys.argv[1:],'hi:v:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts :
        if opt == '-h':
            usage()
        elif opt in ("-i"):
            fname = arg                      
        elif opt in ('-v'):
            verbose = int(arg)
    
    # display usage if a mandatory argument is not provided
    if fname == '' :
        usage()
    
    # check existence of input files
    print'\nCheck if file exists ...'
    
    sct.check_file_exist(fname)
    
 
    # Display arguments
    print'\nCheck input arguments...'
    print'  Input volume ...................... '+fname
    print'  Verbose ........................... '+str(verbose)
    
    
    # create temporary folder
    path_tmp = 'tmp.'+time.strftime("%y%m%d%H%M%S")
    sct.run('mkdir '+path_tmp)
    
    fname = os.path.abspath(fname)
    path_data, file_data, ext_data = sct.extract_fname(fname)
    
    # copy files into tmp folder
    sct.run('cp '+fname+' '+path_tmp)

    # go to tmp folder
    os.chdir(path_tmp)
    
   
    # Get size of data
    print '\nGet dimensions of template...'
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname)
    print '.. '+str(nx)+' x '+str(ny)+' y '+str(nz)+' z '+str(nt)
    
    sct.run('sct_crop_image -i '+fname+' -o left.nii.gz -dim 0 -start '+str(int(0))+' -end '+str(int(floor(nx/2)-1)))
    sct.run('sct_crop_image -i '+fname+' -o right.nii.gz -dim 0 -start '+str(int(floor(nx/2)))+' -end '+str(int(nx-1)))
        
    right = nibabel.load('right.nii.gz')
    data_right = right.get_data()
    hdr_right = right.get_header()
    
    nx_r, ny_r, nz_r, nt_r, px_r, py_r, pz_r, pt_r = sct.get_dimension('right.nii.gz')
    
    mirror_right = data_right*0

    for i in xrange(nx_r):
        for j in xrange(ny_r):
            for k in xrange(nz_r):

                mirror_right[i,j,k] = data_right[(nx_r-1)-i,j,k]
    
  
    print '\nSave volume ...'
    
    img = nibabel.Nifti1Image(mirror_right, None, hdr_right)
    file_name = 'mirror_right.nii.gz'
    nibabel.save(img,file_name)
    
    sct.run ('fslcpgeom left.nii.gz mirror_right.nii.gz')
    
     #MI [fixed,moving]
     # 
    cmd = 'sct_antsRegistration \
    --dimensionality 3 \
    --transform Syn[0.5,3,0] \
    --metric MI[mirror_right.nii.gz,left.nii.gz,1,32] \
    --convergence 50x20 \
    --shrink-factors 4x1 \
    --smoothing-sigmas 1x1mm \
    --Restrict-Deformation 1x1x0 \
    --output [l2r,l2r.nii.gz]'

    status, output = sct.run(cmd)
    if verbose:
        print output
        
    #output are : l2r0InverseWarp.nii.gz l2r.nii.gz l2r0Warp.nii.gz    
        
    status, output = sct.run('sct_c3d -mcs l2r0Warp.nii.gz -oo l2rwarpx.nii.gz l2rwarpy.nii.gz l2rwarpz.nii.gz')   
    status, output = sct.run('sct_c3d -mcs l2r0InverseWarp.nii.gz -oo l2rinvwarpx.nii.gz l2rinvwarpy.nii.gz l2rinvwarpz.nii.gz')
    
    
    warpx = nibabel.load('l2rwarpx.nii.gz')
    data_warpx = warpx.get_data()
    
    warpy = nibabel.load('l2rwarpy.nii.gz')
    data_warpy = warpy.get_data()
    
    warpz = nibabel.load('l2rwarpz.nii.gz')
    data_warpz = warpz.get_data()
    
    invwarpx = nibabel.load('l2rinvwarpx.nii.gz')
    data_invwarpx = invwarpx.get_data()
    
    invwarpy = nibabel.load('l2rinvwarpy.nii.gz')
    data_invwarpy = invwarpy.get_data()
    
    invwarpz = nibabel.load('l2rinvwarpz.nii.gz')
    data_invwarpz = invwarpz.get_data()
    
    data_warpx = (data_warpx - data_warpx(end:-1:1,:)) / 2
    
    if verbose:
        print output
            
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION


USAGE
  """+os.path.basename(__file__)+"""  -i <input_volume> -s <start> -end <end>

MANDATORY ARGUMENTS
  -i <input_volume>         
  
OPTIONAL ARGUMENTS
  -v {0,1}                   verbose. Default="""+str(param.verbose)+"""
  -h                         help. Show this message
"""

    # exit program
    sys.exit(2)



#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = param()
    # call main function
    main()



