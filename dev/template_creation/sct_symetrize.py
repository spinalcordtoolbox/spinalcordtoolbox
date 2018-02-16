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
import sys, io, os, getopt, time, shutil
from math import floor

import nibabel

import sct_utils as sct

def main():
    
    #Initialization
    fname = ''
    verbose = param.verbose
        
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
    print('\nCheck if file exists ...')

    sct.check_file_exist(fname)


    # Display arguments
    print('\nCheck input arguments...')
    print('  Input volume ...................... '+fname)
    print('  Verbose ........................... '+str(verbose))


    path_tmp = sct.tmp_create(basename="symetrize", verbose=verbose)

    fname = os.path.abspath(fname)
    path_data, file_data, ext_data = sct.extract_fname(fname)

    # copy files into tmp folder
    sct.copy(fname, path_tmp)

    # go to tmp folder
    os.chdir(path_tmp)


    # Get size of data
    print '\nGet dimensions of template...'
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname)
    print '.. '+str(nx)+' x '+str(ny)+' y '+str(nz)+' z '+str(nt)

    # extract left side and right side
    sct.run('sct_crop_image -i '+fname+' -o left.nii.gz -dim 0 -start '+str(int(0))+' -end '+str(int(floor(nx/2)-1)))
    sct.run('sct_crop_image -i '+fname+' -o right.nii.gz -dim 0 -start '+str(int(floor(nx/2)))+' -end '+str(int(nx-1)))

    # create mirror right
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

    #copy header of left to mirror right
    sct.run ('fslcpgeom left.nii.gz mirror_right.nii.gz')



    # compute transfo from left to mirror right
     #MI [fixed,moving]
    ### Beause it takes time there's a output that were computed on guillimin /home/django/jtouati/data/test_templateANTS/final_preprocessed/MI/test/tmp.141015123447

     #
    cmd = 'isct_antsRegistration \
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

    # separate the 2 warping fields along the 3 directions
    status, output = sct.run('isct_c3d -mcs l2r0Warp.nii.gz -oo l2rwarpx.nii.gz l2rwarpy.nii.gz l2rwarpz.nii.gz')
    status, output = sct.run('isct_c3d -mcs l2r0InverseWarp.nii.gz -oo l2rinvwarpx.nii.gz l2rinvwarpy.nii.gz l2rinvwarpz.nii.gz')
    print 'Loading ..'
    # load warping fields
    warpx = nibabel.load('l2rwarpx.nii.gz')
    data_warpx = warpx.get_data()
    hdr_warpx=warpx.get_header()

    warpy = nibabel.load('l2rwarpy.nii.gz')
    data_warpy = warpy.get_data()
    hdr_warpy=warpy.get_header()

    warpz = nibabel.load('l2rwarpz.nii.gz')
    data_warpz = warpz.get_data()
    hdr_warpz=warpz.get_header()

    invwarpx = nibabel.load('l2rinvwarpx.nii.gz')
    data_invwarpx = invwarpx.get_data()
    hdr_invwarpx=invwarpx.get_header()

    invwarpy = nibabel.load('l2rinvwarpy.nii.gz')
    data_invwarpy = invwarpy.get_data()
    hdr_invwarpy=invwarpy.get_header()

    invwarpz = nibabel.load('l2rinvwarpz.nii.gz')
    data_invwarpz = invwarpz.get_data()
    hdr_invwarpz=invwarpz.get_header()
    print 'Creating..'
    # create demi warping fields
    data_warpx = (data_warpx - data_warpx[::-1,:,:])/2
    data_warpy = (data_warpy + data_warpy[::-1,:,:])/2
    data_warpz = (data_warpz + data_warpz[::-1,:,:])/2
    data_invwarpx = (data_invwarpx - data_invwarpx[::-1,:,:])/2
    data_invwarpy = (data_invwarpy + data_invwarpy[::-1,:,:])/2
    data_invwarpz = (data_invwarpz + data_invwarpz[::-1,:,:])/2
    print 'Saving ..'
    # save demi warping fields
    img = nibabel.Nifti1Image(data_warpx, None, hdr_warpx)
    file_name = 'warpx_demi.nii.gz'
    nibabel.save(img,file_name)

    img = nibabel.Nifti1Image(data_warpy, None, hdr_warpy)
    file_name = 'warpy_demi.nii.gz'
    nibabel.save(img,file_name)

    img = nibabel.Nifti1Image(data_warpz, None, hdr_warpz)
    file_name = 'warpz_demi.nii.gz'
    nibabel.save(img,file_name)

    img = nibabel.Nifti1Image(data_invwarpx, None, hdr_invwarpx)
    file_name = 'invwarpx_demi.nii.gz'
    nibabel.save(img,file_name)

    img = nibabel.Nifti1Image(data_invwarpy, None, hdr_invwarpy)
    file_name = 'invwarpy_demi.nii.gz'
    nibabel.save(img,file_name)

    img = nibabel.Nifti1Image(data_invwarpz, None, hdr_invwarpz)
    file_name = 'invwarpz_demi.nii.gz'
    nibabel.save(img,file_name)
    print 'Copy ..'
    # copy transform
    status,output = sct.run('isct_c3d l2rwarpx.nii.gz warpx_demi.nii.gz -copy-transform -o warpx_demi.nii.gz')
    status,output = sct.run('isct_c3d l2rwarpy.nii.gz warpy_demi.nii.gz -copy-transform -o warpy_demi.nii.gz')
    status,output = sct.run('isct_c3d l2rwarpz.nii.gz warpz_demi.nii.gz -copy-transform -o warpz_demi.nii.gz')
    status,output = sct.run('isct_c3d l2rinvwarpx.nii.gz invwarpx_demi.nii.gz -copy-transform -o invwarpx_demi.nii.gz')
    status,output = sct.run('isct_c3d l2rinvwarpy.nii.gz invwarpy_demi.nii.gz -copy-transform -o invwarpy_demi.nii.gz')
    status,output = sct.run('isct_c3d l2rinvwarpz.nii.gz invwarpz_demi.nii.gz -copy-transform -o invwarpz_demi.nii.gz')
    
    # combine warping fields
    print 'Combine ..'
    sct.run('isct_c3d warpx_demi.nii.gz warpy_demi.nii.gz warpz_demi.nii.gz -omc 3 warpl2r_demi.nii.gz')
    sct.run('isct_c3d invwarpx_demi.nii.gz invwarpy_demi.nii.gz invwarpz_demi.nii.gz -omc 3 invwarpl2r_demi.nii.gz')
    
    #warpl2r_demi.nii.gz invwarpl2r_demi.nii.gz
    
    # apply demi warping fields
    sct.run('sct_apply_transfo -i left.nii.gz -d left.nii.gz -w warpl2r_demi.nii.gz -o left_demi.nii.gz')
    sct.run('sct_apply_transfo -i mirror_right.nii.gz -d mirror_right.nii.gz -w invwarpl2r_demi.nii.gz -o mirror_right_demi.nii.gz')
    
    #unmirror right
    
    demi_right = nibabel.load('mirror_right_demi.nii.gz')
    data_demi_right = demi_right.get_data()
    hdr_demi_right = demi_right.get_header()
    
    nx_r, ny_r, nz_r, nt_r, px_r, py_r, pz_r, pt_r = sct.get_dimension('mirror_right_demi.nii.gz')
    
    unmirror_right = data_demi_right*0

    for i in xrange(nx_r):
        for j in xrange(ny_r):
            for k in xrange(nz_r):

                unmirror_right[i,j,k] = data_demi_right[(nx_r-1)-i,j,k]
    
    print '\nSave volume ...'
    
    img = nibabel.Nifti1Image(unmirror_right, None, hdr_right)
    file_name = 'un_mirror_right.nii.gz'
    nibabel.save(img,file_name)
    
    
    sct.run('fslmaths left_demi.nii.gz -add un_mirror_right.nii.gz symetrize_template.nii.gz')
    
    
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION


USAGE
  """+os.path.basename(__file__)+"""  -i <input_volume> 

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



