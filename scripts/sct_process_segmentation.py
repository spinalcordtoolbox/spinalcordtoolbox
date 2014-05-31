#!/usr/bin/env python
#########################################################################################
#
# Perform various types of processing from the spinal cord segmentation (e.g. extract centerline, compute CSA, etc.).
# (extract_centerline) extract the spinal cord centerline from the segmentation. Output file is an image in the same
# space as the segmentation.
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener
# Created: 2014-05-24
# Modified: 2014-05-26 - made it working (Simon Levy)
#
# About the license: see the file LICENSE.TXT
#########################################################################################



# DEFAULT PARAMETERS



class param:
    ## The constructor
    def __init__(self):
        self.debug              = 1
        self.verbose            = 0 # verbose

import re
import math
import sys
import getopt
import os
import numpy
import time
import sct_utils as sct
from sct_straighten_spinalcord import b_spline_centerline
from sct_nurbs import NURBS

try:
    import nibabel
except ImportError:
    print '--- nibabel not installed! Exit program. ---'
    sys.exit(2)

# MAIN
# ==========================================================================================
def main():

    # Initialization
    path_script = os.path.dirname(__file__)
    fsloutput = 'export FSLOUTPUTTYPE=NIFTI; ' # for faster processing, all outputs are in NIFTI
    # THIS DOES NOT WORK IN MY LAPTOP: path_sct = os.environ['SCT_DIR'] # path to spinal cord toolbox
    path_sct = path_script[:-8] # TODO: make it cleaner!
    fname_segmentation = ''
    name_process = ''
    processes = ['extract_centerline']
    verbose = param.verbose
    start_time = time.time()

    # Parameters for debug mode
    if param.debug:
        fname_segmentation = os.path.expanduser("~")+'/code/spinalcordtoolbox_dev/testing/data/errsm_23/t2/t2_manual_segmentation.nii.gz'
        verbose = 1
    
    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],'hi:p:v:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ("-i"):
            fname_segmentation = arg
	elif opt in ("-p"):
            name_process = arg
        elif opt in ('-v'):
            verbose = int(arg)

    # display usage if a mandatory argument is not provided
    if fname_segmentation == '' or name_process == '':
        usage()
	
    # display usage if the requested process is not available
    if name_process not in processes:
	usage()
	
    # check existence of input files
    sct.check_file_exist(fname_segmentation)
	
    # print arguments
    print '\nCheck parameters:'
    print '.. segmentation file:             '+fname_segmentation
	
    if name_process == 'extract_centerline':
	extract_centerline(fname_segmentation)

    # display elapsed time
    elapsed_time = time.time() - start_time
    print '\nFinished! Elapsed time: '+str(int(round(elapsed_time)))+'s'
	
    # End of Main


# EXTRACT_CENTERLINE
# ==========================================================================================

def extract_centerline(fname_segmentation):
    # Extract path, file and extension
    path_data, file_data, ext_data = sct.extract_fname(fname_segmentation)
	
    # create temporary folder
    path_tmp = 'tmp.'+time.strftime("%y%m%d%H%M%S")
    sct.run('mkdir '+path_tmp)

    # copy files into tmp folder
    sct.run('cp '+fname_segmentation+' '+path_tmp)

    # go to tmp folder
    os.chdir(path_tmp)
	
    # Change orientation of the input segmentation into RPI
    print '\nOrient segmentation image to RPI orientation...'
    fname_segmentation_orient = 'tmp.segmentation_rpi' + ext_data
    sct.run('sct_orientation -i ' + file_data+ext_data + ' -o ' + fname_segmentation_orient + ' -orientation RPI')
	
    # Extract orientation of the input segmentation
    status,sct_orientation_output = sct.run('sct_orientation -i ' + file_data+ext_data + ' -get')
    orientation = sct_orientation_output[-3:]
    print '\nOrientation of segmentation image: ' + orientation
	
    # Get size of data
    print '\nGet dimensions data...'
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_segmentation_orient)
    print '.. '+str(nx)+' x '+str(ny)+' y '+str(nz)+' z '+str(nt)
	
    print '\nOpen segmentation volume...'
    file = nibabel.load(fname_segmentation_orient)
    data = file.get_data()
    hdr = file.get_header()
	
    # Extract min and max index in Z direction
    X, Y, Z = (data>0).nonzero()
    min_z_index, max_z_index = min(Z), max(Z)

    x_centerline = [0 for i in range(0,max_z_index-min_z_index+1)]
    y_centerline = [0 for i in range(0,max_z_index-min_z_index+1)]
    z_centerline = [iz for iz in range(min_z_index, max_z_index+1)]
    # Extract segmentation points and average per slice
    for iz in range(min_z_index, max_z_index+1):
        x_seg, y_seg = (data[:,:,iz]>0).nonzero()
        x_centerline[iz-min_z_index] = numpy.mean(x_seg)
        y_centerline[iz-min_z_index] = numpy.mean(y_seg)
	
    for k in range(len(X)):
	    data[X[k],Y[k],Z[k]] = 0
	
    # Fit the centerline points with splines and return the new fitted coordinates
    x_centerline_fit, y_centerline_fit,x_centerline_deriv,y_centerline_deriv,z_centerline_deriv = b_spline_centerline(x_centerline,y_centerline,z_centerline)


    # Create an image with the centerline
    for iz in range(min_z_index, max_z_index+1):
	    data[round(x_centerline_fit[iz-min_z_index]),round(y_centerline_fit[iz-min_z_index]),iz] = 1
	
    # Write the centerline image in RPI orientation
    hdr.set_data_dtype('uint8') # set imagetype to uint8
    print '\nWrite NIFTI volumes...'
    img = nibabel.Nifti1Image(data, None, hdr)
    nibabel.save(img, 'tmp.centerline.nii')
    sct.generate_output_file('tmp.centerline.nii','./',file_data+'_centerline',ext_data)
	
    del data
	
    # come back to parent folder
    os.chdir('..')
	
    # Change orientation of the output centerline into input orientation
    print '\nOrient centerline image to input orientation: ' + orientation
    fname_segmentation_orient = 'tmp.segmentation_rpi' + ext_data
    sct.run('sct_orientation -i ' + path_tmp+'/'+file_data+'_centerline'+ext_data + ' -o ' + file_data+'_centerline'+ext_data + ' -orientation ' + orientation)
	

   # Remove temporary files
    print('\nRemove temporary files...')
    sct.run('rm -rf '+path_tmp)


    # to view results
    print '\nTo view results, type:'
    print 'fslview '+file_data+'_centerline &\n'
	
    # End of extract_centerline
	
# Print usage
# ==========================================================================================
def usage():
    print '\n' \
        ''+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        '  Perform various types of processing from the spinal cord segmentation (e.g. extract centerline, compute CSA,' \
        ' etc.).\n' \
        '\n' \
        'USAGE\n' \
        '  '+os.path.basename(__file__)+' -i <segmentation> -p <process>\n' \
        '\n' \
        'MANDATORY ARGUMENTS\n' \
        '  -i <segmentation>          segmentation data\n' \
        '  -p <process>               process to perform {extract_centerline}\n' \
        '\n' \
        'OPTIONAL ARGUMENTS\n' \
        '  -v <0,1>                   verbose. Default='+str(param.verbose)+'.\n'

    # exit program
    sys.exit(2)

# START PROGRAM
# =========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = param()
    # call main function
    main()
