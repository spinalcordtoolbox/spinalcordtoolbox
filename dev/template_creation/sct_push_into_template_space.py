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
import os
from commands import getstatusoutput
def main():
    
    
    # get path of the toolbox
    status, path_sct = getstatusoutput('echo $SCT_DIR')
    #print path_sct


    #Initialization
    fname = ''
    landmarks_native = ''
    landmarks_template = path_sct + '/dev/template_creation/template_landmarks-mm.nii.gz'
    reference = path_sct + '/dev/template_creation/template_shape.nii.gz'
    verbose = param.verbose
        
    try:
         opts, args = getopt.getopt(sys.argv[1:],'hi:n:t:R:v:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts :
        if opt == '-h':
            usage()
        elif opt in ("-i"):
            fname = arg   
        elif opt in ("-n"):
            landmarks_native = arg
        elif opt in ("-t"):
            landmarks_template = arg
        elif opt in ("-R"):
            reference = arg                
        elif opt in ('-v'):
            verbose = int(arg)
    
    # display usage if a mandatory argument is not provided
    if fname == '' :
        usage()
    
    # check existence of input files
    print'\nCheck if file exists ...'
    
    sct.check_file_exist(fname)
    sct.check_file_exist(landmarks_native)
    sct.check_file_exist(landmarks_template)
    sct.check_file_exist(reference)
    
    path_input, file_input, ext_input = sct.extract_fname(fname)
    
        
    output_name = path_input + file_input + '_2temp' + ext_input
    print output_name
    transfo = 'native2temp.txt'
    # Display arguments
    print'\nCheck input arguments...'
    print'  Input volume ...................... '+fname
    print'  Landmarks in native space ...................... '+landmarks_native
    print'  Landmarks in template space ...................... '+landmarks_template
    print'  Reference ...................... '+reference
    print'  Verbose ........................... '+str(verbose)


    print '\nEstimate rigid transformation between paired landmarks...'
    sct.run('ANTSUseLandmarkImagesToGetAffineTransform ' + landmarks_template + ' '+ landmarks_native + ' affine ' + transfo)
    
    # Apply rigid transformation
    print '\nApply affine transformation to native landmarks...'
    sct.run('WarpImageMultiTransform 3 ' + fname + ' ' + output_name + ' -R ' + reference + ' ' + transfo)
    
    print '\nFile created : ' + output_name
  
    
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION

Projects the anatomical image into the template space using landmark image generated with
sct_create_cross.py Also need empty image with template diemnsion as reference

USAGE
  """+os.path.basename(__file__)+"""  -i <input_volume> -n <anatomical_landmarks> -t <template_landmarks> -R <reference>

MANDATORY ARGUMENTS
  -i <input_volume>         input straight cropped volume. No Default value
  -n <anatomical_landmarks> landmarks in native space. See sct_create_cross.py
  -t <template_landmarks>   landmarks in template_space. See sct_create_croos.py 
  -R <reference>            Reference image. Empty template image
  
OPTIONAL ARGUMENTS
  -v {0,1}                   verbose. Default="""+str(param.verbose)+"""
  -h                         help. Show this message

EXAMPLE
  """+os.path.basename(__file__)+""" -i volume_image.nii.gz -n native_landmarks.nii.gz -t native_template.nii.gz -R template_shape.nii.gz\n"""

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






