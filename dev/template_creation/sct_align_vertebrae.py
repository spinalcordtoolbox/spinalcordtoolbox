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
        self.final_warp = ''
        self.compose = 0
        
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
    landmark = ''
    verbose = param.verbose
    output_name = 'aligned.nii.gz'
    template_landmark = ''
    final_warp = param.final_warp
    compose = param.compose
    transfo = 'affine'
        
    try:
         opts, args = getopt.getopt(sys.argv[1:],'hi:l:o:R:t:w:c:v:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts :
        if opt == '-h':
            usage()
        elif opt in ("-i"):
            fname = arg
        elif opt in ("-l"):
            landmark = arg       
        elif opt in ("-o"):
            output_name = arg  
        elif opt in ("-R"):
            template_landmark = arg
        elif opt in ("-t"):
            transfo = arg    
        elif opt in ("-w"):
            final_warp = arg
        elif opt in ("-c"):
            compose = int(arg)                            
        elif opt in ('-v'):
            verbose = int(arg)
    
    # display usage if a mandatory argument is not provided
    if fname == '' or landmark == '' or template_landmark == '' :
        usage()
        
    if final_warp not in ['','spline','NN']:
        usage()
        
    if transfo not in ['affine','bspline']:
        usage()       
    
    # check existence of input files
    print'\nCheck if file exists ...'
    
    sct.check_file_exist(fname)
    sct.check_file_exist(landmark)
    sct.check_file_exist(template_landmark)
    
    
        
    # Display arguments
    print'\nCheck input arguments...'
    print'  Input volume ...................... '+fname
    print'  Verbose ........................... '+str(verbose)

    
    print 'Creating cross using input landmarks\n...'
    sct.run('sct_label_utils.py -i ' + landmark + ' -o ' + 'cross_native.nii.gz -t cross ' )
    
    print 'Creating cross using template landmarks\n...'
    sct.run('sct_label_utils.py -i ' + template_landmark + ' -o ' + 'cross_template.nii.gz -t cross ' )
    
    if transfo == 'affine' :
        print 'Computing affine transformation between subject and destination landmarks\n...'
        sct.run('ANTSUseLandmarkImagesToGetAffineTransform cross_template.nii.gz cross_native.nii.gz affine n2t.txt')
        warping = 'n2t.txt'
        
    if transfo == 'bspline' :
        print 'Computing bspline transformation between subject and destination landmarks\n...'
        sct.run('ANTSUseLandmarkImagesToGetBSplineDisplacementField cross_template.nii.gz cross_native.nii.gz warp_ntotemp.nii.gz 5x5x5 3 2 0')    
        warping = 'warp_ntotemp.nii.gz'
        
    if final_warp == '' :    
        print 'Apply transfo to input image\n...'
        sct.run('WarpImageMultiTransform 3 ' + fname + ' ' + output_name + ' -R ' + template_landmark + ' ' + warping)
        
    if final_warp == 'NN':
        print 'Apply transfo to input image\n...'
        sct.run('WarpImageMultiTransform 3 ' + fname + ' ' + output_name + ' -R ' + template_landmark + ' ' + warping + ' --use-NN')
        
    if final_warp == 'spline':
        print 'Apply transfo to input image\n...'
        sct.run('WarpImageMultiTransform 3 ' + fname + ' ' + output_name + ' -R ' + template_landmark + ' ' + warping + ' --use-BSpline')
            
    
    if compose :
        
        print 'Computing affine transformation between subject and destination landmarks\n...'
        sct.run('ANTSUseLandmarkImagesToGetAffineTransform cross_template.nii.gz cross_native.nii.gz affine n2t.txt')
        warping_affine = 'n2t.txt'
        
        
        print 'Apply transfo to input landmarks\n...'
        sct.run('WarpImageMultiTransform 3 ' + cross_native + ' cross_affine.nii.gz -R ' + template_landmark + ' ' + warping_affine + ' --use-NN')
        
        print 'Computing transfo between moved landmarks and template landmarks\n...'
        sct.run('ANTSUseLandmarkImagesToGetBSplineDisplacementField cross_template.nii.gz cross_affine.nii.gz warp_affine2temp.nii.gz 5x5x5 3 2 0')    
        warping_bspline = 'warp_affine2temp.nii.gz'
        
        print 'Composing transformations\n...'
        sct.run('ComposeMultiTransform 3 warp_full.nii.gz -R ' + template_landmark + ' ' + warping_bspline + ' ' + warping_affine)
        warping_concat = 'warp_full.nii.gz'
        
        if final_warp == '' :    
            print 'Apply concat warp to input image\n...'
            sct.run('WarpImageMultiTransform 3 ' + fname + ' ' + output_name + ' -R ' + template_landmark + ' ' + warping_concat)
        
        if final_warp == 'NN':
            print 'Apply concat warp to input image\n...'
            sct.run('WarpImageMultiTransform 3 ' + fname + ' ' + output_name + ' -R ' + template_landmark + ' ' + warping_concat + ' --use-NN')
        
        if final_warp == 'spline':
            print 'Apply concat warp to input image\n...'
            sct.run('WarpImageMultiTransform 3 ' + fname + ' ' + output_name + ' -R ' + template_landmark + ' ' + warping_concat + ' --use-BSpline')
          
    
    
    print '\nFile created : ' + output_name
  
    
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
Takes an input volume, a mask containing labels in the spinalcord at several levels, a mask
containning the same labels in the space you want to push into. It registers your input image
using transformation between landmark images.

USAGE
  """+os.path.basename(__file__)+"""  -i <input_volume> -l <landmark_native> -R <template_landmark>

MANDATORY ARGUMENTS
  -i <input_volume>         input image. No Default value
  -l <landmark_native>      mask with labels. No Default Value
  -R <template_landmark>    mask with labels in template space. No Default Value


OPTIONAL ARGUMENTS
  -o <output_name>          output name. Default : aligned.nii.gz
  -t {affine,bspline}       type of initial transformation. Default : affine
  -w {NN,spline}            final warp interpolation. Default : trilinear
  -c {0,1}                  compose affine and bspline transformation. Default="""+str(param.compose)+"""
  -v {0,1}                  verbose. Default="""+str(param.verbose)+"""
  -h                        help. Show this message

EXAMPLE
  """+os.path.basename(__file__)+""" -i volume_image.nii.gz -l landmarks_native.nii.gz -R landmarks_template.nii.gz -t bspline -w NN\n"""

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






