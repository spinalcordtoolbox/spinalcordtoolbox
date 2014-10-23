#!/usr/bin/env python
#
# This program returns the grey matter segmentation given anatomical, landmarks and t2star images
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener, Augustin Roux
# Created: 2014-10-18
#
# About the license: see the file LICENSE.TXT
#########################################################################################
import sct_utils as sct
import os
import sys
import getopt


def main():
    fname_anat = ''
    fname_landmarks = ''
    fname_t2star = ''
    fname_grayMatter_template = ''
    fname_seg = ''

    # Check input param
    try:
        opts, args = getopt.getopt(sys.argv[1:],'hi:t:l:h:s:')
    except getopt.GetoptError as err:
        print str(err)
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ('-i'):
            fname_anat = arg
        elif opt in ('-t'):
            fname_t2star = arg
        elif opt in ('-l'):
            fname_landmarks = arg
        elif opt in ('-s'):
            fname_seg = arg

    # register template to anat file: this generate warp_template2anat.nii
    cmd = "sct_register_to_template -i " + fname_anat + " -l " + fname_landmarks + " -m " + fname_seg + " -s normal"
    sct.run(cmd)
    warp_template2anat = ""

    # register anat file to t2star: generate  warp_anat2t2star
    cmd = "sct_register_multimodal -i " + fname_anat + " -d " + fname_t2star
    sct.run(cmd)
    warp_anat2t2star = ""

    # concatenation of the two warping fields
    warp_template2t2star = "warp_template2t2star.nii.gz"
    cmd = "sct_concat_transfo -w " + warp_template2anat + ',' + warp_anat2t2star + " -d " + fname_t2star + " -o " + warp_template2t2star
    sct.run(cmd)

    # apply the concatenated warping field to the template
    cmd = "sct_warp_template -d " + fname_t2star + " -w " + warp_template2anat + " -s 1 -o template_in_t2star_space"
    sct.run(cmd)


    #sct_register_to_template --> warp_template2anat
    # register anat file to t2star
    #sct_register_multimodal --> warp_anat2t2star
    # concatenate warp_template2anat with warp_anat2t2star
    #--> warp_template2t2star





    # registration of the grey matter
    cmd = "sct_antsRegistration --dimensionality 3 --transform BSplineSyN[0.8,3,0] ", \
          "--metric MI[data4d_mean_in_seg_denoised.nii,MNI-Poly-AMU_GM_reg.nii.gz,1,32] ", \
          "--convergence 20x15 --shrink-factors 2x1 --smoothing-sigmas 0mm --Restrict-Deformation 1x1x0 ", \
          "--output [regSeg,regSeg_1.nii.gz]"

    return


#=======================================================================================================================
# usage
#=======================================================================================================================
def usage():
    print '\n' \
        ''+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        '  This function returns the grey matter segmentation.\n' \
        '\n'\
        'USAGE\n' \
        '  '+os.path.basename(__file__)+' -i <anat volume> -t <t2star volume> -l <landmarks>\n' \
        '\n'\
        'MANDATORY ARGUMENTS\n' \
        '  -i                   input anatomical volume.\n' \
        '  -t                   t2star volume.\n' \
        '  -l                   landmarks at spinal cord center.' \
        '\n'\
        'OPTIONAL ARGUMENTS\n' \
        '  -s <segmentation>    spinal cord segmentation.\n'
    sys.exit(2)



if __name__ == "__main__":
    # initialize parameters
    # param = param()
    # call main function
    main()


