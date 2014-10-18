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


def main():
    fname_anat = ''
    fname_landmarks = ''
    fname_t2star = ''
    fname_grayMatter_template = ''


    # register template to anat file
    cmd = "sct_register_to_template -i " + fname_anat + " -l " +
    sct_register_to_template -i t2_cropped.nii.gz -l t2_cropped_landmarks_C2_T2.nii.gz -m t2_cropped_seg.nii.gz -s normal

    sct_register_to_template --> warp_template2anat
    # register anat file to t2star
    sct_register_multimodal --> warp_anat2t2star
    # concatenate warp_template2anat with warp_anat2t2star
    --> warp_template2t2star





    # registration of the grey matter
    cmd = "sct_antsRegistration --dimensionality 3 --transform BSplineSyN[0.8,3,0] ", \
          "--metric MI[data4d_mean_in_seg_denoised.nii,MNI-Poly-AMU_GM_reg.nii.gz,1,32] ", \
          "--convergence 20x15 --shrink-factors 2x1 --smoothing-sigmas 0mm --Restrict-Deformation 1x1x0 ", \
          "--output [regSeg,regSeg_1.nii.gz]"

    return


if __name__ == "__main__":
    # initialize parameters
    param = param()
    # call main function
    main()


