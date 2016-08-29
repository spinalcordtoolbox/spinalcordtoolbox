#!/usr/bin/env python
########################################################################################################################
#
#
# Gray matter segmentation - new implementation
#
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Sara Dupont
# Modified: 2016-06-14
#
# About the license: see the file LICENSE.TXT
########################################################################################################################

## TODO: use chunks of this code to register image to segment to model space

'''
## REGISTRATION WITH IMAGES FIRST, THEN APPLY TO GM/WM SEG
def coregister_data(self):
    im_mean_image = Image(param=self.mean_image)
    for dic_slice in self.slices:
        # create a directory to get the warping fields
        warp_dir = 'warping_fields_slice'+str(dic_slice.id)
        os.mkdir(warp_dir)
        # get dic image
        im_dic_slice = Image(param=dic_slice.im)
        # register image
        im_reg = register_data(im_src=im_dic_slice, im_dest=im_mean_image, param_reg=self.data_param.register_param, path_warp=warp_dir)
        # use warping fields to register gm seg
        list_gmseg_reg = []
        for gm_seg in dic_slice.gm_seg:
            im_gmseg = Image(param=gm_seg)
            im_gmseg_reg = apply_transfo(im_src=im_gmseg, im_dest=im_mean_image, warp=warp_dir+'/warp_src2dest.nii.gz')
            list_gmseg_reg.append(im_gmseg_reg.data)
        # set slie attributes with data registered into the model space
        dic_slice.set(im_m=im_reg.data)
        dic_slice.set(gm_seg_m=list_gmseg_reg)
        # remove warping fields directory
        shutil.rmtree(warp_dir)
'''
