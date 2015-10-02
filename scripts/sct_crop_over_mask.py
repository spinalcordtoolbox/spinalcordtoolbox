#!/usr/bin/env python
#########################################################################################
#
# Asman et al. groupwise multi-atlas segmentation method implementation
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Augustin Roux, Sara Dupont
# Modified: 2014-11-20
#
# About the license: see the file LICENSE.TXT
#########################################################################################
from msct_parser import *
import sys
from msct_image import Image


class Param:
    def __init__(self):
        self.debug = 0


def main():
    fname_out = ''
    square = 0
    parser = Parser(__file__)
    parser.usage.set_description('Crop the image depending the mask \n'
                                 'The mask and the image must have the same dimensions\n')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Image you want to crop",
                      mandatory=True,
                      example='t2.nii.gz')
    parser.add_option(name="-mask",
                      type_value="file",
                      description="Mask you want to crop with",
                      mandatory=True,
                      example='t2_mask.nii.gz')
    parser.add_option(name="-o",
                      type_value="str",
                      description="Output file name",
                      mandatory=False,
                      example='t2_croped.nii.gz')
    parser.add_option(name="-square",
                      type_value="int",
                      description="Crop from a square mask (the image dimension will be changed), default is 0\n"
                                  "WARNING : if this argument is used,the returned fil is oriented in IRP",
                      mandatory=False,
                      example=1)

    # Getting the arguments
    arguments = parser.parse(sys.argv[1:])
    fname_input = arguments["-i"]
    fname_mask = arguments["-mask"]
    if "-o" in arguments:
        fname_out = arguments["-o"]
    if "-square" in arguments:
        square = arguments["-square"]

    input_img = Image(fname_input)
    if len(input_img.data.shape) == 3:
        if input_img.data.shape[2] == 1:
            input_img.data = input_img.data.reshape(input_img.data.shape[:-1])

    mask = Image(fname_mask)
    if square:
        if len(input_img.data.shape) == 3:
            orientation_init = input_img.orientation

            if orientation_init != 'IRP':
                status, output = sct.run('sct_orientation -i ' + fname_input + ' -s IRP ')
                fname_input = sct.extract_fname(fname_input)[0] + sct.extract_fname(fname_input)[1] + '_IRP.nii.gz'
                input_img = Image(fname_input)

            if mask.orientation != 'IRP':
                # elif sct.run('sct_orientation -i ' + fname_mask)[1][4:7] != 'IRP':
                status, output = sct.run('sct_orientation -i ' + mask.absolutepath + ' -s IRP ')
                mask.file_name += '_IRP'
                mask = Image(mask.path+mask.file_name+mask.ext)


    # mask = Image(fname_mask)
    if len(mask.data.shape) == 3:
        if mask.data.shape[2] == 1:
            mask.data = mask.data.reshape(mask.data.shape[:-1])

    if square:
        input_img.crop_and_straighten(mask)
    else:
        input_img.crop_from_mask(mask)

    input_img.path = './'

    if fname_out == '':
        fname_out = input_img.file_name + '_smartly_cropped_over_mask'


    if fname_out[-7:] == '.nii.gz':
        input_img.file_name = fname_out[:-7]
    else:
        input_img.file_name = fname_out

    print 'shape output',input_img.data.shape
    input_img.save()


    '''
    if square and orientation_init != 'IRP':
        status, output = sct.run('sct_orientation -i ' + fname_out + '.nii.gz -s ' + orientation_init + ' ')
    '''


if __name__ == "__main__":
    # call main function
    main()