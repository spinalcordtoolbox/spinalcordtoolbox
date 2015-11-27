#!/usr/bin/env python
# ==========================================================================================
#
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener
#
# License: see the LICENSE.TXT
# ==========================================================================================

import sys
from msct_image import Image
from msct_parser import Parser
import sct_utils as sct

class Param:
    def __init__(self):
        self.debug = 0
        self.remove_tmp_files = 1
        self.verbose = 1

        self.path_output = './'
        self.size_crop = 80


def main(fname_in, fname_seg, path_output_im, path_output_seg, size_crop, remove_tmp_files, verbose):
    off_x, off_y = 10, 10
    till_x, till_y = 10, 10

    for x in range(-off_x, off_x+1, till_x):
        for y in range(-off_y, off_y+1, till_y):
            offset = str(x) + ',' + str(y)
            print fname_in, offset
            # For each couple of image, extract square of each slice around the spinal cord centerline
            crop_image_around_segmentation(fname_in, fname_seg, path_output_im, path_output_seg, size_crop, offset, remove_tmp_files, verbose)


def crop_image_around_segmentation(fname_in, fname_seg, path_output_im, path_output_seg, size_crop, offset, remove_tmp_files, verbose):
    # 1. Resample to 1mm^3 isotropic
    fname_in_resampled = sct.add_suffix(fname_in, 'r')
    sct.run('sct_resample -i ' + fname_in + ' -mm 1x1x1 -o ' + fname_in_resampled, verbose=verbose)
    fname_in = fname_in_resampled
    fname_seg_resample = sct.add_suffix(fname_seg, 'r')
    sct.run('sct_resample -i ' + fname_seg + ' -mm 1x1x1 -o ' + fname_seg_resample, verbose=verbose)
    fname_seg = fname_seg_resample

    # 2. Orient both input images to RPI for the sake of simplicity
    sct.run('sct_image -i ' + fname_in + ' -setorient RPI', verbose=verbose)
    fname_in = sct.add_suffix(fname_in, '_RPI')
    sct.run('sct_image -i ' + fname_seg + ' -setorient RPI', verbose=verbose)
    fname_seg = sct.add_suffix(fname_seg, '_RPI')

    # 3. Pad both images to avoid edge issues when cropping
    fname_in_pad = sct.add_suffix(fname_in, 'p')
    pad_image = str(int(int(size_crop)/2))
    sct.run('sct_image -i ' + fname_in + ' -pad ' + pad_image + ',' + pad_image + ',0 -o ' + fname_in_pad, verbose=verbose)
    fname_in = fname_in_pad
    fname_seg_pad = sct.add_suffix(fname_seg, 'p')
    sct.run('sct_image -i ' + fname_seg + ' -pad ' + pad_image + ',' + pad_image + ',0 -o ' + fname_seg_pad, verbose=verbose)
    fname_seg = fname_seg_pad

    # 4. Extract centerline from segmentation
    fname_centerline = sct.add_suffix(fname_seg, '_centerline')
    sct.run('sct_process_segmentation -i ' + fname_seg + ' -p centerline', verbose=verbose) # -o ' + fname_centerline)

    # 5. Create a square mask around the spinal cord centerline
    fname_mask_box = 'mask_box.nii.gz'
    sct.run('sct_create_mask -i ' + fname_in + ' -m centerline,' + fname_centerline + ' -s ' + str(size_crop) +
            ' -o ' + fname_mask_box + ' -f box -e 1 -k '+offset, verbose=verbose)

    # 6. Crop image around the spinal cord and create a stack of square images
    sct.printv('Cropping around mask and stacking slices...', verbose=verbose)
    im_mask_box = Image(fname_mask_box)
    im_input = Image(fname_in)
    im_input.crop_and_stack(im_mask_box, suffix='_stack', save=True)
    im_seg = Image(fname_seg)
    im_seg.crop_and_stack(im_mask_box, suffix='_stack', save=True)

    # 6.5 Change name of images
    fname_stack_image = sct.add_suffix(fname_in, '_stack')
    fname_stack_seg = sct.add_suffix(fname_seg, '_stack')
    import random
    output_im_filename = str(random.randint(1, 1000000000000))
    output_im_fname = output_im_filename + '_im.nii.gz'
    output_seg_fname = output_im_filename + '_seg.nii.gz'
    sct.run('mv ' + fname_stack_image + ' ' + output_im_fname, verbose=verbose)
    sct.run('mv ' + fname_stack_seg + ' ' + output_seg_fname, verbose=verbose)

    # 7. Split the two stack images and save each slice
    sct.run('sct_image -i ' + output_im_fname + ' -split z', verbose=verbose)
    sct.run('sct_image -i ' + output_seg_fname + ' -split z', verbose=verbose)

    # 8. Move all images to output folders
    path_fname, file_fname, ext_fname = sct.extract_fname(output_im_fname)
    sct.run('mv ' + file_fname + '_* ' + path_output_im, verbose=verbose)
    path_fname, file_fname, ext_fname = sct.extract_fname(output_seg_fname)
    sct.run('mv ' + file_fname + '_* ' + path_output_seg, verbose=verbose)


def get_parser():
    """
    :return: Returns the parser with the command line documentation contained in it.
    """
    param = Param()

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('''This program reads two images, one with structural information and one with manual segmentation of the spinal cord, and extract all sub-regions of spinal cord per slice.''')
    parser.add_option(name='-i',
                      type_value='image_nifti',
                      description='Image to crop.',
                      mandatory=True,
                      example='t2.nii.gz')
    parser.add_option(name='-seg',
                      type_value='image_nifti',
                      description='Segmentation of the spinal cord',
                      mandatory=True,
                      example='t2_seg.nii.gz')
    parser.usage.addSection('General options')
    parser.add_option(name='-ofolder-im',
                      type_value='folder_creation',
                      description='Output folder for images',
                      mandatory=False,
                      example='./',
                      default_value=param.path_output)
    parser.add_option(name='-ofolder-seg',
                      type_value='folder_creation',
                      description='Output folder for segmentation',
                      mandatory=False,
                      example='./',
                      default_value=param.path_output)
    parser.add_option(name='-size',
                      type_value='int',
                      description='Size (in pixel) of the square image to crop per slice.',
                      mandatory=False,
                      default_value=str(param.size_crop),
                      example='80')
    parser.add_option(name='-r',
                      type_value='multiple_choice',
                      description='Removes the temporary folder and debug folder used for the algorithm at the end of execution.',
                      mandatory=False,
                      default_value=param.remove_tmp_files,
                      example=['0', '1'])
    parser.add_option(name='-v',
                      type_value='multiple_choice',
                      description='1: display on, 0: display off (default).',
                      mandatory=False,
                      example=['0', '1', '2'],
                      default_value=str(param.verbose))
    parser.add_option(name='-h',
                      type_value=None,
                      description='Display this help.',
                      mandatory=False)

    return parser

if __name__ == '__main__':
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])

    fname_in = arguments['-i']
    fname_seg = arguments['-seg']
    path_output_im = arguments['-ofolder-im']
    path_output_seg = arguments['-ofolder-seg']
    size_crop = arguments['-size']
    remove_tmp_files = arguments['-r']
    verbose = arguments['-v']

    main(fname_in, fname_seg, path_output_im, path_output_seg, size_crop, remove_tmp_files, verbose)
