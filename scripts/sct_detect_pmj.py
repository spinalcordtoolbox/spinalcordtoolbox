#!/usr/bin/env python

# Detect Ponto-Medullary Junction
#
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Charley
# Created: 2017-07-21
#
# About the license: see the file LICENSE.TXT

import os
import shutil
import sys
import numpy as np

from msct_image import Image
from msct_parser import Parser
# from sct_image import set_orientation, get_orientation
# from sct_utils import (add_suffix, extract_fname, printv, run,
#                        slash_at_the_end, Timer, tmp_create)


def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Detection of the Ponto-Medullary Junction (PMJ).\n'
                                 ' This method is machine-learning based and adapted for T1w-like or T2w-like images.\n'
                                 ' If the PMJ is detected from the input image, a nifti mask with one voxel, with the value 50,\n'
                                 ' located at the predicted PMJ level, is output ("*_pmj.nii.gz").\n'
                                 ' If the PMJ is not detected, anything is output from this function.')
    parser.add_option(name="-i",
                      type_value="file",
                      description="input image.",
                      mandatory=True,
                      example="t2.nii.gz")
    parser.add_option(name="-c",
                      type_value="multiple_choice",
                      description="type of image contrast, if your contrast is not in the available options (t1, t2), use t1 (cord bright / CSF dark) or t2 (cord dark / CSF bright)",
                      mandatory=True,
                      example=["t1", "t2"])
    parser.add_option(name="-s",
                      type_value="file",
                      description="SC segmentation or SC centerline mask. To provide this mask could help the detection of the PMJ",
                      mandatory=False,
                      example="t2_seg.nii.gz")
    parser.add_option(name="-ofolder",
                      type_value="folder_creation",
                      description="Output folder",
                      mandatory=False,
                      example="My_Output_Folder/")
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="Remove temporary files.",
                      mandatory=False,
                      default_value="1",
                      example=["0", "1"])
    parser.add_option(name="-v",
                      type_value='multiple_choice',
                      description="Verbose: 0 = nothing, 1 = classic, 2 = expended",
                      mandatory=False,
                      example=["0", "1", "2"],
                      default_value="1")

    return parser


# class ExtractGLCM:
#     def __init__(self, param=None, param_glcm=None):
#         self.param = param if param is not None else Param()
#         self.param_glcm = param_glcm if param_glcm is not None else ParamGLCM()

#         # create tmp directory
#         self.tmp_dir = tmp_create(verbose=self.param.verbose)  # path to tmp directory

#         if self.param.dim == 'ax':
#             self.orientation_extraction = 'RPI'
#         elif self.param.dim == 'sag':
#             self.orientation_extraction = 'IPR'
#         else:
#             self.orientation_extraction = 'IRP'

#         # metric_lst=['property_distance_angle']
#         self.metric_lst = []
#         for m in list(itertools.product(self.param_glcm.feature.split(','), self.param_glcm.angle.split(','))):
#             text_name = m[0] if m[0].upper() != 'asm'.upper() else m[0].upper()
#             self.metric_lst.append(text_name + '_' + str(self.param_glcm.distance) + '_' + str(m[1]))

#         # dct_im_seg{'im': list_of_axial_slice, 'seg': list_of_axial_masked_slice}
#         self.dct_im_seg = {'im': None, 'seg': None}

#         # to re-orient the data at the end if needed
#         self.orientation_im = get_orientation(Image(self.param.fname_im))

#         self.fname_metric_lst = {}

#     def extract(self):
#         self.ifolder2tmp()

#         # fill self.dct_metric --> for each key_metric: create an Image with zero values
#         self.init_metric_im()

#         # fill self.dct_im_seg --> extract axial slices from self.param.fname_im and self.param.fname_seg
#         self.extract_slices()

#         # compute texture
#         self.compute_texture()

#         # reorient data
#         if self.orientation_im != self.orientation_extraction:
#             self.reorient_data()

#         # mean across angles
#         self.mean_angle()

#         # save results to ofolder
#         self.tmp2ofolder()

#         return [self.param.path_results + self.fname_metric_lst[f] for f in self.fname_metric_lst]

#     def tmp2ofolder(self):

#         os.chdir('..')  # go back to original directory

#         printv('\nSave resulting files...', self.param.verbose, 'normal')
#         for f in self.fname_metric_lst:  # Copy from tmp folder to ofolder
#             shutil.copy(self.tmp_dir + self.fname_metric_lst[f],
#                         self.param.path_results + self.fname_metric_lst[f])

#     def ifolder2tmp(self):
#         # copy input image
#         if self.param.fname_im is not None:
#             shutil.copy(self.param.fname_im, self.tmp_dir)
#             self.param.fname_im = ''.join(extract_fname(self.param.fname_im)[1:])
#         else:
#             printv('ERROR: No input image', self.param.verbose, 'error')

#         # copy masked image
#         if self.param.fname_seg is not None:
#             shutil.copy(self.param.fname_seg, self.tmp_dir)
#             self.param.fname_seg = ''.join(extract_fname(self.param.fname_seg)[1:])
#         else:
#             printv('ERROR: No mask image', self.param.verbose, 'error')

#         os.chdir(self.tmp_dir)  # go to tmp directory

#     def mean_angle(self):

#         im_metric_lst = [self.fname_metric_lst[f].split('_' + str(self.param_glcm.distance) + '_')[0] + '_' for f in self.fname_metric_lst]
#         im_metric_lst = list(set(im_metric_lst))

#         printv('\nMean across angles...', self.param.verbose, 'normal')
#         extension = extract_fname(self.param.fname_im)[2]
#         for im_m in im_metric_lst:     # Loop across GLCM texture properties
#             # List images to mean
#             im2mean_lst = [im_m + str(self.param_glcm.distance) + '_' + a + extension for a in self.param_glcm.angle.split(',')]

#             # Average across angles and save it as wrk_folder/fnameIn_feature_distance_mean.extension
#             fname_out = im_m + str(self.param_glcm.distance) + '_mean' + extension
#             run('sct_image -i ' + ','.join(im2mean_lst) + ' -concat t -o ' + fname_out, error_exit='warning', raise_exception=True)
#             run('sct_maths -i ' + fname_out + ' -mean t -o ' + fname_out, error_exit='warning', raise_exception=True)
#             self.fname_metric_lst[im_m + str(self.param_glcm.distance) + '_mean'] = fname_out

#     def extract_slices(self):
#         # open image and re-orient it to RPI if needed
#         im, seg = Image(self.param.fname_im), Image(self.param.fname_seg)
#         if self.orientation_im != self.orientation_extraction:
#             im, seg = set_orientation(im, self.orientation_extraction), set_orientation(seg, self.orientation_extraction)

#         # extract axial slices in self.dct_im_seg
#         self.dct_im_seg['im'], self.dct_im_seg['seg'] = [im.data[:, :, z] for z in range(im.dim[2])], [seg.data[:, :, z] for z in range(im.dim[2])]

#     def init_metric_im(self):
#         # open image and re-orient it to RPI if needed
#         im_tmp = Image(self.param.fname_im)
#         if self.orientation_im != self.orientation_extraction:
#             im_tmp = set_orientation(im_tmp, self.orientation_extraction)

#         # create Image objects with zeros values for each output image needed
#         for m in self.metric_lst:
#             im_2save = im_tmp.copy()
#             im_2save.changeType(type='float64')
#             im_2save.data *= 0
#             fname_out = add_suffix(''.join(extract_fname(self.param.fname_im)[1:]), '_' + m)
#             im_2save.setFileName(fname_out)
#             im_2save.save()
#             self.fname_metric_lst[m] = fname_out

#     def compute_texture(self):

#         offset = int(self.param_glcm.distance)
#         printv('\nCompute texture metrics...', self.param.verbose, 'normal')

#         dct_metric = {}
#         for m in self.metric_lst:
#             dct_metric[m] = Image(self.fname_metric_lst[m])

#         timer = Timer(number_of_iteration=len(self.dct_im_seg['im']))
#         timer.start()

#         for im_z, seg_z, zz in zip(self.dct_im_seg['im'], self.dct_im_seg['seg'], range(len(self.dct_im_seg['im']))):
#             for xx in range(im_z.shape[0]):
#                 for yy in range(im_z.shape[1]):
#                     if not seg_z[xx, yy]:
#                         continue
#                     if xx < offset or yy < offset:
#                         continue
#                     if xx > (im_z.shape[0] - offset - 1) or yy > (im_z.shape[1] - offset - 1):
#                         continue  # to check if the whole glcm_window is in the axial_slice
#                     if False in np.unique(seg_z[xx - offset: xx + offset + 1, yy - offset: yy + offset + 1]):
#                         continue  # to check if the whole glcm_window is in the mask of the axial_slice

#                     glcm_window = im_z[xx - offset: xx + offset + 1, yy - offset: yy + offset + 1]
#                     glcm_window = glcm_window.astype(np.uint8)

#                     dct_glcm = {}
#                     for a in self.param_glcm.angle.split(','):  # compute the GLCM for self.param_glcm.distance and for each self.param_glcm.angle
#                         dct_glcm[a] = greycomatrix(glcm_window,
#                                                    [self.param_glcm.distance], [radians(int(a))],
#                                                    symmetric=self.param_glcm.symmetric,
#                                                    normed=self.param_glcm.normed)

#                     for m in self.metric_lst:  # compute the GLCM property (m.split('_')[0]) of the voxel xx,yy,zz
#                         dct_metric[m].data[xx, yy, zz] = greycoprops(dct_glcm[m.split('_')[2]], m.split('_')[0])[0][0]

#             timer.add_iteration()

#         timer.stop()

#         for m in self.metric_lst:
#             dct_metric[m].setFileName(self.fname_metric_lst[m])
#             dct_metric[m].save()

#     def reorient_data(self):
#         for f in self.fname_metric_lst:
#             im = Image(self.fname_metric_lst[f])
#             im = set_orientation(im, self.orientation_im)
#             im.setFileName(self.fname_metric_lst[f])
#             im.save()

def main(args=None):
    if args is None:
        args = sys.argv[1:]

    # get parser
    parser = get_parser()
    arguments = parser.parse(args)

    # set param arguments ad inputted by user
    fname_im = arguments["-i"]
    contrast = arguments["-c"]

    if '-s' in arguments:
        fname_seg = arguments['-s']
    else:
        fname_seg = None
    if not os.path.isfile(fname_seg):
        fname_seg = None
        printv('WARNING: -s input file: "' + arguments['-s'] + '" does not exist.\nDetecting PMJ without using segmentation information', 1, 'warning')

    if '-ofolder' in arguments:
        path_results = slash_at_the_end(arguments["-ofolder"], slash=1)
    else:
        path_results = None
    if not os.path.isdir(path_results) and os.path.exists(path_results):
        printv("ERROR output directory %s is not a valid directory" % path_results, 1, 'error')
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    if '-r' in arguments:
        rm_tmp = bool(int(arguments['-r']))
    else:
        rm_tmp = None
    if '-v' in arguments:
        verbose = bool(int(arguments['-v']))
    else:
        verbose = None

    # # Initialize DetectPMJ
    # detector = DetectPMJ(fname_im=fname_im, contrast=contrast, fname_seg=fname_seg, path_out=path_results, rm_tmp=rm_tmp, verbose=verbose)
    # # run the extraction
    # fname_out = detector.apply()

    # # remove tmp_dir
    # if rm_tmp:
    #     shutil.rmtree(tmp_dir)

    # printv('\nDone! To view results, type:', verbose)
    # printv('fslview ' + arguments["-i"] + ' ' + fname_out + ' -l Red -t 0.7 & \n', verbose, 'info')

    # """
    #   - reflechir si path_results doit etre mis a None si no isdir
    #   - output a png with red dot : cf GM seg
    # """


if __name__ == "__main__":
    main()