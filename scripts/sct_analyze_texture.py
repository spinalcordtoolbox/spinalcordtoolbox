#!/usr/bin/env python

# Analyse texture
#
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Charley
# Modified: 2017-06-22
#
# About the license: see the file LICENSE.TXT

from __future__ import absolute_import

import os
import shutil
import sys
import numpy as np
import itertools
import argparse

from skimage.feature import greycomatrix, greycoprops

import sct_utils as sct
import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils import Metavar, SmartFormatter, ActionCreateFolder, sct_progress_bar

def get_parser():
    # Initialize the parser

    parser = argparse.ArgumentParser(
        description='Extraction of grey level co-occurence matrix (GLCM) texture features from an image within a given '
                    'mask. The textures features are those defined in the sckit-image implementation: '
                    'http://scikit-image.org/docs/dev/api/skimage.feature.html#greycoprops. This function outputs '
                    'one nifti file per texture metric (' + ParamGLCM().feature + ') and per orientation called '
                    'fnameInput_feature_distance_angle.nii.gz. Also, a file averaging each metric across the angles, '
                    'called fnameInput_feature_distance_mean.nii.gz, is output.',
        add_help=None,
        formatter_class=SmartFormatter,
        prog=os.path.basename(__file__).strip(".py")
    )

    mandatoryArguments = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatoryArguments.add_argument(
        "-i",
        required=True,
        help='Image to analyze. Example: t2.nii.gz',
        metavar=Metavar.file,
    )
    mandatoryArguments.add_argument(
        "-m",
        required=True,
        metavar=Metavar.file,
        help='Image mask Example: t2_seg.nii.gz',
    )

    optional = parser.add_argument_group("\nOPTIONALS ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")
    optional.add_argument(
        "-feature",
        metavar=Metavar.str,
        help='List of GLCM texture features (separate arguments with ",").',
        required=False,
        default=ParamGLCM().feature)
    optional.add_argument(
        "-distance",
        metavar=Metavar.int,
        help='Distance offset for GLCM computation, in pixel (suggested distance values between 1 and 5). Example: 1',
        required=False,
        default=ParamGLCM().distance)
    optional.add_argument(
        "-angle",
        metavar=Metavar.list,
        help='List of angles for GLCM computation, separate arguments with ",", in degrees (suggested distance values '
             'between 0 and 179). Example: 0,90',
        required=False,
        default=ParamGLCM().angle)
    optional.add_argument(
        "-dim",
        help="Compute the texture on the axial (ax), sagittal (sag) or coronal (cor) slices.",
        required=False,
        choices=('ax', 'sag', 'cor'),
        default=Param().dim)
    optional.add_argument(
        "-ofolder",
        metavar=Metavar.folder,
        help='Output folder. Example: /my_texture/',
        action=ActionCreateFolder,
        required=False,
        default=Param().path_results)
    optional.add_argument(
        "-igt",
        metavar=Metavar.str,
        help="File name of ground-truth texture metrics.",
        required=False)
    optional.add_argument(
        "-r",
        help="Remove temporary files.",
        required=False,
        type=int,
        choices=(0, 1),
        default=int(Param().rm_tmp))
    optional.add_argument(
        "-v",
        help="Verbose: 0 = nothing, 1 = classic, 2 = expended.",
        required=False,
        type=int,
        choices=(0, 1, 2),
        default=Param().verbose)

    return parser


class ExtractGLCM:
    def __init__(self, param=None, param_glcm=None):
        self.param = param if param is not None else Param()
        self.param_glcm = param_glcm if param_glcm is not None else ParamGLCM()

        # create tmp directory
        self.tmp_dir = sct.tmp_create(verbose=self.param.verbose)  # path to tmp directory

        if self.param.dim == 'ax':
            self.orientation_extraction = 'RPI'
        elif self.param.dim == 'sag':
            self.orientation_extraction = 'IPR'
        else:
            self.orientation_extraction = 'IRP'

        # metric_lst=['property_distance_angle']
        self.metric_lst = []
        for m in list(itertools.product(self.param_glcm.feature.split(','), self.param_glcm.angle.split(','))):
            text_name = m[0] if m[0].upper() != 'asm'.upper() else m[0].upper()
            self.metric_lst.append(text_name + '_' + str(self.param_glcm.distance) + '_' + str(m[1]))

        # dct_im_seg{'im': list_of_axial_slice, 'seg': list_of_axial_masked_slice}
        self.dct_im_seg = {'im': None, 'seg': None}

        # to re-orient the data at the end if needed
        self.orientation_im = Image(self.param.fname_im).orientation

        self.fname_metric_lst = {}

    def extract(self):
        self.ifolder2tmp()

        # fill self.dct_metric --> for each key_metric: create an Image with zero values
        # self.init_metric_im()

        # fill self.dct_im_seg --> extract axial slices from self.param.fname_im and self.param.fname_seg
        self.extract_slices()

        # compute texture
        self.compute_texture()

        # reorient data
        if self.orientation_im != self.orientation_extraction:
            self.reorient_data()

        # mean across angles
        self.mean_angle()

        # save results to ofolder
        self.tmp2ofolder()

        return [os.path.join(self.param.path_results, self.fname_metric_lst[f]) for f in self.fname_metric_lst]

    def tmp2ofolder(self):

        os.chdir(self.curdir)  # go back to original directory

        sct.printv('\nSave resulting files...', self.param.verbose, 'normal')
        for f in self.fname_metric_lst:  # Copy from tmp folder to ofolder
            sct.copy(os.path.join(self.tmp_dir, self.fname_metric_lst[f]),
                        os.path.join(self.param.path_results, self.fname_metric_lst[f]))

    def ifolder2tmp(self):
        self.curdir = os.getcwd()
        # copy input image
        if self.param.fname_im is not None:
            sct.copy(self.param.fname_im, self.tmp_dir)
            self.param.fname_im = ''.join(sct.extract_fname(self.param.fname_im)[1:])
        else:
            sct.printv('ERROR: No input image', self.param.verbose, 'error')

        # copy masked image
        if self.param.fname_seg is not None:
            sct.copy(self.param.fname_seg, self.tmp_dir)
            self.param.fname_seg = ''.join(sct.extract_fname(self.param.fname_seg)[1:])
        else:
            sct.printv('ERROR: No mask image', self.param.verbose, 'error')

        os.chdir(self.tmp_dir)  # go to tmp directory

    def mean_angle(self):

        im_metric_lst = [self.fname_metric_lst[f].split('_' + str(self.param_glcm.distance) + '_')[0] + '_' for f in self.fname_metric_lst]
        im_metric_lst = list(set(im_metric_lst))

        sct.printv('\nMean across angles...', self.param.verbose, 'normal')
        extension = sct.extract_fname(self.param.fname_im)[2]
        for im_m in im_metric_lst:     # Loop across GLCM texture properties
            # List images to mean
            im2mean_lst = [im_m + str(self.param_glcm.distance) + '_' + a + extension for a in self.param_glcm.angle.split(',')]

            # Average across angles and save it as wrk_folder/fnameIn_feature_distance_mean.extension
            fname_out = im_m + str(self.param_glcm.distance) + '_mean' + extension
            sct.run('sct_image -i ' + ' '.join(im2mean_lst) + ' -concat t -o ' + fname_out)
            sct.run('sct_maths -i ' + fname_out + ' -mean t -o ' + fname_out)
            self.fname_metric_lst[im_m + str(self.param_glcm.distance) + '_mean'] = fname_out

    def extract_slices(self):
        # open image and re-orient it to RPI if needed
        im, seg = Image(self.param.fname_im), Image(self.param.fname_seg)
        if self.orientation_im != self.orientation_extraction:
            im.change_orientation(self.orientation_extraction)
            seg.change_orientation(self.orientation_extraction)

        # extract axial slices in self.dct_im_seg
        self.dct_im_seg['im'], self.dct_im_seg['seg'] = [im.data[:, :, z] for z in range(im.dim[2])], [seg.data[:, :, z] for z in range(im.dim[2])]

    # def init_metric_im(self):
    #     # open image and re-orient it to RPI if needed
    #     im_tmp = Image(self.param.fname_im)
    #     if self.orientation_im != self.orientation_extraction:
    #         im_tmp = msct_image.change_orientation(im_tmp, self.orientation_extraction)

    #     # create Image objects with zeros values for each output image needed
    #     for m in self.metric_lst:
    #         im_2save = msct_image.zeros_like(im_tmp, dtype=np.float64)
    #         fname_out = sct.add_suffix(''.join(sct.extract_fname(self.param.fname_im)[1:]), '_' + m)
    #         im_2save.save(fname_out)
    #         self.fname_metric_lst[m] = fname_out

    def compute_texture(self):

        offset = int(self.param_glcm.distance)
        sct.printv('\nCompute texture metrics...', self.param.verbose, 'normal')

        # open image and re-orient it to RPI if needed
        im_tmp = Image(self.param.fname_im)
        if self.orientation_im != self.orientation_extraction:
            im_tmp.change_orientation(self.orientation_extraction)

        dct_metric = {}
        for m in self.metric_lst:
            im_2save = msct_image.zeros_like(im_tmp, dtype='float64')
            dct_metric[m] = im_2save
            # dct_metric[m] = Image(self.fname_metric_lst[m])

        with sct_progress_bar() as pbar:
            for im_z, seg_z, zz in zip(self.dct_im_seg['im'], self.dct_im_seg['seg'], range(len(self.dct_im_seg['im']))):
                for xx in range(im_z.shape[0]):
                    for yy in range(im_z.shape[1]):
                        if not seg_z[xx, yy]:
                            continue
                        if xx < offset or yy < offset:
                            continue
                        if xx > (im_z.shape[0] - offset - 1) or yy > (im_z.shape[1] - offset - 1):
                            continue  # to check if the whole glcm_window is in the axial_slice
                        if False in np.unique(seg_z[xx - offset: xx + offset + 1, yy - offset: yy + offset + 1]):
                            continue  # to check if the whole glcm_window is in the mask of the axial_slice

                        glcm_window = im_z[xx - offset: xx + offset + 1, yy - offset: yy + offset + 1]
                        glcm_window = glcm_window.astype(np.uint8)

                        dct_glcm = {}
                        for a in self.param_glcm.angle.split(','):  # compute the GLCM for self.param_glcm.distance and for each self.param_glcm.angle
                            dct_glcm[a] = greycomatrix(glcm_window,
                                                       [self.param_glcm.distance], [np.radians(int(a))],
                                                       symmetric=self.param_glcm.symmetric,
                                                       normed=self.param_glcm.normed)

                        for m in self.metric_lst:  # compute the GLCM property (m.split('_')[0]) of the voxel xx,yy,zz
                            dct_metric[m].data[xx, yy, zz] = greycoprops(dct_glcm[m.split('_')[2]], m.split('_')[0])[0][0]

                        pbar.set_postfix(pos="{}/{}".format(zz, len(self.dct_im_seg["im"])))
                        pbar.update(1)

        for m in self.metric_lst:
            fname_out = sct.add_suffix("".join(sct.extract_fname(self.param.fname_im)[1:]), '_' + m)
            dct_metric[m].save(fname_out)
            self.fname_metric_lst[m] = fname_out

    def reorient_data(self):
        for f in self.fname_metric_lst:
            os.rename(self.fname_metric_lst[f], sct.add_suffix("".join(sct.extract_fname(self.param.fname_im)[1:]), '_2reorient'))
            im = Image(sct.add_suffix("".join(sct.extract_fname(self.param.fname_im)[1:]), '_2reorient')) \
             .change_orientation(self.orientation_im) \
             .save(self.fname_metric_lst[f])


class Param:
    def __init__(self):
        self.fname_im = None
        self.fname_seg = None
        self.path_results = './texture/'
        self.verbose = 1
        self.dim = 'ax'
        self.rm_tmp = True


class ParamGLCM(object):
    def __init__(self, symmetric=True, normed=True, feature='contrast,dissimilarity,homogeneity,energy,correlation,ASM', distance=1, angle='0,45,90,135'):
        self.symmetric = True  # If True, the output matrix P[:, :, d, theta] is symmetric.
        self.normed = True  # If True, normalize each matrix P[:, :, d, theta] by dividing by the total number of accumulated co-occurrences for the given offset. The elements of the resulting matrix sum to 1.
        self.feature = 'contrast,dissimilarity,homogeneity,energy,correlation,ASM'  # The property formulae are detailed here: http://scikit-image.org/docs/dev/api/skimage.feature.html#greycoprops
        self.distance = 1  # Size of the window: distance = 1 --> a reference pixel and its immediate neighbor
        self.angle = '0,45,90,135'  # Rotation angles for co-occurrence matrix


def main(args=None):
    """
    Main function
    :param args:
    :return:
    """
    # get parser args
    if args is None:
        args = None if sys.argv[1:] else ['--help']
    parser = get_parser()
    arguments = parser.parse_args(args=args)

    # create param object
    param = Param()
    param_glcm = ParamGLCM()

    # set param arguments ad inputted by user
    param.fname_im = arguments.i
    param.fname_seg = arguments.m

    if arguments.ofolder is not None:
        param.path_results = arguments.ofolder

    if not os.path.isdir(param.path_results) and os.path.exists(param.path_results):
        sct.printv("ERROR output directory %s is not a valid directory" % param.path_results, 1, 'error')
    if not os.path.exists(param.path_results):
        os.makedirs(param.path_results)

    if arguments.feature is not None:
        param_glcm.feature = arguments.feature
    if arguments.distance is not None:
        param_glcm.distance = arguments.distance
    if arguments.angle is not None:
        param_glcm.angle = arguments.angle

    if arguments.dim is not None:
        param.dim = arguments.dim
    if arguments.r is not None:
        param.rm_tmp = bool(arguments.r)
    verbose = arguments.v
    sct.init_sct(log_level=verbose, update=True)  # Update log level

    # create the GLCM constructor
    glcm = ExtractGLCM(param=param, param_glcm=param_glcm)
    # run the extraction
    fname_out_lst = glcm.extract()

    # remove tmp_dir
    if param.rm_tmp:
        sct.rmtree(glcm.tmp_dir)

    sct.printv('\nDone! To view results, type:', param.verbose)
    sct.printv('fsleyes ' + arguments.i + ' ' + ' -cm red-yellow -a 70.0 '.join(fname_out_lst) + ' -cm Red-Yellow -a 70.0 & \n', param.verbose, 'info')

if __name__ == "__main__":
    sct.init_sct()
    main()
