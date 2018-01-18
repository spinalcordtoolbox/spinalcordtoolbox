#!/usr/bin/env python
#######################################################################################################################
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
'''
INFORMATION:
The model used in this function is compound of:
  - a dictionary: a list of slices of WM/GM contrasted images with their manual segmentations [slices.pklz]
  - a model representing this dictionary in a reduced space (a PCA or an isomap model as implemented in sk-learn) [fitted_model.pklz]
  - the dictionary data fitted to this model (i.e. in the model space) [fitted_data.pklz]
  - the averaged median intensity in the white and gray matter in the model [intensities.pklz]
  - an information file indicating which parameters were used to construct this model, and te date of computation [info.txt]

A constructed model is provided in the toolbox here: $PATH_SCT/data/gm_model.
It's made from T2* images of 80 subjects and computed with the parameters that gives the best gray matter segmentation results.
However you can compute you own model with your own data or with other parameters and use it to segment gray matter by using  the flag -model path_new_gm_model/.

To do so, you should have a folder (path_to_dataset/) containing for each subject (with a folder per subject):
        - a WM/GM contrasted image (for ex T2*-w) containing 'im' in its name
        - a segmentation of the spinal cord containing 'seg' in its name
        - a (or several) manual segmentation(s) of the gray matter containing 'gm' in its(their) name(s)
        - a level file containing 'level' in its name : it can be an image containing a level label per slice indicating at wich vertebral level correspond this slice (usually obtained by registering the PAM50 template to the WM/GM contrasted image) or a text file indicating the level of each slice.

For more information on the parameters available to compute the model, type:
msct_multiatlas_seg -h

to compute the model, use the following command line :
msct_multiatlas_seg -path-data path_to_dataset/

Then use the folder gm_model/ (output from msct_multiatlas_seg) in this function the flag -model gm_model/

'''

import os
import shutil
import sys
import time
import copy
from math import exp

import numpy as np

import sct_maths
import sct_process_segmentation
import sct_register_multimodal
from msct_gmseg_utils import (apply_transfo, average_gm_wm, binarize,
                              normalize_slice, pre_processing, register_data)
from msct_image import Image
from msct_multiatlas_seg import Model, Param, ParamData, ParamModel
from msct_parser import Parser
from sct_image import set_orientation
import sct_utils as sct
from sct_utils import (add_suffix, extract_fname, printv, run, tmp_create)


def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Segmentation of the white and gray matter.'
                                 ' The segmentation is based on a multi-atlas method that uses a dictionary of pre-segmented gray matter images (already included in SCT) and finds the most similar images for identifying the gray matter using label fusion approach. The model used by this method contains: a template of the white/gray matter segmentation along the cervical spinal cord, and a PCA reduced space to describe the variability of intensity in that template.'
                                 ' This method was inspired from [Asman et al., Medical Image Analysis 2014] and features the following additions:\n'
                                 '- possibility to add information from vertebral levels for improved accuracy\n'
                                 '- intensity normalization of the image to segment (allows the segmentation of any kind of contrast)\n'
                                 '- pre-registration based on non-linear transformations')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Image to segment",
                      mandatory=True,
                      example='t2star.nii.gz')
    parser.add_option(name="-s",
                      type_value="file",
                      description="Spinal cord segmentation",
                      mandatory=True,
                      example='sc_seg.nii.gz')

    parser.usage.addSection('SEGMENTATION OPTIONS')

    parser.add_option(name="-vertfile",
                      type_value="str",
                      description='Labels of vertebral levels used as prior for the segmentation. This could either be an image (e.g., label/template/PAM50_levels.nii.gz) or a text file that specifies "slice,level" at each line. Example:\n'
                      "0,3\n"
                      "1,3\n"
                      "2,4\n"
                      "3,4\n"
                      "4,4\n",
                      mandatory=False,
                      default_value=ParamSeg().fname_level)
    parser.add_option(name="-vert",
                      mandatory=False,
                      deprecated_by='-vertfile')
    parser.add_option(name="-l",
                      mandatory=False,
                      deprecated_by='-vertfile')

    parser.add_option(name="-denoising",
                      type_value='multiple_choice',
                      description="1: Adaptative denoising from F. Coupe algorithm, 0: no  WARNING: It affects the model you should use (if denoising is applied to the target, the model should have been computed with denoising too)",
                      mandatory=False,
                      default_value=int(ParamData().denoising),
                      example=['0', '1'])
    parser.add_option(name="-normalization",
                      type_value='multiple_choice',
                      description="Normalization of the target image's intensity using median intensity values of the WM and the GM, recomended with MT images or other types of contrast than T2*",
                      mandatory=False,
                      default_value=int(ParamData().normalization),
                      example=['0', '1'])
    parser.add_option(name="-p",
                      type_value='str',
                      description="Registration parameters to register the image to segment on the model data. Use the same format as for sct_register_to_template and sct_register_multimodal.",
                      mandatory=False,
                      default_value=ParamData().register_param,
                      example='step=1,type=seg,algo=centermassrot,metric=MeanSquares,smooth=2,iter=1:step=2,type=seg,algo=columnwise,metric=MeanSquares,smooth=3,iter=1:step=3,type=seg,algo=bsplinesyn,metric=MeanSquares,iter=3')
    parser.add_option(name="-w-levels",
                      type_value='float',
                      description="Weight parameter on the level differences to compute the similarities",
                      mandatory=False,
                      default_value=ParamSeg().weight_level,
                      example=2.0)
    parser.add_option(name="-w-coordi",
                      type_value='float',
                      description="Weight parameter on the euclidean distance (based on images coordinates in the reduced sapce) to compute the similarities ",
                      mandatory=False,
                      default_value=ParamSeg().weight_coord,
                      example=0.005)
    parser.add_option(name="-thr-sim",
                      type_value='float',
                      description="Threshold to select the dictionary slices most similar to the slice to segment (similarities are normalized to 1)",
                      mandatory=False,
                      default_value=ParamSeg().thr_similarity,
                      example=0.6)
    parser.add_option(name="-model",
                      type_value="folder",
                      description="Path to the computed model",
                      mandatory=False,
                      example='/home/jdoe/gm_seg_model/')

    parser.usage.addSection('\nOUTPUT OPTIONS')

    parser.add_option(name="-res-type",
                      type_value='multiple_choice',
                      description="Type of result segmentation : binary or probabilistic",
                      mandatory=False,
                      default_value=ParamSeg().type_seg,
                      example=['bin', 'prob'])
    parser.add_option(name="-ratio",
                      type_value='multiple_choice',
                      description="Compute GM/WM CSA ratio by slice or by vertebral level (average across levels). For no computation of ratio, set to 0.",
                      mandatory=False,
                      default_value=ParamSeg().ratio,
                      example=['0', 'slice', 'level'])
    parser.add_option(name="-ofolder",
                      type_value="folder_creation",
                      description="Output folder",
                      mandatory=False,
                      default_value=ParamSeg().path_results,
                      example='gm_segmentation_results/')

    parser.usage.addSection('\nQC OPTIONS')

    parser.add_option(name="-ref",
                      type_value="file",
                      description="Compute DICE coefficient, Hausdorff's and median distances between output segmentation and gold-standard segmentation specified here",
                      mandatory=False,
                      example='manual_gm_seg.nii.gz')

    parser.usage.addSection('\nMISC')
    parser.add_option(name='-qc',
                      type_value='multiple_choice',
                      description='Output images for quality control.',
                      mandatory=False,
                      example=['0', '1'],
                      default_value=str(int(ParamSeg().qc)))
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description='Remove temporary files.',
                      mandatory=False,
                      default_value=str(int(Param().rm_tmp)),
                      example=['0', '1'])
    parser.add_option(name="-v",
                      type_value='multiple_choice',
                      description="Verbose: 0 = nothing, 1 = classic, 2 = expended",
                      mandatory=False,
                      example=['0', '1', '2'],
                      default_value=str(Param().verbose))

    return parser


class ParamSeg:
    def __init__(self):
        self.fname_im = None
        self.fname_im_original = None
        self.fname_seg = None
        self.fname_level = 'label/template/PAM50_levels.nii.gz'
        self.fname_manual_gmseg = None
        self.path_results = './'

        # param to compute similarities:
        self.weight_level = 2.5  # gamma
        self.weight_coord = 0.0065  # tau --> need to be validated for specific dataset
        self.thr_similarity = 0.0005  # epsilon but on normalized to 1 similarities (by slice of dic and slice of target)
        # TODO = find the best thr

        self.type_seg = 'prob'  # 'prob' or 'bin'
        self.thr_bin = 0.5
        self.ratio = '0'  # '0', 'slice' or 'level'

        self.qc = True


class SegmentGM:
    def __init__(self, param_seg=None, param_model=None, param_data=None, param=None):
        self.param_seg = param_seg if param_seg is not None else ParamSeg()
        self.param_model = param_model if param_model is not None else ParamModel()
        self.param_data = param_data if param_data is not None else ParamData()
        self.param = param if param is not None else Param()

        # create model:
        self.model = Model(param_model=self.param_model, param_data=self.param_data, param=self.param)

        # create tmp directory
        self.tmp_dir = tmp_create(verbose=self.param.verbose)  # path to tmp directory

        self.target_im = None  # list of slices
        self.info_preprocessing = None  # dic containing {'orientation': 'xxx', 'im_sc_seg_rpi': im, 'interpolated_images': [list of im = interpolated image data per slice]}

        self.projected_target = None  # list of coordinates of the target slices in the model reduced space
        self.im_res_gmseg = None
        self.im_res_wmseg = None

    def segment(self):
        self.copy_data_to_tmp()
        # go to tmp directory
        curdir = os.getcwd()
        os.chdir(self.tmp_dir)
        # load model
        self.model.load_model()

        self.target_im, self.info_preprocessing = pre_processing(self.param_seg.fname_im, self.param_seg.fname_seg, self.param_seg.fname_level, new_res=self.param_data.axial_res, square_size_size_mm=self.param_data.square_size_size_mm, denoising=self.param_data.denoising, verbose=self.param.verbose, rm_tmp=self.param.rm_tmp)

        printv('\nRegister target image to model data...', self.param.verbose, 'normal')
        # register target image to model dictionary space
        path_warp = self.register_target()

        if self.param_data.normalization:
            printv('\nNormalize intensity of target image...', self.param.verbose, 'normal')
            self.normalize_target()

        printv('\nProject target image into the model reduced space...', self.param.verbose, 'normal')
        self.project_target()

        printv('\nCompute similarities between target slices and model slices using model reduced space...', self.param.verbose, 'normal')
        list_dic_indexes_by_slice = self.compute_similarities()

        printv('\nLabel fusion of model slices most similar to target slices...', self.param.verbose, 'normal')
        self.label_fusion(list_dic_indexes_by_slice)

        printv('\nWarp back segmentation into image space...', self.param.verbose, 'normal')
        self.warp_back_seg(path_warp)

        printv('\nPost-processing...', self.param.verbose, 'normal')
        self.im_res_gmseg, self.im_res_wmseg = self.post_processing()

        if (self.param_seg.path_results != './') and (not os.path.exists(os.path.join(curdir, self.param_seg.path_results))):
            # create output folder
            printv('\nCreate output folder ...', self.param.verbose, 'normal')
            os.chdir(curdir)
            os.mkdir(self.param_seg.path_results)
            os.chdir(self.tmp_dir)

        if self.param_seg.fname_manual_gmseg is not None:
            # compute validation metrics
            printv('\nCompute validation metrics...', self.param.verbose, 'normal')
            self.validation()

        if self.param_seg.ratio is not '0':
            printv('\nCompute GM/WM CSA ratio...', self.param.verbose, 'normal')
            self.compute_ratio()

        # go back to original directory
        os.chdir(curdir)
        printv('\nSave resulting GM and WM segmentations...', self.param.verbose, 'normal')
        self.fname_res_gmseg = os.path.join(self.param_seg.path_results, add_suffix(''.join(extract_fname(self.param_seg.fname_im)[1:]), '_gmseg'))
        self.fname_res_wmseg = os.path.join(self.param_seg.path_results, add_suffix(''.join(extract_fname(self.param_seg.fname_im)[1:]), '_wmseg'))

        self.im_res_gmseg.setFileName(self.fname_res_gmseg)
        self.im_res_wmseg.setFileName(self.fname_res_wmseg)

        self.im_res_gmseg.save()
        self.im_res_wmseg.save()


    def copy_data_to_tmp(self):
        # copy input image
        if self.param_seg.fname_im is not None:
            sct.copy(self.param_seg.fname_im, self.tmp_dir)
            self.param_seg.fname_im = ''.join(extract_fname(self.param_seg.fname_im)[1:])
        else:
            printv('ERROR: No input image', self.param.verbose, 'error')

        # copy sc seg image
        if self.param_seg.fname_seg is not None:
            sct.copy(self.param_seg.fname_seg, self.tmp_dir)
            self.param_seg.fname_seg = ''.join(extract_fname(self.param_seg.fname_seg)[1:])
        else:
            printv('ERROR: No SC segmentation image', self.param.verbose, 'error')

        # copy level file
        if self.param_seg.fname_level is not None:
            sct.copy(self.param_seg.fname_level, self.tmp_dir)
            self.param_seg.fname_level = ''.join(extract_fname(self.param_seg.fname_level)[1:])

        if self.param_seg.fname_manual_gmseg is not None:
            sct.copy(self.param_seg.fname_manual_gmseg, self.tmp_dir)
            self.param_seg.fname_manual_gmseg = ''.join(extract_fname(self.param_seg.fname_manual_gmseg)[1:])

    def get_im_from_list(self, data):
        im = Image(data)
        # set pix dimension
        im.hdr.structarr['pixdim'][1] = self.param_data.axial_res
        im.hdr.structarr['pixdim'][2] = self.param_data.axial_res
        # set the correct orientation
        im.setFileName('im_to_orient.nii.gz')
        im.save()
        im = set_orientation(im, 'IRP')
        im = set_orientation(im, 'PIL', data_inversion=True)

        return im

    def register_target(self):
        # create dir to store warping fields
        path_warping_fields = 'warp_target'
        if not os.path.exists(path_warping_fields):
            os.mkdir(path_warping_fields)
        # get 3D images from list of slices
        im_dest = self.get_im_from_list(np.array([self.model.mean_image for target_slice in self.target_im]))
        im_src = self.get_im_from_list(np.array([target_slice.im for target_slice in self.target_im]))
        # register list of target slices on list of model mean image
        im_src_reg, fname_src2dest, fname_dest2src = register_data(im_src, im_dest, param_reg=self.param_data.register_param, path_copy_warp=path_warping_fields, rm_tmp=self.param.rm_tmp)
        # rename warping fields
        fname_src2dest_save = 'warp_target2dic.nii.gz'
        fname_dest2src_save = 'warp_dic2target.nii.gz'
        shutil.move(os.path.join(path_warping_fields, fname_src2dest), os.path.join(path_warping_fields, fname_src2dest_save))
        shutil.move(os.path.join(path_warping_fields, fname_dest2src), os.path.join(path_warping_fields, fname_dest2src_save))
        #
        for i, target_slice in enumerate(self.target_im):
            # set moved image for each slice
            target_slice.set(im_m=im_src_reg.data[i])

        return path_warping_fields

    def normalize_target(self):
        # get gm seg from model by level
        gm_seg_model, wm_seg_model = self.model.get_gm_wm_by_level()

        # for each target slice: normalize
        for target_slice in self.target_im:
            level_int = int(round(target_slice.level))
            if level_int not in self.model.intensities.index:
                level_int = 0
            norm_im_M = normalize_slice(target_slice.im_M, gm_seg_model[level_int], wm_seg_model[level_int], self.model.intensities['GM'][level_int], self.model.intensities['WM'][level_int], val_min=self.model.intensities['MIN'][level_int], val_max=self.model.intensities['MAX'][level_int])
            target_slice.set(im_m=norm_im_M)

    def project_target(self):
        projected_target_slices = []
        for target_slice in self.target_im:
            # get slice data in the good shape
            slice_data = target_slice.im_M.flatten()
            slice_data = slice_data.reshape(1, -1)  # data with single sample
            # project slice data into the model
            slice_data_projected = self.model.fitted_model.transform(slice_data)
            projected_target_slices.append(slice_data_projected.reshape(-1, ))
        # store projected target slices
        self.projected_target = projected_target_slices

    def compute_similarities(self):
        list_dic_indexes_by_slice = []
        for i, target_coord in enumerate(self.projected_target):
            list_dic_similarities = []
            for j, dic_coord in enumerate(self.model.fitted_data):
                # compute square norm using coordinates in the model space
                square_norm = np.linalg.norm((target_coord - dic_coord), 2)
                # compute similarity with or without levels
                if self.param_seg.fname_level is not None:
                    # EQUATION WITH LEVELS
                    similarity = exp(-self.param_seg.weight_level * abs(self.target_im[i].level - self.model.slices[j].level)) * exp(-self.param_seg.weight_coord * square_norm)
                else:
                    # EQUATION WITHOUT LEVELS
                    similarity = exp(-self.param_seg.weight_coord * square_norm)
                # add similarity to list
                list_dic_similarities.append(similarity)
            list_norm_similarities = [float(s) / sum(list_dic_similarities) for s in list_dic_similarities]
            # select indexes of most similar slices
            list_dic_indexes = []
            for j, norm_sim in enumerate(list_norm_similarities):
                if norm_sim >= self.param_seg.thr_similarity:
                    list_dic_indexes.append(j)
            # save list of indexes into list by slice
            list_dic_indexes_by_slice.append(list_dic_indexes)

        return list_dic_indexes_by_slice

    def label_fusion(self, list_dic_indexes_by_slice):
        for target_slice in self.target_im:
            # get list of slices corresponding to the indexes
            list_dic_slices = [self.model.slices[j] for j in list_dic_indexes_by_slice[target_slice.id]]
            # average slices GM and WM
            # WM is not used anymore here, but the average_gm_wm() function is used in other parts of the code that need both the GM and WM averages
            data_mean_gm, data_mean_wm = average_gm_wm(list_dic_slices)
            # set negative values to 0
            data_mean_gm[data_mean_gm < 0] = 0

            # store segmentation into target_im
            target_slice.set(gm_seg_m=data_mean_gm)

    def warp_back_seg(self, path_warp):
        # get 3D images from list of slices
        im_dest = self.get_im_from_list(np.array([target_slice.im for target_slice in self.target_im]))
        im_src_gm = self.get_im_from_list(np.array([target_slice.gm_seg_M for target_slice in self.target_im]))
        #
        fname_dic_space2slice_space = os.path.join(path_warp, 'warp_dic2target.nii.gz')
        interpolation = 'linear'
        # warp GM
        im_src_gm_reg = apply_transfo(im_src_gm, im_dest, fname_dic_space2slice_space, interp=interpolation, rm_tmp=self.param.rm_tmp)

        for i, target_slice in enumerate(self.target_im):
            # set GM for each slice
            target_slice.set(gm_seg=im_src_gm_reg.data[i])

    def post_processing(self):
        # DO INTERPOLATION BACK TO ORIGINAL IMAGE
        # get original SC segmentation oriented in RPI
        im_sc_seg_original_rpi = self.info_preprocessing['im_sc_seg_rpi'].copy()
        nx_ref, ny_ref, nz_ref, nt_ref, px_ref, py_ref, pz_ref, pt_ref = im_sc_seg_original_rpi.dim

        # create res GM seg image
        im_res_gmseg = im_sc_seg_original_rpi.copy()
        im_res_gmseg.data = np.zeros(im_res_gmseg.data.shape)

        printv('  Interpolate result back into original space...', self.param.verbose, 'normal')

        for iz, im_iz_preprocessed in enumerate(self.info_preprocessing['interpolated_images']):
            # im gmseg for slice iz
            im_gmseg = im_iz_preprocessed.copy()
            im_gmseg.data = np.zeros(im_gmseg.data.shape)
            im_gmseg.data = self.target_im[iz].gm_seg

            im_res_slice, im_res_tot = (im_gmseg, im_res_gmseg)
            # get reference image for this slice
            # (use only one slice to accelerate interpolation)
            im_ref = im_sc_seg_original_rpi.copy()
            im_ref.data = im_ref.data[:, :, iz]
            im_ref.dim = (nx_ref, ny_ref, 1, nt_ref, px_ref, py_ref, pz_ref, pt_ref)
            # correct reference header for this slice
            [[x_0_ref, y_0_ref, z_0_ref]] = im_ref.transfo_pix2phys(coordi=[[0, 0, iz]])
            im_ref.hdr.as_analyze_map()['qoffset_x'] = x_0_ref
            im_ref.hdr.as_analyze_map()['qoffset_y'] = y_0_ref
            im_ref.hdr.as_analyze_map()['qoffset_z'] = z_0_ref
            im_ref.hdr.set_sform(im_ref.hdr.get_qform())
            im_ref.hdr.set_qform(im_ref.hdr.get_qform())

            # set im_res_slice header with im_sc_seg_original_rpi origin
            im_res_slice.hdr.as_analyze_map()['qoffset_x'] = x_0_ref
            im_res_slice.hdr.as_analyze_map()['qoffset_y'] = y_0_ref
            im_res_slice.hdr.as_analyze_map()['qoffset_z'] = z_0_ref
            im_res_slice.hdr.set_sform(im_res_slice.hdr.get_qform())
            im_res_slice.hdr.set_qform(im_res_slice.hdr.get_qform())

            # get physical coordinates of center of sc
            x_seg, y_seg = (im_sc_seg_original_rpi.data[:, :, iz] > 0).nonzero()
            x_center, y_center = np.mean(x_seg), np.mean(y_seg)
            [[x_center_phys, y_center_phys, z_center_phys]] = im_sc_seg_original_rpi.transfo_pix2phys(coordi=[[x_center, y_center, iz]])

            # get physical coordinates of center of square WITH im_res_slice WITH SAME ORIGIN AS im_sc_seg_original_rpi
            sq_size_pix = int(self.param_data.square_size_size_mm / self.param_data.axial_res)
            [[x_square_center_phys, y_square_center_phys, z_square_center_phys]] = im_res_slice.transfo_pix2phys(
                coordi=[[int(sq_size_pix / 2), int(sq_size_pix / 2), 0]])

            # set im_res_slice header by adding center of SC and center of square (in the correct space) to origin
            im_res_slice.hdr.as_analyze_map()['qoffset_x'] += x_center_phys - x_square_center_phys
            im_res_slice.hdr.as_analyze_map()['qoffset_y'] += y_center_phys - y_square_center_phys
            im_res_slice.hdr.as_analyze_map()['qoffset_z'] += z_center_phys
            im_res_slice.hdr.set_sform(im_res_slice.hdr.get_qform())
            im_res_slice.hdr.set_qform(im_res_slice.hdr.get_qform())

            # reshape data
            im_res_slice.data = im_res_slice.data.reshape((sq_size_pix, sq_size_pix, 1))
            # interpolate to reference image
            interp = 1
            im_res_slice_interp = im_res_slice.interpolate_from_image(im_ref, interpolation_mode=interp, border='nearest')
            # set correct slice of total image with this slice
            if len(im_res_slice_interp.data.shape) == 3:
                shape_x, shape_y, shape_z = im_res_slice_interp.data.shape
                im_res_slice_interp.data = im_res_slice_interp.data.reshape((shape_x, shape_y))
            im_res_tot.data[:, :, iz] = im_res_slice_interp.data

        if self.param_seg.type_seg == 'bin':
            # binarize GM seg
            data_gm = im_res_gmseg.data
            data_gm[data_gm >= self.param_seg.thr_bin] = 1
            data_gm[data_gm < self.param_seg.thr_bin] = 0
            im_res_gmseg.data = data_gm

        # create res WM seg image from GM and SC
        im_res_wmseg = im_sc_seg_original_rpi.copy()
        im_res_wmseg.data = im_res_wmseg.data - im_res_gmseg.data

        # Put res back in original orientation
        printv('  Reorient resulting segmentations to native orientation...', self.param.verbose, 'normal')
        fname_gmseg = 'res_gmseg.nii.gz'
        fname_wmseg = 'res_wmseg.nii.gz'

        im_res_gmseg.setFileName(fname_gmseg)
        im_res_gmseg.save()

        im_res_wmseg.setFileName(fname_wmseg)
        im_res_wmseg.save()

        im_res_gmseg = set_orientation(im_res_gmseg, self.info_preprocessing['orientation'])
        im_res_wmseg = set_orientation(im_res_wmseg, self.info_preprocessing['orientation'])

        return im_res_gmseg, im_res_wmseg

    def validation(self):
        tmp_dir_val = sct.tmp_create(basename="segment_graymatter_validation")
        # copy data into tmp dir val
        sct.copy(self.param_seg.fname_manual_gmseg, tmp_dir_val)
        sct.copy(self.param_seg.fname_seg, tmp_dir_val)
        curdir = os.getcwd()
        os.chdir(tmp_dir_val)
        fname_manual_gmseg = ''.join(extract_fname(self.param_seg.fname_manual_gmseg)[1:])
        fname_seg = ''.join(extract_fname(self.param_seg.fname_seg)[1:])

        im_gmseg = self.im_res_gmseg.copy()
        im_wmseg = self.im_res_wmseg.copy()

        if self.param_seg.type_seg == 'prob':
            im_gmseg = binarize(im_gmseg, thr_max=0.5, thr_min=0.5)
            im_wmseg = binarize(im_wmseg, thr_max=0.5, thr_min=0.5)

        fname_gmseg = 'res_gmseg.nii.gz'
        im_gmseg.setFileName(fname_gmseg)
        im_gmseg.save()

        fname_wmseg = 'res_wmseg.nii.gz'
        im_wmseg.setFileName(fname_wmseg)
        im_wmseg.save()

        # get manual WM seg:
        fname_manual_wmseg = 'manual_wmseg.nii.gz'
        sct_maths.main(args=['-i', fname_seg,
                             '-sub', fname_manual_gmseg,
                             '-o', fname_manual_wmseg])

        # compute DC:
        try:
            status_gm, output_gm = run('sct_dice_coefficient -i ' + fname_manual_gmseg + ' -d ' + fname_gmseg + ' -2d-slices 2')
            status_wm, output_wm = run('sct_dice_coefficient -i ' + fname_manual_wmseg + ' -d ' + fname_wmseg + ' -2d-slices 2')
        except Exception:
            # put ref and res in the same space if needed
            fname_manual_gmseg_corrected = add_suffix(fname_manual_gmseg, '_reg')
            sct_register_multimodal.main(args=['-i', fname_manual_gmseg,
                                               '-d', fname_gmseg,
                                               '-identity', '1'])
            sct_maths.main(args=['-i', fname_manual_gmseg_corrected,
                                 '-bin', '0.1',
                                 '-o', fname_manual_gmseg_corrected])
            #
            fname_manual_wmseg_corrected = add_suffix(fname_manual_wmseg, '_reg')
            sct_register_multimodal.main(args=['-i', fname_manual_wmseg,
                                               '-d', fname_wmseg,
                                               '-identity', '1'])
            sct_maths.main(args=['-i', fname_manual_wmseg_corrected,
                                 '-bin', '0.1',
                                 '-o', fname_manual_wmseg_corrected])
            # recompute DC
            status_gm, output_gm = run('sct_dice_coefficient -i ' + fname_manual_gmseg_corrected + ' -d ' + fname_gmseg + ' -2d-slices 2')
            status_wm, output_wm = run('sct_dice_coefficient -i ' + fname_manual_wmseg_corrected + ' -d ' + fname_wmseg + ' -2d-slices 2')
        # save results to a text file
        fname_dc = 'dice_coefficient_' + extract_fname(self.param_seg.fname_im)[1] + '.txt'
        file_dc = open(fname_dc, 'w')

        if self.param_seg.type_seg == 'prob':
            file_dc.write('WARNING : the probabilistic segmentations were binarized with a threshold at 0.5 to compute the dice coefficient \n')

        file_dc.write('\n--------------------------------------------------------------\nDice coefficient on the Gray Matter segmentation:\n')
        file_dc.write(output_gm)
        file_dc.write('\n\n--------------------------------------------------------------\nDice coefficient on the White Matter segmentation:\n')
        file_dc.write(output_wm)
        file_dc.close()

        # compute HD and MD:
        fname_hd = 'hausdorff_dist_' + extract_fname(self.param_seg.fname_im)[1] + '.txt'
        run('sct_compute_hausdorff_distance -i ' + fname_gmseg + ' -d ' + fname_manual_gmseg + ' -thinning 1 -o ' + fname_hd + ' -v ' + str(self.param.verbose))

        # get out of tmp dir to copy results to output folder
        os.chdir(curdir)
        sct.copy(os.path.join(self.tmp_dir, tmp_dir_val, fname_dc), self.param_seg.path_results)
        sct.copy(os.path.join(self.tmp_dir, tmp_dir_val, fname_hd), self.param_seg.path_results)

        if self.param.rm_tmp:
            shutil.rmtree(tmp_dir_val)

    def compute_ratio(self):
        type_ratio = self.param_seg.ratio

        tmp_dir_ratio = tmp_create(basename="compute_ratio")
        curdir = os.getcwd()
        os.chdir(tmp_dir_ratio)

        fname_gmseg = self.im_res_gmseg.absolutepath
        fname_wmseg = self.im_res_wmseg.absolutepath

        self.im_res_gmseg.save()
        self.im_res_wmseg.save()

        if self.im_res_gmseg.orientation is not 'RPI':
            im_res_gmseg = set_orientation(self.im_res_gmseg, 'RPI')
            im_res_wmseg = set_orientation(self.im_res_wmseg, 'RPI')
            fname_gmseg = im_res_gmseg.absolutepath
            fname_wmseg = im_res_wmseg.absolutepath

        sct_process_segmentation.main(['-i', fname_gmseg, '-p', 'csa', '-ofolder', 'gm_csa', '-no-angle', '1'])
        sct_process_segmentation.main(['-i', fname_wmseg, '-p', 'csa', '-ofolder', 'wm_csa', '-no-angle', '1'])

        gm_csa = open(os.path.join('gm_csa', 'csa_per_slice.txt'), 'r')
        wm_csa = open(os.path.join('wm_csa', 'csa_per_slice.txt'), 'r')
        gm_csa_lines = gm_csa.readlines()
        wm_csa_lines = wm_csa.readlines()
        gm_csa.close()
        wm_csa.close()

        fname_ratio = 'ratio_by_' + type_ratio + '.txt'
        file_ratio = open(fname_ratio, 'w')

        file_ratio.write(type_ratio + ', ratio GM/WM CSA\n')
        csa_gm_wm_by_level = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 13: [], 14: [], 15: [], 16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], 24: []}
        for gm_line, wm_line in zip(gm_csa_lines[1:], wm_csa_lines[1:]):
            i, gm_area, gm_angle = gm_line.split(',')
            j, wm_area, wm_angle = wm_line.split(',')
            assert i == j
            if type_ratio == 'level':
                level_slice = int(self.target_im[int(i)].level)
                csa_gm_wm_by_level[level_slice].append((float(gm_area), float(wm_area)))
            else:
                file_ratio.write(i + ', ' + str(float(gm_area) / float(wm_area)) + '\n')

        if type_ratio == 'level':
            for l, gm_wm_list in sorted(csa_gm_wm_by_level.items()):
                if str(gm_wm_list) != '[]':
                    csa_gm_list = []
                    csa_wm_list = []
                    for gm, wm in gm_wm_list:
                        csa_gm_list.append(gm)
                        csa_wm_list.append(wm)
                    csa_gm = np.mean(csa_gm_list)
                    csa_wm = np.mean(csa_wm_list)
                    file_ratio.write(str(l) + ', ' + str(csa_gm / csa_wm) + '\n')

        file_ratio.close()
        sct.copy(fname_ratio, os.path.join(self.param_seg.path_results, fname_ratio))

        os.chdir(curdir)


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    # create param objects
    param_seg = ParamSeg()
    param_data = ParamData()
    param_model = ParamModel()
    param = Param()

    # get parser
    parser = get_parser()
    arguments = parser.parse(args)

    # set param arguments ad inputted by user
    param_seg.fname_im = arguments["-i"]
    param_seg.fname_im_original = arguments["-i"]
    param_seg.fname_seg = arguments["-s"]

    if '-vertfile' in arguments:
        if extract_fname(arguments['-vertfile'])[1].lower() == "none":
            param_seg.fname_level = None
        elif os.path.isfile(arguments['-vertfile']):
            param_seg.fname_level = arguments['-vertfile']
        else:
            param_seg.fname_level = None
            printv('WARNING: -vertfile input file: "' + arguments['-vertfile'] + '" does not exist.\nSegmenting GM without using vertebral information', 1, 'warning')
    if '-denoising' in arguments:
        param_data.denoising = bool(int(arguments['-denoising']))
    if '-normalization' in arguments:
        param_data.normalization = bool(int(arguments['-normalization']))
    if '-p' in arguments:
        param_data.register_param = arguments['-p']
    if '-w-levels' in arguments:
        param_seg.weight_level = arguments['-w-levels']
    if '-w-coordi' in arguments:
        param_seg.weight_coord = arguments['-w-coordi']
    if '-thr-sim' in arguments:
        param_seg.thr_similarity = arguments['-thr-sim']
    if '-model' in arguments:
        param_model.path_model_to_load = os.path.abspath(arguments['-model'])
    if '-res-type' in arguments:
        param_seg.type_seg = arguments['-res-type']
    if '-ratio' in arguments:
        param_seg.ratio = arguments['-ratio']
    if '-ref' in arguments:
        param_seg.fname_manual_gmseg = arguments['-ref']
    if '-ofolder' in arguments:
        param_seg.path_results = os.path.abspath(arguments['-ofolder'])
    if '-qc' in arguments:
        param_seg.qc = bool(int(arguments['-qc']))
    if '-r' in arguments:
        param.rm_tmp = bool(int(arguments['-r']))
    if '-v' in arguments:
        param.verbose = arguments['-v']

    start_time = time.time()
    seg_gm = SegmentGM(param_seg=param_seg, param_data=param_data, param_model=param_model, param=param)
    seg_gm.segment()
    elapsed_time = time.time() - start_time
    printv('\nFinished! Elapsed time: ' + str(int(round(elapsed_time))) + 's', param.verbose)

    # save quality control and sct.printv(info)
    if param_seg.type_seg == 'bin':
        wm_col = 'red'
        gm_col = 'blue'
        b = '0,1'
    else:
        wm_col = 'blue-lightblue'
        gm_col = 'red-yellow'
        b = '0.4,1'

    if param_seg.qc:
        # output QC image
        printv('\nSave quality control images...', param.verbose, 'normal')
        im = Image(os.path.join(seg_gm.tmp_dir, param_seg.fname_im))
        im.save_quality_control(plane='axial', n_slices=5, seg=seg_gm.im_res_gmseg, thr=float(b.split(',')[0]),
                                cmap_col='red-yellow', path_output=param_seg.path_results)

    if param.rm_tmp:
        # remove tmp_dir
        shutil.rmtree(seg_gm.tmp_dir)

    sct.display_viewer_syntax([param_seg.fname_im_original, seg_gm.fname_res_gmseg, seg_gm.fname_res_wmseg], colormaps=['gray', gm_col, wm_col], opacities=['1', '0.7', '0.7'], verbose=param.verbose)


if __name__ == "__main__":
    sct.start_stream_logger()
    main()
