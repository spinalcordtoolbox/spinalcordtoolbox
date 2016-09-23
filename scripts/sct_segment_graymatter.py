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
from msct_multiatlas_seg import Param, ParamData, ParamModel, Model
from msct_gmseg_utils import pre_processing, register_data, apply_transfo, normalize_slice, average_gm_wm, binarize
from sct_utils import printv, tmp_create, extract_fname, add_suffix, slash_at_the_end, run
import sct_image
from sct_image import set_orientation
from msct_image import Image
from msct_parser import *
import sct_maths, sct_register_multimodal
from math import exp
import numpy as np
import shutil, os, sys, time


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
    parser.add_option(name="-vertfile",
                      type_value="file",
                      description='Labels of vertebral levels. This could either be an image (e.g., label/template/PAM50_levels.nii.gz) or a text file that specifies "slice,level" at each line. Example:\n'
                      "0,3\n"
                      "1,3\n"
                      "2,4\n"
                      "3,4\n"
                      "4,4",
                      mandatory=False,
                      default_value=ParamSeg().fname_level,
                      example='label/template/PAM50_levels.nii.gz')
    parser.add_option(name="-vert",
                      mandatory=False,
                      deprecated_by='-vertfile')
    parser.add_option(name="-l",
                      mandatory=False,
                      deprecated_by='-vertfile')

    parser.usage.addSection('SEGMENTATION OPTIONS')
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
    parser.usage.addSection('\nOUTPUT OTIONS')
    parser.add_option(name="-res-type",
                      type_value='multiple_choice',
                      description="Type of result segmentation : binary or probabilistic",
                      mandatory=False,
                      default_value=ParamSeg().type_seg,
                      example=['bin', 'prob'])
    # parser.add_option(name="-ratio",
    #                   type_value='multiple_choice',
    #                   description="Compute GM/WM ratio by slice or by vertebral level (average across levels)",
    #                   mandatory=False,
    #                   default_value='0',
    #                   example=['0', 'slice', 'level'])
    parser.add_option(name="-ref",
                      type_value="file",
                      description="Reference segmentation of the gray matter for segmentation validation --> output Dice coefficient and Hausdorff's and median distances)",
                      mandatory=False,
                      example='manual_gm_seg.nii.gz')
    parser.add_option(name="-ofolder",
                      type_value="folder_creation",
                      description="Output folder",
                      mandatory=False,
                      default_value=ParamSeg().path_results,
                      example='gm_segmentation_results/')
    parser.usage.addSection('MISC')
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
        self.fname_level = 'label/template/PAM50_levels_continuous.nii.gz'
        self.fname_manual_gmseg = None
        self.path_results = './'

        # param to compute similarities:
        self.weight_level = 2.5 # gamma
        self.weight_coord = 0.0065 # tau --> need to be validated for specific dataset
        self.thr_similarity = 0.0005 # epsilon but on normalized to 1 similarities (by slice of dic and slice of target)
        # TODO = find the best thr

        self.type_seg = 'prob' # 'prob' or 'bin'

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
        self.tmp_dir = tmp_create(verbose=self.param.verbose) # path to tmp directory

        self.target_im = None # list of slices
        self.info_preprocessing = None # dic containing {'orientation': 'xxx', 'im_sc_seg_rpi': im, 'interpolated_images': [list of im = interpolated image data per slice]}

        self.projected_target = None # list of coordinates of the target slices in the model reduced space
        self.im_res_gmseg = None
        self.im_res_wmseg = None


    def segment(self):
        self.copy_data_to_tmp()
        # go to tmp directory
        os.chdir(self.tmp_dir)
        # load model
        self.model.load_model()

        # pad images to avoid bug with centermassrot if SC is too close to the edges
        sct_image.main(['-i', self.param_seg.fname_im, '-pad-asym', '25,25,25,25,0,0', '-o', self.param_seg.fname_im])
        sct_image.main(['-i', self.param_seg.fname_seg, '-pad-asym', '25,25,25,25,0,0', '-o', self.param_seg.fname_seg])

        self.target_im, self.info_preprocessing = pre_processing(self.param_seg.fname_im, self.param_seg.fname_seg, self.param_seg.fname_level, new_res=self.param_data.axial_res, square_size_size_mm=self.param_data.square_size_size_mm, denoising=self.param_data.denoising, verbose=self.param.verbose, rm_tmp=self.param.rm_tmp)

        printv('\nRegistering target image to model data ...', self.param.verbose, 'normal')
        # register target image to model dictionary space
        path_warp = self.register_target()

        printv('\nNormalizing intensity of target image ...', self.param.verbose, 'normal')
        self.normalize_target()

        printv('\nProjecting target image into the model reduced space ...', self.param.verbose, 'normal')
        self.project_target()

        printv('\nComputing similarities between target slices and model slices using model reduced space ...', self.param.verbose, 'normal')
        list_dic_indexes_by_slice = self.compute_similarities()

        printv('\nDoing label fusion of model slices most similar to target slices ...', self.param.verbose, 'normal')
        self.label_fusion(list_dic_indexes_by_slice)

        printv('\nWarping back segmentation into image space...', self.param.verbose, 'normal')
        self.warp_back_seg(path_warp)

        printv('\nPost-processing ...', self.param.verbose, 'normal')
        self.im_res_gmseg, self.im_res_wmseg = self.post_processing()

        if (self.param_seg.path_results != './') and (not os.path.exists('../'+self.param_seg.path_results)):
            # create output folder
            printv('\nCreate output folder ...', self.param.verbose, 'normal')
            os.chdir('..')
            os.mkdir(self.param_seg.path_results)
            os.chdir(self.tmp_dir)

        if self.param_seg.fname_manual_gmseg is not None:
            # compute validation metrics
            printv('\nCompute validation metrics...', self.param.verbose, 'normal')
            self.validation()

        # go back to original directory
        os.chdir('..')
        printv('\nSaving result GM and WM segmentation...', self.param.verbose, 'normal')
        fname_res_gmseg = self.param_seg.path_results+add_suffix(''.join(extract_fname(self.param_seg.fname_im)[1:]), '_gmseg')
        fname_res_wmseg = self.param_seg.path_results+add_suffix(''.join(extract_fname(self.param_seg.fname_im)[1:]), '_wmseg')

        self.im_res_gmseg.setFileName(fname_res_gmseg)
        self.im_res_wmseg.setFileName(fname_res_wmseg)

        self.im_res_gmseg.save()
        self.im_res_wmseg.save()

        # save quality control and print info
        if self.param_seg.type_seg == 'bin':
            wm_col = 'Red'
            gm_col = 'Blue'
            b = '0,1'
        else:
            wm_col = 'Blue-Lightblue'
            gm_col = 'Red-Yellow'
            b = '0.4,1'

        if self.param_seg.qc:
            # output QC image
            printv('\nSaving quality control images...', self.param.verbose, 'normal')
            im = Image(self.tmp_dir+self.param_seg.fname_im)
            im.save_quality_control(plane='axial', n_slices=5, seg=self.im_res_gmseg, thr=float(b.split(',')[0]), cmap_col='red-yellow', path_output=self.param_seg.path_results)

        printv('\n--> To visualize the results, write:\n'
               'fslview '+self.param_seg.fname_im_original+' '+fname_res_gmseg+' -b '+b+' -l '+gm_col+' -t 0.7 '+fname_res_wmseg+' -b '+b+' -l '+wm_col+' -t 0.7  & \n', self.param.verbose, 'info')

        if self.param.rm_tmp:
            # remove tmp_dir
            shutil.rmtree(self.tmp_dir)


    def copy_data_to_tmp(self):
        # copy input image
        if self.param_seg.fname_im is not None:
            shutil.copy(self.param_seg.fname_im, self.tmp_dir)
            self.param_seg.fname_im = ''.join(extract_fname(self.param_seg.fname_im)[1:])
        else:
            printv('ERROR: No input image', self.param.verbose, 'error')

        # copy sc seg image
        if self.param_seg.fname_seg is not None:
            shutil.copy(self.param_seg.fname_seg, self.tmp_dir)
            self.param_seg.fname_seg = ''.join(extract_fname(self.param_seg.fname_seg)[1:])
        else:
            printv('ERROR: No SC segmentation image', self.param.verbose, 'error')

        # copy level file
        if self.param_seg.fname_level is not None:
            shutil.copy(self.param_seg.fname_level, self.tmp_dir)
            self.param_seg.fname_level = ''.join(extract_fname(self.param_seg.fname_level)[1:])

        if self.param_seg.fname_manual_gmseg is not None:
            shutil.copy(self.param_seg.fname_manual_gmseg, self.tmp_dir)
            self.param_seg.fname_manual_gmseg = ''.join(extract_fname(self.param_seg.fname_manual_gmseg)[1:])

    def register_target(self):
        # create dir to store warping fields
        path_warping_fields = 'warp_target/'
        if not os.path.exists(path_warping_fields):
            os.mkdir(path_warping_fields)

        # get destination image
        im_dest = Image(self.model.mean_image)

        for target_slice in self.target_im:
            im_src = Image(target_slice.im)
            # register slice image to mean dictionary image
            im_src_reg, fname_src2dest, fname_dest2src = register_data(im_src, im_dest, param_reg=self.param_data.register_param, path_copy_warp=path_warping_fields, rm_tmp=self.param.rm_tmp)

            # rename warping fields
            fname_src2dest_slice = 'warp_target_slice'+str(target_slice.id)+'2dic.nii.gz'
            fname_dest2src_slice = 'warp_dic2target_slice' + str(target_slice.id) + '.nii.gz'
            shutil.move(path_warping_fields+fname_src2dest, path_warping_fields+fname_src2dest_slice)
            shutil.move(path_warping_fields+fname_dest2src, path_warping_fields+fname_dest2src_slice)

            # set moved image
            target_slice.set(im_m=im_src_reg.data)

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
            slice_data = slice_data.reshape(1, -1) # data with single sample
            # project slice data into the model
            slice_data_projected = self.model.fitted_model.transform(slice_data)
            projected_target_slices.append(slice_data_projected)
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
            list_norm_similarities =  [float(s)/sum(list_dic_similarities) for s in list_dic_similarities]
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
            data_mean_gm, data_mean_wm = average_gm_wm(list_dic_slices)
            if self.param_seg.type_seg == 'bin':
                # binarize GM seg
                data_mean_gm[data_mean_gm >= 0.5] = 1
                data_mean_gm[data_mean_gm < 0.5] = 0
                # binarize WM seg
                data_mean_wm[data_mean_wm >= 0.5] = 1
                data_mean_wm[data_mean_wm < 0.5] = 0
            # store segmentation into target_im
            target_slice.set(gm_seg_m=data_mean_gm, wm_seg_m=data_mean_wm)

    def warp_back_seg(self, path_warp):
        for target_slice in self.target_im:
            printv('\nSlice '+str(target_slice.id)+':', self.param.verbose, 'normal')
            fname_dic_space2slice_space = slash_at_the_end(path_warp, slash=1)+'warp_dic2target_slice' + str(target_slice.id) + '.nii.gz'
            im_dest = Image(target_slice.im)
            interpolation = 'nn' if self.param_seg.type_seg == 'bin' else 'linear'
            # warp GM
            im_src_gm = Image(target_slice.gm_seg_M)
            im_src_gm_reg = apply_transfo(im_src_gm, im_dest, fname_dic_space2slice_space, interp=interpolation, rm_tmp=self.param.rm_tmp)
            # warp WM
            im_src_wm = Image(target_slice.wm_seg_M)
            im_src_wm_reg = apply_transfo(im_src_wm, im_dest, fname_dic_space2slice_space, interp=interpolation, rm_tmp=self.param.rm_tmp)
            # set slice attributes
            target_slice.set(gm_seg=im_src_gm_reg.data, wm_seg=im_src_wm_reg.data)

    def post_processing(self):
        ## DO INTERPOLATION BACK TO ORIGINAL IMAGE
        # get original SC segmentation oriented in RPI
        im_sc_seg_original_rpi = self.info_preprocessing['im_sc_seg_rpi'].copy()
        nx_ref, ny_ref, nz_ref, nt_ref, px_ref, py_ref, pz_ref, pt_ref = im_sc_seg_original_rpi.dim

        # create res GM seg image
        im_res_gmseg = im_sc_seg_original_rpi.copy()
        im_res_gmseg.data = np.zeros(im_res_gmseg.data.shape)
        # create res WM seg image
        im_res_wmseg = im_sc_seg_original_rpi.copy()
        im_res_wmseg.data = np.zeros(im_res_wmseg.data.shape)

        printv('\n\tInterpolate result back into original space ...', self.param.verbose, 'normal')


        for iz, im_iz_preprocessed in enumerate(self.info_preprocessing['interpolated_images']):
            # im gmseg for slice iz
            im_gmseg = im_iz_preprocessed.copy()
            im_gmseg.data = np.zeros(im_gmseg.data.shape)
            im_gmseg.data = self.target_im[iz].gm_seg

            # im wmseg for slice iz
            im_wmseg = im_iz_preprocessed.copy()
            im_wmseg.data = np.zeros(im_wmseg.data.shape)
            im_wmseg.data = self.target_im[iz].wm_seg

            for im_res_slice, im_res_tot in [(im_gmseg, im_res_gmseg), (im_wmseg, im_res_wmseg)]:
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
                interp = 0 if self.param_seg.type_seg == 'bin' else 1
                im_res_slice_interp = im_res_slice.interpolate_from_image(im_ref, interpolation_mode=interp, border='nearest')
                # set correct slice of total image with this slice
                if len(im_res_slice_interp.data.shape) == 3:
                    shape_x, shape_y, shape_z = im_res_slice_interp.data.shape
                    im_res_slice_interp.data = im_res_slice_interp.data.reshape((shape_x, shape_y))
                im_res_tot.data[:, :, iz] = im_res_slice_interp.data
        printv('\n\tPut result into original orientation ...', self.param.verbose, 'normal')

        ## PUT RES BACK IN ORIGINAL ORIENTATION
        im_res_gmseg.setFileName('res_gmseg.nii.gz')
        im_res_gmseg.save()
        im_res_gmseg = set_orientation(im_res_gmseg, self.info_preprocessing['orientation'])

        im_res_wmseg.setFileName('res_wmseg.nii.gz')
        im_res_wmseg.save()
        im_res_wmseg = set_orientation(im_res_wmseg, self.info_preprocessing['orientation'])

        return im_res_gmseg, im_res_wmseg

    def validation(self):
        tmp_dir_val = 'tmp_validation/'
        if not os.path.exists(tmp_dir_val):
            os.mkdir(tmp_dir_val)
        # copy data into tmp dir val
        shutil.copy(self.param_seg.fname_manual_gmseg, tmp_dir_val)
        shutil.copy(self.param_seg.fname_seg, tmp_dir_val)
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

        ## compute DC:
        try:
            status_gm, output_gm = run('sct_dice_coefficient -i ' + fname_manual_gmseg + ' -d ' + fname_gmseg + ' -2d-slices 2',error_exit='warning', raise_exception=True)
            status_wm, output_wm = run('sct_dice_coefficient -i ' + fname_manual_wmseg + ' -d ' + fname_wmseg + ' -2d-slices 2',error_exit='warning', raise_exception=True)
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
            status_gm, output_gm = run('sct_dice_coefficient -i ' + fname_manual_gmseg_corrected + ' -d ' + fname_gmseg + ' -2d-slices 2',error_exit='warning', raise_exception=True)
            status_wm, output_wm = run('sct_dice_coefficient -i ' + fname_manual_wmseg_corrected + ' -d ' + fname_wmseg + ' -2d-slices 2',error_exit='warning', raise_exception=True)
        # save results to a text file
        fname_dc = 'dice_coefficient_' + sct.extract_fname(self.param_seg.fname_im)[1] + '.txt'
        file_dc = open(fname_dc, 'w')

        if self.param_seg.type_seg == 'prob':
            file_dc.write('WARNING : the probabilistic segmentations were binarized with a threshold at 0.5 to compute the dice coefficient \n')

        file_dc.write('\n--------------------------------------------------------------\nDice coefficient on the Gray Matter segmentation:\n')
        file_dc.write(output_gm)
        file_dc.write('\n\n--------------------------------------------------------------\nDice coefficient on the White Matter segmentation:\n')
        file_dc.write(output_wm)
        file_dc.close()

        ## compute HD and MD:
        fname_hd = 'hausdorff_dist_' + sct.extract_fname(self.param_seg.fname_im)[1] + '.txt'
        run('sct_compute_hausdorff_distance -i ' + fname_gmseg + ' -d ' + fname_manual_gmseg + ' -thinning 1 -o ' + fname_hd + ' -v ' + str(self.param.verbose))

        # get out of tmp dir to copy results to output folder
        os.chdir('../..')
        shutil.copy(self.tmp_dir+tmp_dir_val+'/'+fname_dc, self.param_seg.path_results)
        shutil.copy(self.tmp_dir + tmp_dir_val + '/' + fname_hd, self.param_seg.path_results)

        os.chdir(self.tmp_dir)

        if self.param.rm_tmp:
            shutil.rmtree(tmp_dir_val)


########################################################################################################################
# ------------------------------------------------------  MAIN ------------------------------------------------------- #
########################################################################################################################

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
        param_seg.fname_level = arguments['-vertfile']
    if '-denoising' in arguments:
        param_data.denoising = arguments['-denoising']
    if '-normalization' in arguments:
        param_data.normalization = arguments['-normalization']
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
        param_seg.type_seg= arguments['-res-type']
    if '-ref' in arguments:
        param_seg.fname_manual_gmseg = arguments['-ref']
    if '-ofolder' in arguments:
        param_seg.path_results= arguments['-ofolder']
    if '-qc' in arguments:
        param_seg.qc = bool(int(arguments['-qc']))
    if '-r' in arguments:
        param.rm_tmp= bool(int(arguments['-r']))
    if '-v' in arguments:
        param.verbose= arguments['-v']

    if not os.path.isfile(param_seg.fname_level):
        param_seg.fname_level = None

    seg_gm = SegmentGM(param_seg=param_seg, param_data=param_data, param_model=param_model, param=param)
    start = time.time()
    seg_gm.segment()
    end = time.time()
    t = end - start
    printv('Done in ' + str(int(round(t / 60))) + ' min, ' + str(round(t % 60,1)) + ' sec', param.verbose, 'info')

if __name__ == "__main__":
    main()
