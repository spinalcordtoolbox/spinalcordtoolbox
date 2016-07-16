#!/usr/bin/env python
#
# This program returns the gray matter segmentation given anatomical, spinal cord segmentation and t2star images
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Sara Dupont
# Modified: 2015-05-20
#
# About the license: see the file LICENSE.TXT
#########################################################################################

'''
INFORMATION:
The model used in this function is compound of:
  - a dictionary of WM/GM contrasted images with their manual segmentations
  - a PCA representing ths dictionary (ie. eigenvectors and eigenvalues)
  - a parameter value
  - an information file indicating which parameters were used to construct this model

A constructed model is provided in the toolbox here: $PATH_SCT/data/gm_model.
It's made from T2* images of 37 subjects and computed with the parameters that gives the best gray matter segmentation results.
However you can compute you own model with your own data or with other parameters and use it to segment gray matter by using  the flag -model path_new_gm_model/.

To do so, you should have a folder (path_to_dataset/) containing for each subject (with a folder per subject):
        - a WM/GM contrasted image (for ex T2*-w) containing 'im' in its name
        - a segmentation of the spinal cord containing 'seg' in its name
        - a manual segmentation of the gray matter containing 'gm' in its name
        - a 'level image' containing 'level' in its name : the level image is an image containing a level label per slice indicating at wich vertebral level correspond this slice (usually obtained by registering the MNI-Poly-AMU or PAM50 template to the WM/GM contrasted image).

Use the following command lines :
# Preprocess the data
msct_gmseg_utils -preprocess path_to_dataset/

# Compute the model
msct_multiatlas_seg -model path_to_dataset_by_slice/ -todo-model compute

Then use the folder gm_model/ (output from msct_multiatlas_seg) in this function the flag -model gm_model/

'''

import sct_utils as sct
import os
import time
import sys
import getopt
from msct_parser import *
from msct_image import Image, get_dimension
import random
from msct_multiatlas_seg import ModelParam, Model, SegmentationParam, SupervisedSegmentationMethod
from msct_gmseg_utils import *
from sct_image import set_orientation, get_orientation_3d, orientation,pad_image
import shutil


def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Segmentation of the white and gray matter.'
                                 ' The segmentation is based on a multi-atlas method that uses a dictionary of pre-segmented gray matter images (already included in SCT) and finds the most similar images for identifying the gray matter using label fusion approach. The model used by this method contains: a template of the white/gray matter segmentation along the cervical spinal cord, and a PCA space to describe the variability of intensity in that template.'
                                 ' This method was inspired from [Asman et al., Medical Image Analysis 2014] and features the following additions:\n'
                                 '- possibility to add information from vertebral levels for improved accuracy\n'
                                 '- intensity normalization of the image to segment (allows the segmentation of any kind of contrast)\n'
                                 '- pre-registration based on affine transformation')
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
                      example='label/template/PAM50_levels.nii.gz')
    parser.add_option(name="-verttype",
                      type_value='multiple_choice',
                      description="if float is selected, vertebral labeling is interpolated for higher accuracy. Default value = "+ModelParam().use_levels,
                      mandatory=False,
                      example=['int', 'float'])
    parser.add_option(name="-vert",
                      mandatory=False,
                      deprecated_by='-vertfile')
    parser.add_option(name="-l",
                      mandatory=False,
                      deprecated_by='-vertfile')

    parser.usage.addSection('SEGMENTATION OPTIONS')
    parser.add_option(name="-weight",
                      type_value='float',
                      description="weight parameter on the level differences to compute the similarities (beta)",
                      mandatory=False,
                      default_value=2.5,
                      example=2.0)
    '''
    parser.add_option(name="-weight-similarity",
                      type_value='multiple_choice',
                      description="Use the modes eigenvalues as weight for the similarity beta",
                      mandatory=False,
                      default_value=0,
                      example=['0', '1'])
    parser.add_option(name="-weight-label-fusion",
                      type_value='multiple_choice',
                      description="Use the similarity beta as weight for the label fusion",
                      mandatory=False,
                      default_value=0,
                      example=['0', '1'])
    '''
    parser.add_option(name="-denoising",
                      type_value='multiple_choice',
                      description="1: Adaptative denoising from F. Coupe algorithm, 0: no  WARNING: It affects the model you should use (if denoising is applied to the target, the model should have been coputed with denoising too",
                      mandatory=False,
                      default_value=1,
                      example=['0', '1'])
    parser.add_option(name="-normalize",
                      type_value='multiple_choice',
                      description="Normalization of the target image's intensity using median intensity values of the WM and the GM, recomended with MT images or other types of contrast than T2*",
                      mandatory=False,
                      default_value=1,
                      example=['0', '1'])
    parser.add_option(name="-medians",
                      type_value=[[','], 'float'],
                      description="Median intensity values in the target white matter and gray matter (separated by a comma without white space)\n"
                                  "If not specified, the mean intensity values of the target WM and GM  are estimated automatically using the dictionary average segmentation by level.\n"
                                  "Only if the -normalize flag is used",
                      mandatory=False,
                      default_value=None,
                      example=["450,540"])
    parser.add_option(name="-k",
                      type_value='float',
                      description="Percentage of variability explained by the kept eigen vectors in the PCA (between 0 and 1)",
                      mandatory=False,
                      default_value=0.8,
                      example=0.6)
    parser.add_option(name="-model",
                      type_value="folder",
                      description="Path to the model data",
                      mandatory=False,
                      example='/home/jdoe/gm_seg_model_data/')
    parser.usage.addSection('\nOUTPUT OTIONS')
    parser.add_option(name="-res-type",
                      type_value='multiple_choice',
                      description="Type of result segmentation : binary or probabilistic",
                      mandatory=False,
                      default_value='prob',
                      example=['binary', 'prob'])
    parser.add_option(name="-ratio",
                      type_value='multiple_choice',
                      description="Compute GM/WM ratio by slice or by vertebral level (average across levels)",
                      mandatory=False,
                      default_value='0',
                      example=['0', 'slice', 'level'])
    '''
    parser.add_option(name="-ratio-level",
                      type_value='str',
                      description="Compute GM/WM ratio across several vertebral levels.",
                      mandatory=False,
                      default_value='0',
                      example='C2:C4')
    '''
    parser.add_option(name="-ofolder",
                      type_value="folder_creation",
                      description="Output folder",
                      mandatory=False,
                      default_value='./',
                      example='gm_segmentation_results/')
    parser.add_option(name="-ref",
                      type_value="file",
                      description="Reference segmentation of the gray matter for segmentation validation (outputs Dice coefficient and Hausdoorff's distance)",
                      mandatory=False,
                      example='manual_gm_seg.nii.gz')
    parser.usage.addSection('MISC')
    parser.add_option(name='-qc',
                      type_value='multiple_choice',
                      description='Output images for quality control.',
                      mandatory=False,
                      example=['0', '1'],
                      default_value='1')
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description='Remove temporary files.',
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])
    parser.add_option(name="-v",
                      type_value='multiple_choice',
                      description="verbose: 0 = nothing, 1 = classic, 2 = expended",
                      mandatory=False,
                      example=['0', '1', '2'],
                      default_value='1')

    return parser


class Preprocessing:
    def __init__(self, target_fname, sc_seg_fname, tmp_dir='', level_fname=None, denoising=True, verbose=1):

        # initiate de file names and copy the files into the temporary directory
        self.original_target = 'target.nii.gz'
        self.t2star = self.original_target
        self.original_sc_seg = 'target_sc_seg.nii.gz'
        self.sc_seg = self.original_sc_seg
        self.resample_to = 0.3
        self.verbose = verbose

        self.tmp_dir = tmp_dir
        self.denoising = denoising
        self.high_res = False

        if level_fname is not None:
            path_level, file_level, ext_level = sct.extract_fname(level_fname)
            if ext_level != '.txt':
                level_fname = check_file_to_niigz(level_fname)
            path_level, file_level, ext_level = sct.extract_fname(level_fname)
            self.fname_level = file_level + ext_level
            sct.run('cp ' + level_fname + ' ' + tmp_dir + '/' + self.fname_level)
        else:
            self.fname_level  = None

        # processes:
        self.copy_to_tmp(target_fname=target_fname, sc_seg_fname=sc_seg_fname)

    def copy_to_tmp(self, target_fname, sc_seg_fname):
        sct.run('cp ' + target_fname + ' ' + self.tmp_dir + '/' + self.original_target)
        sct.run('cp ' + sc_seg_fname + ' ' + self.tmp_dir + '/' + self.original_sc_seg)

    def process(self):
        # preprocessing
        os.chdir(self.tmp_dir)
        im_target = Image(self.original_target)
        im_sc_seg = Image(self.original_sc_seg)

        assert im_target.orientation == im_sc_seg.orientation, "ERROR: the image to segment and it's SC segmentation are not in the same orientation"
        self.original_header = im_target.hdr
        self.original_orientation = im_target.orientation
        index_x = self.original_orientation.find('R') if 'R' in self.original_orientation else self.original_orientation.find('L')
        index_y = self.original_orientation.find('P') if 'P' in self.original_orientation else self.original_orientation.find('A')
        index_z = self.original_orientation.find('I') if 'I' in self.original_orientation else self.original_orientation.find('S')

        # resampling of the images
        nx, ny, nz, nt, px, py, pz, pt = im_target.dim
        nx_s, ny_s, nz_s, nt_s, px_s, py_s, pz_s, pt_s = im_sc_seg.dim

        assert (nx == nx_s) and (ny == ny_s) and (nz == nz_s), "ERROR: the image to segment and it's SC segmentation does not have the same size"

        if self.fname_level is not None and 'nii' in sct.extract_fname(self.fname_level)[2]:
            im_level = Image(self.fname_level)
            assert im_target.orientation == im_level.orientation, "ERROR: the image to segment and the level image are not in the same orientation"
            nx_l, ny_l, nz_l, nt_l, px_l, py_l, pz_l, pt_l = im_level.dim
            assert (nx == nx_l) and (ny == ny_l) and (nz == nz_l), "ERROR: the image to segment and the level image does not have the same size"


        pix_dim = [px, py, pz]
        self.original_px = pix_dim[index_x]
        self.original_py = pix_dim[index_y]

        if round(self.original_px, 2) != self.resample_to or round(self.original_py, 2) != self.resample_to:
            if round(self.original_px, 2) < self.resample_to or round(self.original_py, 2) < self.resample_to:
                sct.printv('\n\n-----------------------------------------------------------------------------------------'
                '\nWARNING: the in-plane resolution of the input image is higher than the resolution of the model images (0.3x0.3mm). '
                'The size of the result images might be different than the original image and an extra post-processing will be added.'
                'This post-processing can be time consuming and can slow down your computer. '
                'To avoid this, please resample your input image to an axial resolution of 0.3x0.3mm.'
                '\n-----------------------------------------------------------------------------------------', self.verbose, 'warning')
                self.high_res = True

            self.t2star = resample_image(self.original_target, npx=self.resample_to, npy=self.resample_to)
            self.sc_seg = resample_image(self.original_sc_seg, binary=True, npx=self.resample_to, npy=self.resample_to)

        # denoising (optional)
        im_target = Image(self.t2star)
        if self.denoising:
            from sct_maths import denoise_nlmeans
            print im_target.data.shape
            im_target.data = denoise_nlmeans(im_target.data)
            im_target.save()
            self.t2star = im_target.file_name + im_target.ext

        box_size = int(22.5/self.resample_to)

        # Pad in case the spinal cord is too close to the edges
        pad_size = box_size/2 + 2
        self.pad = [str(pad_size)]*3

        self.pad[index_z] = str(0)

        t2star_pad = sct.add_suffix(self.t2star, '_pad')
        sc_seg_pad = sct.add_suffix(self.sc_seg, '_pad')
        sct.run('sct_image -i '+self.t2star+' -pad '+self.pad[0]+','+self.pad[1]+','+self.pad[2]+' -o '+t2star_pad)
        sct.run('sct_image -i '+self.sc_seg+' -pad '+self.pad[0]+','+self.pad[1]+','+self.pad[2]+' -o '+sc_seg_pad)
        self.t2star = t2star_pad
        self.sc_seg = sc_seg_pad

        # put data in RPI
        t2star_rpi = sct.add_suffix(self.t2star, '_RPI')
        sc_seg_rpi = sct.add_suffix(self.sc_seg, '_RPI')
        sct.run('sct_image -i '+self.t2star+' -setorient RPI -o '+t2star_rpi)
        sct.run('sct_image -i '+self.sc_seg+' -setorient RPI -o '+sc_seg_rpi)
        self.t2star = t2star_rpi
        self.sc_seg = sc_seg_rpi

        self.square_mask, self.processed_target = crop_t2_star(self.t2star, self.sc_seg, box_size=box_size)

        if self.fname_level is not None and sct.extract_fname(self.fname_level)[2] == '.nii.gz':
            level_orientation = get_orientation_3d(self.fname_level, filename=True)
            if level_orientation != 'IRP':
                self.fname_level = set_orientation(self.fname_level, 'IRP', filename=True)

        os.chdir('..')


class FullGmSegmentation:

    def __init__(self, target_fname, sc_seg_fname, level_fname, ref_gm_seg=None, model=None, compute_ratio=False, model_param=None, seg_param=None):
        self.model_param = model_param
        self.seg_param = seg_param
        self.ratio = compute_ratio
        sct.printv('\nBuilding the appearance model...', verbose=self.seg_param.verbose, type='normal')
        if model is None:
            self.model = Model(model_param=self.model_param)
        else:
            self.model = model
        self.target_fname = check_file_to_niigz(target_fname)
        self.sc_seg_fname = check_file_to_niigz(sc_seg_fname)
        if level_fname is not None:
            level_path, level_file, level_ext = sct.extract_fname(level_fname)
            if level_ext == '.txt':
                self.level_fname = level_fname
            else:
                self.level_fname = check_file_to_niigz(level_fname)
        else:
            self.level_fname = level_fname

        self.ref_gm_seg_fname = ref_gm_seg

        self.gm_seg = None
        self.res_names = {}
        self.dice_name = None
        self.hausdorff_name = None
        self.tmp_dir = 'tmp_' + sct.extract_fname(self.target_fname)[1] + '_' + time.strftime("%y%m%d%H%M%S")+ '_'+str(random.randint(1, 1000000))+'/'

    def segment(self):
        before = time.time()
        sct.run('mkdir ' + self.tmp_dir)

        self.segmentation_pipeline()

        # Generate output files:
        for res_fname in self.res_names.values():
            sct.generate_output_file(self.tmp_dir+res_fname, self.seg_param.output_path+res_fname)
        if self.ref_gm_seg_fname is not None:
            sct.generate_output_file(self.tmp_dir+self.dice_name, self.seg_param.output_path+self.dice_name)
            sct.generate_output_file(self.tmp_dir+self.hausdorff_name, self.seg_param.output_path+self.hausdorff_name)
        if self.ratio:
            sct.generate_output_file(self.tmp_dir+self.ratio_name, self.seg_param.output_path+self.ratio_name)

        after = time.time()
        sct.printv('Done! (in ' + str(after-before) + ' sec) \nTo see the result, type :')
        if self.seg_param.res_type == 'binary':
            wm_col = 'Red'
            gm_col = 'Blue'
            b = '0,1'
        else:
            wm_col = 'Blue-Lightblue'
            gm_col = 'Red-Yellow'
            b = '0.3,1'
        sct.printv('fslview ' + self.target_fname + ' '+self.seg_param.output_path+self.res_names['wm_seg']+' -l '+wm_col+' -t 0.4 -b '+b+' '+self.seg_param.output_path+self.res_names['gm_seg']+' -l '+gm_col+' -t 0.4  -b '+b+' &', self.seg_param.verbose, 'info')

        if self.seg_param.qc:
            # output QC image
            im = Image(self.target_fname)
            im_gmseg = Image(self.seg_param.output_path+self.res_names['gm_seg'])
            im.save_quality_control(plane='axial', n_slices=5, seg=im_gmseg, thr=float(b.split(',')[0]), cmap_col='red-yellow', path_output=self.seg_param.output_path)

        if self.seg_param.remove_tmp:
            sct.printv('Remove temporary folder ...', self.seg_param.verbose, 'normal')
            sct.run('rm -rf '+self.tmp_dir)

    # ------------------------------------------------------------------------------------------------------------------
    def segmentation_pipeline(self):
        sct.printv('\nDoing target pre-processing ...', verbose=self.seg_param.verbose, type='normal')
        self.preprocessed = Preprocessing(self.target_fname, self.sc_seg_fname, tmp_dir=self.tmp_dir, level_fname=self.level_fname, denoising=self.seg_param.target_denoising, verbose=self.seg_param.verbose)
        self.preprocessed.process()

        os.chdir(self.tmp_dir)

        if self.preprocessed.fname_level is not None:
            self.level_to_use = self.preprocessed.fname_level
        else:
            self.level_to_use = None

        sct.printv('\nDoing target gray matter segmentation ...', verbose=self.seg_param.verbose, type='normal')
        self.gm_seg = SupervisedSegmentationMethod(self.preprocessed.processed_target, self.level_to_use, self.model, gm_seg_param=self.seg_param)

        sct.printv('\nDoing result post-processing ...', verbose=self.seg_param.verbose, type='normal')
        self.post_processing()

        if self.ref_gm_seg_fname is not None:
            os.chdir('..')
            ref_gmseg = 'ref_gmseg.nii.gz'
            sct.run('cp ' + self.ref_gm_seg_fname + ' ' + self.tmp_dir + '/' + ref_gmseg)
            os.chdir(self.tmp_dir)
            sct.printv('Computing Dice coefficient and Hausdorff distance ...', verbose=self.seg_param.verbose, type='normal')
            self.dice_name, self.hausdorff_name = self.validation(ref_gmseg)

        if self.ratio:
            sct.printv('\nComputing ratio GM/WM ...', verbose=self.seg_param.verbose, type='normal')
            self.ratio_name = self.compute_ratio(type=self.ratio)

        os.chdir('..')

    # ------------------------------------------------------------------------------------------------------------------
    def post_processing(self):
        square_mask = Image(self.preprocessed.square_mask)
        tmp_res_names = []
        for res_im in [self.gm_seg.res_wm_seg, self.gm_seg.res_gm_seg]:
            res_im_original_space = inverse_square_crop(res_im, square_mask)
            res_im_original_space.save()
            res_im_original_space = set_orientation(res_im_original_space, self.preprocessed.original_orientation)
            res_im_original_space.save()
            res_fname_original_space = res_im_original_space.file_name
            ext = res_im_original_space.ext

            # crop from the same pad size
            output_crop = res_fname_original_space+'_crop'
            sct.run('sct_crop_image -i '+res_fname_original_space+ext+' -dim 0,1,2 -start '+self.preprocessed.pad[0]+','+self.preprocessed.pad[1]+','+self.preprocessed.pad[2]+' -end -'+self.preprocessed.pad[0]+',-'+self.preprocessed.pad[1]+',-'+self.preprocessed.pad[2]+' -o '+output_crop+ext)
            res_fname_original_space = output_crop

            target_path, target_name, target_ext = sct.extract_fname(self.target_fname)
            res_name = target_name + res_im.file_name[len(sct.extract_fname(self.preprocessed.processed_target)[1]):] + '.nii.gz'

            if self.seg_param.res_type == 'binary':
                bin = True
            else:
                bin = False

            old_res_name = resample_image(res_fname_original_space+ext, npx=self.preprocessed.original_px, npy=self.preprocessed.original_py, binary=bin)
            if self.preprocessed.high_res:
                old_res_name_correct_space = sct.extract_fname(old_res_name)[1]+'_correct_spacing'+ext
                os.chdir('..')
                target_fname = sct.extract_fname(self.target_fname)[1]+sct.extract_fname(self.target_fname)[2]
                sct.run('cp '+self.target_fname+' '+self.tmp_dir+'/'+target_fname)

                os.chdir(self.tmp_dir)
                sct.run('sct_register_multimodal -i '+old_res_name+' -d '+target_fname+' -identity 1 -o '+old_res_name_correct_space)
                old_res_name = old_res_name_correct_space

            if self.seg_param.res_type == 'prob':
                # sct.run('fslmaths ' + old_res_name + ' -thr 0.05 ' + old_res_name)
                sct.run('sct_maths -i ' + old_res_name + ' -thr 0.05 -o ' + old_res_name)

            sct.run('cp ' + old_res_name + ' '+res_name)

            tmp_res_names.append(res_name)
        self.res_names['wm_seg'] = tmp_res_names[0]
        self.res_names['gm_seg'] = tmp_res_names[1]

    # ------------------------------------------------------------------------------------------------------------------
    def compute_ratio(self, type='slice'):
        from numpy import mean, nonzero
        from math import isnan
        ratio_dir =  'ratio/'
        sct.run('mkdir '+ratio_dir)
        if type is not 'slice':
            assert self.preprocessed.fname_level is not None, 'No vertebral level information, you cannot compute GM/WM ratio per vertebral level.'
            levels = [int(round(mean(dat[nonzero(dat)]), 0)) if not isnan(mean(dat[nonzero(dat)])) else 0 for dat in Image(self.preprocessed.fname_level).data]
            csa_gm_wm_by_level = {}
            for l in levels:
                csa_gm_wm_by_level[l] = []
        gm_seg = 'res_gmseg.nii.gz'
        wm_seg = 'res_wmseg.nii.gz'
        sct.run('cp '+self.res_names['gm_seg']+' '+ratio_dir+gm_seg)
        sct.run('cp '+self.res_names['wm_seg']+' '+ratio_dir+wm_seg)

        # go to ratio folder
        os.chdir(ratio_dir)

        sct.run('sct_process_segmentation -i '+gm_seg+' -p csa -o gm_ ', error_exit='warning')
        # sct.run('mv csa.txt gm_csa.txt')

        sct.run('sct_process_segmentation -i '+wm_seg+' -p csa -o wm_ ', error_exit='warning')
        # sct.run('mv csa.txt wm_csa.txt')

        gm_csa = open('gm_csa_per_slice.txt', 'r')
        wm_csa = open('wm_csa_per_slice.txt', 'r')

        ratio_fname = 'ratio.txt'
        ratio = open('../'+ratio_fname, 'w')

        gm_lines = gm_csa.readlines()
        wm_lines = wm_csa.readlines()

        gm_csa.close()
        wm_csa.close()

        ratio.write(type+' , ratio GM/WM \n')
        for gm_line, wm_line in zip(gm_lines, wm_lines):
            i, gm_area = gm_line.split(',')
            j, wm_area = wm_line.split(',')
            assert i == j
            if type is not 'slice':
                csa_gm_wm_by_level[levels[int(i)]].append((float(gm_area), float(wm_area)))
            else:
                ratio.write(i+','+str(float(gm_area)/float(wm_area))+'\n')
        if type == 'level':
            for l, gm_wm_list in sorted(csa_gm_wm_by_level.items()):
                csa_gm_list = []
                csa_wm_list = []
                for gm, wm in gm_wm_list:
                    csa_gm_list.append(gm)
                    csa_wm_list.append(wm)
                csa_gm = mean(csa_gm_list)
                csa_wm = mean(csa_wm_list)
                ratio.write(str(l)+','+str(csa_gm/csa_wm)+'\n')
        elif type is not 'slice':
            li, lf = type.split(':')
            level_str_to_int = {'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5, 'C6': 6, 'C7': 7, 'T1': 8, 'T2': 9}
            li = level_str_to_int[li]
            lf = level_str_to_int[lf]
            csa_gm_list = []
            csa_wm_list = []
            for l in range(li, lf+1):
                gm_wm_list = csa_gm_wm_by_level[l]
                for gm, wm in gm_wm_list:
                    csa_gm_list.append(gm)
                    csa_wm_list.append(wm)
            csa_gm = mean(csa_gm_list)
            csa_wm = mean(csa_wm_list)
            ratio.write(type+','+str(csa_gm/csa_wm)+'\n')

        ratio.close()
        os.chdir('..')
        return ratio_fname

    # ------------------------------------------------------------------------------------------------------------------
    def validation(self, ref_gmseg):
        ext = '.nii.gz'
        validation_dir = 'validation/'
        sct.run('mkdir ' + validation_dir)

        gm_seg = 'res_gmseg.nii.gz'
        wm_seg = 'res_wmseg.nii.gz'

        # Copy images to the validation folder
        sct.run('cp '+ref_gmseg+' '+validation_dir+ref_gmseg)
        sct.run('cp '+self.preprocessed.original_sc_seg+' '+validation_dir+self.preprocessed.original_sc_seg)
        sct.run('cp '+self.res_names['gm_seg']+' '+validation_dir+gm_seg)
        sct.run('cp '+self.res_names['wm_seg']+' '+validation_dir+wm_seg)

        # go to validation folder
        os.chdir(validation_dir)

        # get reference WM segmentation from SC segmentation and reference GM segmentation
        ref_wmseg = 'ref_wmseg.nii.gz'
        sct.run('sct_maths -i '+self.preprocessed.original_sc_seg+' -sub '+ref_gmseg+' -o '+ref_wmseg)

        # Binarize results if it was probabilistic results
        if self.seg_param.res_type == 'prob':
            sct.run('sct_maths -i '+gm_seg+' -thr 0.5 -o '+gm_seg)
            sct.run('sct_maths -i '+wm_seg+' -thr 0.4999 -o '+wm_seg)
            sct.run('sct_maths -i '+gm_seg+' -bin -o '+gm_seg)
            sct.run('sct_maths -i '+wm_seg+' -bin -o '+wm_seg)

        # Compute Dice coefficient
        try:
            status_gm, output_gm = sct.run('sct_dice_coefficient -i '+ref_gmseg+' -d '+gm_seg+' -2d-slices 2', error_exit='warning', raise_exception=True)
        except Exception:
            # put the result and the reference in the same space using a registration with ANTs with no iteration:
            corrected_ref_gmseg = sct.extract_fname(ref_gmseg)[1]+'_in_res_space'+ext
            sct.run('isct_antsRegistration -d 3 -t Translation[0] -m MI['+gm_seg+','+ref_gmseg+',1,16] -o [reg_ref_to_res,'+corrected_ref_gmseg+'] -n BSpline[3] -c 0 -f 1 -s 0')
            sct.run('sct_maths -i '+corrected_ref_gmseg+' -thr 0.1 -o '+corrected_ref_gmseg)
            sct.run('sct_maths -i '+corrected_ref_gmseg+' -bin -o '+corrected_ref_gmseg)
            status_gm, output_gm = sct.run('sct_dice_coefficient -i '+corrected_ref_gmseg+' -d '+gm_seg+'  -2d-slices 2', error_exit='warning')

        try:
            status_wm, output_wm = sct.run('sct_dice_coefficient -i '+ref_wmseg+' -d '+wm_seg+' -2d-slices 2', error_exit='warning', raise_exception=True)
        except Exception:
            # put the result and the reference in the same space using a registration with ANTs with no iteration:
            corrected_ref_wmseg = sct.extract_fname(ref_wmseg)[1]+'_in_res_space'+ext
            sct.run('isct_antsRegistration -d 3 -t Translation[0] -m MI['+wm_seg+','+ref_wmseg+',1,16] -o [reg_ref_to_res,'+corrected_ref_wmseg+'] -n BSpline[3] -c 0 -f 1 -s 0')
            sct.run('sct_maths -i '+corrected_ref_wmseg+' -thr 0.1 -o '+corrected_ref_wmseg)
            sct.run('sct_maths -i '+corrected_ref_wmseg+' -bin -o '+corrected_ref_wmseg)
            status_wm, output_wm = sct.run('sct_dice_coefficient -i '+corrected_ref_wmseg+' -d '+wm_seg+'  -2d-slices 2', error_exit='warning')

        dice_name = 'dice_' + sct.extract_fname(self.target_fname)[1] + '_' + self.seg_param.res_type + '.txt'
        dice_fic = open('../'+dice_name, 'w')
        if self.seg_param.res_type == 'prob':
            dice_fic.write('WARNING : the probabilistic segmentations were binarized with a threshold at 0.5 to compute the dice coefficient \n')
        dice_fic.write('\n--------------------------------------------------------------\n'
                       'Dice coefficient on the Gray Matter segmentation:\n')
        dice_fic.write(output_gm)
        dice_fic.write('\n\n--------------------------------------------------------------\n'
                       'Dice coefficient on the White Matter segmentation:\n')
        dice_fic.write(output_wm)
        dice_fic.close()

        # Compute Hausdorff distance
        hd_name = 'hd_' + sct.extract_fname(self.target_fname)[1] + '_' + self.seg_param.res_type + '.txt'
        sct.run('sct_compute_hausdorff_distance -i '+gm_seg+' -d '+ref_gmseg+' -thinning 1 -o '+hd_name+' -v '+str(self.seg_param.verbose))
        sct.run('mv ./' + hd_name + ' ../')

        os.chdir('..')
        return dice_name, hd_name


########################################################################################################################
# ------------------------------------------------------  MAIN ------------------------------------------------------- #
########################################################################################################################
if __name__ == "__main__":
    model_param = ModelParam()
    seg_param = SegmentationParam()
    input_target_fname = None
    input_sc_seg_fname = None
    input_level_fname = None
    input_ref_gm_seg = None
    compute_ratio = False
    if seg_param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        fname_input = model_param.path_model + "/errsm_34.nii.gz"
        fname_input = model_param.path_model + "/errsm_34_seg_in.nii.gz"
    else:
        parser = get_parser()
        arguments = parser.parse(sys.argv[1:])
        input_target_fname = arguments["-i"]
        input_sc_seg_fname = arguments["-s"]
        if "-model" in arguments:
            model_param.path_model = arguments["-model"]
        model_param.todo_model = 'load'
        seg_param.output_path = sct.slash_at_the_end(arguments["-ofolder"], slash=1)

        if "-vertfile" in arguments:
            input_level_fname = arguments["-vertfile"]
        if "-verttype" in arguments:
            if input_level_fname is None:
                sct.printv('WARNING: no input vertebral level file. -verttype can not be use and will be ignored.',1, 'warning')
            else:
                model_param.use_levels = arguments["-verttype"]

        if "-weight" in arguments:
            model_param.weight_gamma = arguments["-weight"]
        if "-weight-similarity" in arguments:
            model_param.mode_weight_similarity = bool(int(arguments["-weight-similarity"]))
        if "-weight-label-fusion" in arguments:
            model_param.weight_label_fusion = bool(int(arguments["-weight-label-fusion"]))
        if "-denoising" in arguments:
            seg_param.target_denoising = bool(int(arguments["-denoising"]))
        if "-normalize" in arguments:
            seg_param.target_normalization = bool(int(arguments["-normalize"]))
        if "-means" in arguments:
            seg_param.target_means = arguments["-means"]
        if "-k" in arguments:
            model_param.k = arguments["-k"]
        if "-ratio" in arguments:
            if arguments["-ratio"] == '0':
                compute_ratio = False
            else:
                compute_ratio = arguments["-ratio"]
        '''
        if "-ratio-level" in arguments:
            if arguments["-ratio-level"] == '0':
                compute_ratio = False
            else:
                if ':' in arguments["-ratio-level"]:
                    compute_ratio = arguments["-ratio-level"]
                else:
                    sct.printv('WARNING: -ratio-level function should be used with a range of vertebral levels (for ex: "C2:C5"). Ignoring option.', 1, 'warning')
        '''
        if "-res-type" in arguments:
            seg_param.res_type = arguments["-res-type"]
        if "-ref" in arguments:
            input_ref_gm_seg = arguments["-ref"]
        seg_param.verbose = int(arguments["-v"])
        model_param.verbose = int(arguments["-v"])
        seg_param.qc = int(arguments["-qc"])
        seg_param.remove_tmp = int(arguments["-r"])

        '''
        if not sct.check_file_exist(input_level_fname):
                sct.printv(parser.usage.generate(error='ERROR: the file '+input_level_fname+' does not exist. Please specify a file with flag -vertfile or if you don\'t want to use vertebral level information, use flag -use-levels 0' ))
        '''
        if input_level_fname is None:
            model_param.use_levels = '0'
            model_param.weight_gamma = 0

    gmsegfull = FullGmSegmentation(input_target_fname, input_sc_seg_fname, input_level_fname, ref_gm_seg=input_ref_gm_seg, compute_ratio=compute_ratio, model_param=model_param, seg_param=seg_param)
    gmsegfull.segment()
