#!/usr/bin/env python
########################################################################################################################
#
#
# Processing function for other group's data
#
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Sara Dupont
# Modified: 2015-03-24
#
# About the license: see the file LICENSE.TXT
########################################################################################################################

import os
import numpy as np
import sys
import sct_utils as sct
from msct_image import Image
from msct_parser import Parser
from msct_gmseg_utils import resample_image, get_key_from_val


# ------------------------------------------------------------------------------------------------------------------
def amu_processing(data_path):
    """
    get a segmentation image of the spinal cord an of the graymatter from a three level mask

    :param data_path: path to the data

    :return:
    """
    im_ext = '.nii'
    if data_path[-1] == '/':
        data_path = data_path[:-1]
    original_path = os.path.abspath('.')
    os.chdir(data_path)
    for subject_dir in os.listdir('.'):
        subject_path = data_path + '/' + subject_dir
        if os.path.isdir(subject_dir):
            os.chdir(subject_dir)
            sc_seg_list = []
            gm_seg_list = []
            im_list = []
            for file_name in os.listdir('.'):
                ext = sct.extract_fname(file_name)[2]
                if 'mask' in file_name and ext != '.hdr':
                    mask_im = Image(file_name)

                    sc_seg_im = mask_im.copy()
                    sc_seg_im.file_name = sct.extract_fname(file_name)[1][:-5] + '_manual_sc_seg'
                    sc_seg_im.ext = '.nii.gz'
                    sc_seg_im.data = (sc_seg_im.data > 1).astype(int)
                    # sc_seg_im = Image(param=sc_seg, absolutepath=subject_path + '/' + sct.extract_fname(file_name)[1][:-5] + '_manual_sc_seg.nii.gz')
                    # sc_seg_im.orientation = 'RPI'
                    sc_seg_im.save()
                    sc_seg_list.append(sc_seg_im.file_name + sc_seg_im.ext)

                    gm_seg_im = mask_im.copy()
                    gm_seg_im.file_name = sct.extract_fname(file_name)[1][:-5] + '_manual_gm_seg'
                    gm_seg_im.ext = '.nii.gz'
                    gm_seg_im.data = (gm_seg_im.data > 2).astype(int)
                    # gm_seg_im = Image(param=gm_seg, absolutepath=subject_path + '/' + sct.extract_fname(file_name)[1][:-5] + '_manual_gm_seg.nii.gz')
                    # gm_seg_im.orientation = 'RPI'
                    gm_seg_im.save()
                    gm_seg_list.append(gm_seg_im.file_name + gm_seg_im.ext)

                    im_list.append(file_name[:2] + im_ext)

            # merging the slice images into a 3D image
            im_list.reverse()
            gm_seg_list.reverse()
            sc_seg_list.reverse()
            cmd_merge = 'fslmerge -z '
            im_name = subject_dir + '_im.nii.gz '
            cmd_merge_im = cmd_merge + im_name
            gmseg_name = subject_dir + '_manual_gmseg.nii.gz '
            cmd_merge_gm_seg = cmd_merge + gmseg_name
            scseg_name = subject_dir + '_manual_scseg.nii.gz '
            cmd_merge_sc_seg = cmd_merge + scseg_name

            for im_i, gm_i, sc_i in zip(im_list, gm_seg_list, sc_seg_list):
                cmd_merge_im += im_i + ' '
                cmd_merge_gm_seg += gm_i + ' '
                cmd_merge_sc_seg += sc_i + ' '

            sct.run(cmd_merge_im)
            sct.run(cmd_merge_gm_seg)
            sct.run(cmd_merge_sc_seg)

            # creating a level image
            level_label = {0: '', 1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7', 8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6'}
            level_dat = np.zeros((mask_im.data.shape[0], mask_im.data.shape[1], len(im_list)))
            for i, im_i_name in enumerate(im_list):
                level_dat.T[:][:][i] = get_key_from_val(level_label, im_i_name[:2].upper())
            Image(param=level_dat, absolutepath=subject_dir + '_levels.nii.gz').save()

            # resampling
            resample_image(im_name)
            resample_image(gmseg_name, binary=True, thr=0.45)
            resample_image(scseg_name, binary=True, thr=0.55)

            # organizing data
            sct.run('mkdir original_data/')
            sct.run('mkdir extracted_data/')
            sct.run('mkdir 3d_data/')
            sct.run('mkdir 3d_resampled_data/')

            for file_name in os.listdir('.'):
                if 'mask' in file_name:
                    sct.run('mv ' + file_name + ' original_data/')
                elif 'manual' in file_name and 'G1' not in file_name:
                    sct.run('mv ' + file_name + ' extracted_data/')
                elif 'resampled.nii' in file_name:
                    sct.run('mv ' + file_name + ' 3d_resampled_data/')
                elif 'G1' in file_name:
                    sct.run('mv ' + file_name + ' 3d_data/')
                elif not os.path.isdir(os.path.abspath('.') + '/' + file_name):
                    sct.run('mv ' + file_name + ' original_data/')

            os.chdir('..')
    os.chdir(original_path)


# ------------------------------------------------------------------------------------------------------------------
def vanderbilt_processing(data_path):
    """
    get a segmentation image of the spinal cord an of the graymatter from a three level mask

    :param data_path: path to the data

    :return:
    """
    im_ext = '.nii.gz'
    if data_path[-1] == '/':
        data_path = data_path[:-1]
    original_path = os.path.abspath('.')
    os.chdir(data_path)
    for subject_dir in os.listdir('.'):
        if os.path.isdir(subject_dir):
            os.chdir(subject_dir)
            sc_seg_list = []
            gm_seg_list = []
            im_list = []
            for file_name in os.listdir('.'):
                if 'seg' in file_name:
                    mask_im = Image(file_name)

                    sc_seg_im = mask_im.copy()
                    sc_seg_im.file_name = sct.extract_fname(file_name)[1][:-4] + '_manual_sc_seg'
                    sc_seg_im.ext = '.nii.gz'
                    sc_seg_im.data = (sc_seg_im.data > 0).astype(int)
                    sc_seg_im.save()
                    sc_seg_list.append(sc_seg_im.file_name + sc_seg_im.ext)

                    gm_seg_im = mask_im.copy()
                    gm_seg_im.file_name = sct.extract_fname(file_name)[1][:-4] + '_manual_gm_seg'
                    gm_seg_im.ext = '.nii.gz'
                    gm_seg_im.data = (gm_seg_im.data > 1).astype(int)
                    gm_seg_im.save()
                    gm_seg_list.append(gm_seg_im.file_name + gm_seg_im.ext)

                    im_list.append(file_name[:17] + im_ext)

            # merging the slice images into a 3D image
            cmd_merge = 'fslmerge -z '
            im_name = subject_dir + '_im.nii.gz'
            cmd_merge_im = cmd_merge + im_name
            gmseg_name = subject_dir + '_manual_gmseg.nii.gz'
            cmd_merge_gm_seg = cmd_merge + gmseg_name
            scseg_name = subject_dir + '_manual_scseg.nii.gz'
            cmd_merge_sc_seg = cmd_merge + scseg_name

            for im_i, gm_i, sc_i in zip(im_list, gm_seg_list, sc_seg_list):
                cmd_merge_im += ' ' + im_i
                cmd_merge_gm_seg += ' ' + gm_i
                cmd_merge_sc_seg += ' ' + sc_i

            sct.run(cmd_merge_im)
            sct.run(cmd_merge_gm_seg)
            sct.run(cmd_merge_sc_seg)

            label_slices = [im_slice.split('_')[-1][2:4] for im_slice in im_list]
            i_slice_to_level = {0: 6, 1: 6, 2: 6, 3: 6, 4: 6, 5: 5, 6: 5, 7: 5, 8: 5, 9: 5, 10: 4, 11: 4, 12: 4, 13: 4, 14: 4, 15: 3, 16: 3, 17: 3, 18: 3, 19: 3, 20: 3, 21: 2, 22: 2, 23: 2, 24: 2, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1}

            level_dat = np.zeros((mask_im.data.shape[0], mask_im.data.shape[1], len(im_list)))
            for i, l_slice in enumerate(label_slices):
                i_slice = int(l_slice) - 1
                level_dat.T[:][:][i] = i_slice_to_level[i_slice]
            Image(param=level_dat, absolutepath=subject_dir + '_levels.nii.gz').save()

            # resampling
            resample_image(im_name)
            resample_image(gmseg_name, binary=True)
            resample_image(scseg_name, binary=True)

            # organizing data
            sct.run('mkdir original_data/')
            sct.run('mkdir extracted_data/')
            sct.run('mkdir 3d_data/')
            sct.run('mkdir 3d_resampled_data/')
            sct.run('mkdir dic_data/')

            for file_name in os.listdir('.'):
                if '_manual_gm_seg' in file_name and 'sl' in file_name:
                    sct.run('cp ' + file_name + ' dic_data/')
                    sct.run('cp ' + file_name[:-21] + '.nii.gz dic_data/')
            for file_name in os.listdir('.'):
                if 'sl' in file_name and 'manual' not in file_name:
                    sct.run('mv ' + file_name + ' original_data/')
                elif 'manual' in file_name and 'sl' in file_name:
                    sct.run('mv ' + file_name + ' extracted_data/')
                elif 'resampled.nii' in file_name:
                    sct.run('mv ' + file_name + ' 3d_resampled_data/')
                elif '_sl' not in file_name and not os.path.isdir(os.path.abspath('.') + '/' + file_name) or 'level' in file_name:
                    sct.run('mv ' + file_name + ' 3d_data/')
                elif not os.path.isdir(os.path.abspath('.') + '/' + file_name):
                    sct.run('mv ' + file_name + ' original_data/')

            os.chdir('..')
    os.chdir(original_path)


if __name__ == "__main__":
        # Initialize the parser
        parser = Parser(__file__)
        parser.usage.set_description('Utility functions for the gray matter segmentation')
        parser.add_option(name="-AMU",
                          type_value="folder",
                          description="Path to a dictionary folder with images in the AMU format to be processed",
                          mandatory=False,
                          example='dictionary/')
        parser.add_option(name="-vanderbilt",
                          type_value="folder",
                          description="Path to a dictionary folder with images in the vanderbilt format to be processed",
                          mandatory=False,
                          example='dictionary/')

        arguments = parser.parse(sys.argv[1:])
        if "-treat-AMU" in arguments:
            amu_processing(arguments['-AMU'])
        if "-treat-vanderbilt" in arguments:
            vanderbilt_processing(arguments['-vanderbilt'])