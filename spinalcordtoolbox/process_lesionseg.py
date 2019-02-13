#!/usr/bin/env python
# -*- coding: utf-8
# Functions processing lesion segmentation data

from __future__ import absolute_import

import os, math

import numpy as np

import sct_utils as sct
import spinalcordtoolbox.image as msct_image

def _measure_volume(im_data, p_lst, idx, verbose):
	for zz in range(im_data.shape[2]):
		self.volumes[zz, idx - 1] = np.sum(im_data[:, :, zz]) * p_lst[0] * p_lst[1] * p_lst[2]

	vol_tot_cur = np.sum(self.volumes[:, idx - 1])
	self.measure_pd.loc[idx, 'volume [mm3]'] = vol_tot_cur
	printv('  Volume : ' + str(np.round(vol_tot_cur, 2)) + ' mm^3', verbose, type='info')

def _measure_length(im_data, p_lst, idx, verbose):
	length_cur = np.sum([np.cos(self.angles[zz]) * p_lst[2] for zz in np.unique(np.where(im_data)[2])])
	self.measure_pd.loc[idx, 'length [mm]'] = length_cur
	printv('  (S-I) length : ' + str(np.round(length_cur, 2)) + ' mm', verbose, type='info')

def _measure_diameter(im_data, p_lst, idx, verbose):
	area_lst = [np.sum(im_data[:, :, zz]) * np.cos(self.angles[zz]) * p_lst[0] * p_lst[1] for zz in range(im_data.shape[2])]
	diameter_cur = 2 * np.sqrt(max(area_lst) / (4 * np.pi))
	self.measure_pd.loc[idx, 'max_equivalent_diameter [mm]'] = diameter_cur
	printv('  Max. equivalent diameter : ' + str(np.round(diameter_cur, 2)) + ' mm', verbose, type='info')


def _measure_eachLesion_distribution(lesion_id, atlas_data, im_vert, im_lesion, p_lst):
    sheet_name = 'lesion#' + str(lesion_id) + '_distribution'
    self.distrib_matrix_dct[sheet_name] = pd.DataFrame.from_dict({'vert': [str(v) for v in self.vert_lst]})

    # initialized to 0 for each vertebral level and each PAM50 tract
    for tract_id in atlas_data:
        self.distrib_matrix_dct[sheet_name]['PAM50_' + str(tract_id).zfill(2)] = [0] * len(self.vert_lst)

    vol_mask_tot = 0.0  # vol tot of this lesion through the vertebral levels and PAM50 tracts
    for vert in self.vert_lst:  # Loop over vertebral levels
        im_vert_cur = np.copy(im_vert)
        im_vert_cur[np.where(im_vert_cur != vert)] = 0.0
        if np.count_nonzero(im_vert_cur * np.copy(im_lesion)):  # if there is lesion in this vertebral level
            idx = self.distrib_matrix_dct[sheet_name][self.distrib_matrix_dct[sheet_name].vert == str(vert)].index
            for tract_id in atlas_data:  # Loop over PAM50 tracts
                res_lst = self.__relative_ROIvol_in_mask(im_mask_data=np.copy(im_lesion),
                                                         im_atlas_roi_data=np.copy(atlas_data[tract_id]),
                                                         p_lst=p_lst,
                                                         im_template_vert_data=np.copy(im_vert_cur),
                                                         vert_level=vert)
                self.distrib_matrix_dct[sheet_name].loc[idx, 'PAM50_' + str(tract_id).zfill(2)] = res_lst[0]
                vol_mask_tot += res_lst[0]

    # convert the volume values in distrib_matrix_dct to percentage values
    for vert in self.vert_lst:
        idx = self.distrib_matrix_dct[sheet_name][self.distrib_matrix_dct[sheet_name].vert == str(vert)].index
        for tract_id in atlas_data:
            val = self.distrib_matrix_dct[sheet_name].loc[idx, 'PAM50_' + str(tract_id).zfill(2)].values[0]
            self.distrib_matrix_dct[sheet_name].loc[idx, 'PAM50_' + str(tract_id).zfill(2)] = val * 100.0 / vol_mask_tot


def _measure_totLesion_distribution(im_lesion, atlas_data, im_vert, p_lst):

    sheet_name = 'ROI_occupied_by_lesion'
    self.distrib_matrix_dct[sheet_name] = pd.DataFrame.from_dict({'vert': [str(v) for v in self.vert_lst] + ['total']})

    # initialized to 0 for each vertebral level and each PAM50 tract
    for tract_id in atlas_data:
        self.distrib_matrix_dct[sheet_name]['PAM50_' + str(tract_id).zfill(2)] = [0] * len(self.vert_lst + ['total'])

    for vert in self.vert_lst + ['total']:  # loop over the vertebral levels
        if vert != 'total':
            im_vert_cur = np.copy(im_vert)
            im_vert_cur[np.where(im_vert_cur != vert)] = 0
        else:
            im_vert_cur = None
        if im_vert_cur is None or np.count_nonzero(im_vert_cur * np.copy(im_lesion)):
            res_perTract_dct = {}  # for each tract compute the volume occupied by lesion and the volume of the tract
            idx = self.distrib_matrix_dct[sheet_name][self.distrib_matrix_dct[sheet_name].vert == str(vert)].index
            for tract_id in atlas_data:  # loop over the tracts
                res_perTract_dct[tract_id] = self.__relative_ROIvol_in_mask(im_mask_data=np.copy(im_lesion),
                                                                            im_atlas_roi_data=np.copy(atlas_data[tract_id]),
                                                                            p_lst=p_lst,
                                                                            im_template_vert_data=np.copy(im_vert_cur),
                                                                            vert_level=vert)

            # group tracts to compute involvement in GM, WM, DC, VF, LF
            self.distrib_matrix_dct[sheet_name].loc[idx, 'PAM50_GM'] = self.__regroup_per_tracts(vol_dct=res_perTract_dct, tract_limit=[30, 35])
            self.distrib_matrix_dct[sheet_name].loc[idx, 'PAM50_WM'] = self.__regroup_per_tracts(vol_dct=res_perTract_dct, tract_limit=[0, 29])
            self.distrib_matrix_dct[sheet_name].loc[idx, 'PAM50_DC'] = self.__regroup_per_tracts(vol_dct=res_perTract_dct, tract_limit=[0, 3])
            self.distrib_matrix_dct[sheet_name].loc[idx, 'PAM50_VF'] = self.__regroup_per_tracts(vol_dct=res_perTract_dct, tract_limit=[14, 29])
            self.distrib_matrix_dct[sheet_name].loc[idx, 'PAM50_LF'] = self.__regroup_per_tracts(vol_dct=res_perTract_dct, tract_limit=[4, 13])

            # save involvement in each PAM50 tracts
            for tract_id in atlas_data:
                self.distrib_matrix_dct[sheet_name].loc[idx, 'PAM50_' + str(tract_id).zfill(2)] = res_perTract_dct[tract_id][0] * 100.0 / res_perTract_dct[tract_id][1]


def _measure_within_im(im_ref, label_lst):
    printv('\nCompute reference image features...', self.verbose, 'normal')

    for lesion_label in label_lst:
        im_label_data_cur = im_lesion == lesion_label
        im_label_data_cur[np.where(im_ref == 0)] = 0  # if the ref object is eroded compared to the labeled object
        mean_cur, std_cur = np.mean(im_ref[np.where(im_label_data_cur)]), np.std(im_ref[np.where(im_label_data_cur)])

        label_idx = self.measure_pd[self.measure_pd.label == lesion_label].index
        self.measure_pd.loc[label_idx, 'mean_' + extract_fname(self.fname_ref)[1]] = mean_cur
        self.measure_pd.loc[label_idx, 'std_' + extract_fname(self.fname_ref)[1]] = std_cur
        printv('Mean+/-std of lesion #' + str(lesion_label) + ' in ' + extract_fname(self.fname_ref)[1] + ' file: ' + str(np.round(mean_cur, 2)) + '+/-' + str(np.round(std_cur, 2)), self.verbose, type='info')


def measure_lesion(im_lesion, path_template=None, im_vert=None, atlas_roi_lst=None, measure_pd=None, fname_ref=None, verbose='1'):
    im_lesion_data = im_lesion.data
    p_lst = im_lesion.dim[4:7] # voxel size

    label_lst = [l for l in np.unique(im_lesion_data) if l]  # lesion label IDs list

    if path_template is not None:
        if img_vert is not None:
            im_vert_data = img_vert.data
            vert_lst = [v for v in np.unique(im_vert_data) if v]  # list of vertebral levels available in the input image

        else:
            im_vert_data = None
            printv('ERROR: the template file with vertebral labelling does not exist. Please make sure the template was correctly registered and warped (sct_register_to_template or sct_register_multimodal and sct_warp_template)', type='error')

        # In order to open atlas images only one time
        atlas_data_dct = {}  # dict containing the np.array of the registrated atlas
        for fname_atlas_roi in atlas_roi_lst:
            tract_id = int(fname_atlas_roi.split('_')[-1].split('.nii.gz')[0])
            img_cur = Image(fname_atlas_roi)
            img_cur_copy = img_cur.copy()
            atlas_data_dct[tract_id] = img_cur_copy.data
            del img_cur

    volumes = np.zeros((im_lesion.dim[2], len(label_lst)))

    # iteration across each lesion to measure statistics
    for lesion_label in label_lst:
        im_lesion_data_cur = np.copy(im_lesion_data == lesion_label)
        printv('\nMeasures on lesion #' + str(lesion_label) + '...', verbose, 'normal')

        label_idx = measure_pd[measure_pd.label == lesion_label].index
        _measure_volume(im_lesion_data_cur, p_lst, label_idx, verbose)
        _measure_length(im_lesion_data_cur, p_lst, label_idx, verbose)
        _measure_diameter(im_lesion_data_cur, p_lst, label_idx, verbose)

        # compute lesion distribution for each lesion
        if path_template is not None:
            _measure_eachLesion_distribution(lesion_id=lesion_label,
                                                  atlas_data=atlas_data_dct,
                                                  im_vert=im_vert_data,
                                                  im_lesion=im_lesion_data_cur,
                                                  p_lst=p_lst)

    if path_template is not None:
        # compute total lesion distribution
        _measure_totLesion_distribution(im_lesion=np.copy(im_lesion_data > 0),
                                             atlas_data=atlas_data_dct,
                                             im_vert=im_vert_data,
                                             p_lst=p_lst)

    if fname_ref is not None:
        # Compute mean and std value in each labeled lesion
        _measure_within_im(im_lesion=im_lesion_data, im_ref=Image(self.fname_ref).data, label_lst=label_lst)

    return volumes, measure_pd
