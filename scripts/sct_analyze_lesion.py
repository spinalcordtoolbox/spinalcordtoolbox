#!/usr/bin/env python

# Analyze lesions
#
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Charley
# Modified: 2017-08-19
#
# About the license: see the file LICENSE.TXT

from __future__ import print_function, absolute_import, division

import os, math, sys, pickle, shutil
import argparse

import numpy as np
import pandas as pd
from skimage.measure import label

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline
from spinalcordtoolbox.utils import Metavar, SmartFormatter, ActionCreateFolder

import sct_utils as sct
from sct_utils import extract_fname, printv, tmp_create


def get_parser():
    # Initialize the parser

    parser = argparse.ArgumentParser(
        description='R|Compute statistics on segmented lesions. The function assigns an ID value to each lesion (1, 2, '
                    '3, etc.) and then outputs morphometric measures for each lesion:\n'
                    '- volume [mm^3]\n'
                    '- length [mm]: length along the Superior-Inferior axis\n'
                    '- max_equivalent_diameter [mm]: maximum diameter of the lesion, when approximating\n'
                    '                                the lesion as a circle in the axial plane.\n\n'
                    'If the proportion of lesion in each region (e.g. WM and GM) does not sum up to 100%, it means '
                    'that the registered template does not fully cover the lesion. In that case you might want to '
                    'check the registration results.',
        add_help=None,
        formatter_class=SmartFormatter,
        prog=os.path.basename(__file__).strip(".py")
    )

    mandatory_arguments = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory_arguments.add_argument(
        "-m",
        required=True,
        help='Binary mask of lesions (lesions are labeled as "1").',
        metavar=Metavar.file,
    )
    mandatory_arguments.add_argument(
        "-s",
        required=True,
        help="Spinal cord centerline or segmentation file, which will be used to correct morphometric measures with "
             "cord angle with respect to slice. (e.g.'t2_seg.nii.gz')",
        metavar=Metavar.file)

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="show this help message and exit")
    optional.add_argument(
        "-i",
        help='Image from which to extract average values within lesions (e.g. "t2.nii.gz"). If provided, the function '
             'computes the mean and standard deviation values of this image within each lesion.',
        metavar=Metavar.file,
        default=None,
        required=False)
    optional.add_argument(
        "-f",
        help="Path to folder containing the atlas/template registered to the anatomical image. If provided, the "
             "function computes: (i) the distribution of each lesion depending on each vertebral level and on each"
             "region of the template (e.g. GM, WM, WM tracts) and (ii) the proportion of ROI (e.g. vertebral level, "
             "GM, WM) occupied by lesion.",
        metavar=Metavar.str,
        default=None,
        required=False)
    optional.add_argument(
        "-ofolder",
        help='Output folder (e.g. "./")',
        metavar=Metavar.folder,
        action=ActionCreateFolder,
        default='./',
        required=False)
    optional.add_argument(
        "-r",
        type=int,
        help="Remove temporary files.",
        required=False,
        default=1,
        choices=(0, 1))
    optional.add_argument(
        "-v",
        type=int,
        help="Verbose: 0 = nothing, 1 = classic, 2 = expended",
        required=False,
        choices=(0, 1, 2),
        default=1)

    return parser


class AnalyzeLeion:
    def __init__(self, fname_mask, fname_sc, fname_ref, path_template, path_ofolder, verbose):
        self.fname_mask = fname_mask

        self.fname_sc = fname_sc
        self.fname_ref = fname_ref
        self.path_template = path_template
        self.path_ofolder = path_ofolder
        self.verbose = verbose
        self.wrk_dir = os.getcwd()

        if not set(np.unique(Image(fname_mask).data)) == set([0.0, 1.0]):
            if set(np.unique(Image(fname_mask).data)) == set([0.0]):
                printv('WARNING: Empty masked image', self.verbose, 'warning')
            else:
                printv("ERROR input file %s is not binary file with 0 and 1 values" % fname_mask, 1, 'error')

        # create tmp directory
        self.tmp_dir = tmp_create(verbose=verbose)  # path to tmp directory

        # lesion file where each lesion has a different value
        self.fname_label = extract_fname(self.fname_mask)[1] + '_label' + extract_fname(self.fname_mask)[2]

        # initialization of measure sheet
        measure_lst = ['label', 'volume [mm3]', 'length [mm]', 'max_equivalent_diameter [mm]']
        if self.fname_ref is not None:
            for measure in ['mean', 'std']:
                measure_lst.append(measure + '_' + extract_fname(self.fname_ref)[1])
        measure_dct = {}
        for column in measure_lst:
            measure_dct[column] = None
        self.measure_pd = pd.DataFrame(data=measure_dct, index=range(0), columns=measure_lst)

        # orientation of the input image
        self.orientation = None

        # volume object
        self.volumes = None

        # initialization of proportion measures, related to registrated atlas
        if self.path_template is not None:
            self.path_atlas = os.path.join(self.path_template, "atlas")
            self.path_levels = os.path.join(self.path_template, "template", "PAM50_levels.nii.gz")
        else:
            self.path_atlas, self.path_levels = None, None
        self.vert_lst = None
        self.atlas_roi_lst = None
        self.distrib_matrix_dct = {}

        # output names
        self.pickle_name = extract_fname(self.fname_mask)[1] + '_analyzis.pkl'
        self.excel_name = extract_fname(self.fname_mask)[1] + '_analyzis.xls'

    def analyze(self):
        self.ifolder2tmp()

        # Orient input image(s) to RPI
        self.orient2rpi()

        # Label connected regions of the masked image
        self.label_lesion()

        # Compute angle for CSA correction
        self.angle_correction()

        # Compute lesion volume, equivalent diameter, (S-I) length, max axial nominal diameter
        # if registered template provided: across vertebral level, GM, WM, within WM/GM tracts...
        # if ref image is provided: Compute mean and std value in each labeled lesion
        self.measure()

        # reorient data if needed
        self.reorient()

        # print averaged results
        self.show_total_results()

        # save results in excel and pickle files
        self.pack_measures()

        # save results to ofolder
        self.tmp2ofolder()

    def tmp2ofolder(self):
        os.chdir(self.wrk_dir)  # go back to working directory

        printv('\nSave results files...', self.verbose, 'normal')
        printv('\n... measures saved in the files:', self.verbose, 'normal')
        for file_ in [self.fname_label, self.excel_name, self.pickle_name]:
            printv('\n  - ' + os.path.join(self.path_ofolder, file_), self.verbose, 'normal')
            sct.copy(os.path.join(self.tmp_dir, file_), os.path.join(self.path_ofolder, file_))

    def pack_measures(self):
        writer = pd.ExcelWriter(self.excel_name, engine='xlwt')
        self.measure_pd.to_excel(writer, sheet_name='measures', index=False, engine='xlwt')

        # Add the total column and row
        if self.path_template is not None:
            for sheet_name in self.distrib_matrix_dct:
                if '#' in sheet_name:
                    df = self.distrib_matrix_dct[sheet_name].copy()
                    df = df.append(df.sum(numeric_only=True, axis=0), ignore_index=True)
                    df['total'] = df.sum(numeric_only=True, axis=1)
                    df.iloc[-1, df.columns.get_loc('vert')] = 'total'
                    df.to_excel(writer, sheet_name=sheet_name, index=False, engine='xlwt')
                else:
                    self.distrib_matrix_dct[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False, engine='xlwt')

        # Save pickle
        self.distrib_matrix_dct['measures'] = self.measure_pd
        with open(self.pickle_name, 'wb') as handle:
            pickle.dump(self.distrib_matrix_dct, handle)

        # Save Excel
        writer.save()

    def show_total_results(self):
        printv('\n\nAveraged measures...', self.verbose, 'normal')
        for stg, key in zip(['  Volume [mm^3] = ', '  (S-I) Length [mm] = ', '  Equivalent Diameter [mm] = '], ['volume [mm3]', 'length [mm]', 'max_equivalent_diameter [mm]']):
            printv(stg + str(np.round(np.mean(self.measure_pd[key]), 2)) + '+/-' + str(np.round(np.std(self.measure_pd[key]), 2)), self.verbose, type='info')

        printv('\nTotal volume = ' + str(np.round(np.sum(self.measure_pd['volume [mm3]']), 2)) + ' mm^3', self.verbose, 'info')
        printv('Lesion count = ' + str(len(self.measure_pd['volume [mm3]'].values)), self.verbose, 'info')

    def reorient(self):
        if not self.orientation == 'RPI':
            printv('\nOrient output image to initial orientation...', self.verbose, 'normal')
            self._orient(self.fname_label, self.orientation)

    def _measure_within_im(self, im_lesion, im_ref, label_lst):
        printv('\nCompute reference image features...', self.verbose, 'normal')

        for lesion_label in label_lst:
            im_label_data_cur = im_lesion == lesion_label
            im_label_data_cur[np.where(im_ref == 0)] = 0  # if the ref object is eroded compared to the labeled object
            mean_cur, std_cur = np.mean(im_ref[np.where(im_label_data_cur)]), np.std(im_ref[np.where(im_label_data_cur)])

            label_idx = self.measure_pd[self.measure_pd.label == lesion_label].index
            self.measure_pd.loc[label_idx, 'mean_' + extract_fname(self.fname_ref)[1]] = mean_cur
            self.measure_pd.loc[label_idx, 'std_' + extract_fname(self.fname_ref)[1]] = std_cur
            printv('Mean+/-std of lesion #' + str(lesion_label) + ' in ' + extract_fname(self.fname_ref)[1] + ' file: ' + str(np.round(mean_cur, 2)) + '+/-' + str(np.round(std_cur, 2)), self.verbose, type='info')

    def _measure_volume(self, im_data, p_lst, idx):
        for zz in range(im_data.shape[2]):
            self.volumes[zz, idx - 1] = np.sum(im_data[:, :, zz]) * p_lst[0] * p_lst[1] * p_lst[2]

        vol_tot_cur = np.sum(self.volumes[:, idx - 1])
        self.measure_pd.loc[idx, 'volume [mm3]'] = vol_tot_cur
        printv('  Volume : ' + str(np.round(vol_tot_cur, 2)) + ' mm^3', self.verbose, type='info')

    def _measure_length(self, im_data, p_lst, idx):
        length_cur = np.sum([np.cos(self.angles[zz]) * p_lst[2] for zz in np.unique(np.where(im_data)[2])])
        self.measure_pd.loc[idx, 'length [mm]'] = length_cur
        printv('  (S-I) length : ' + str(np.round(length_cur, 2)) + ' mm', self.verbose, type='info')

    def _measure_diameter(self, im_data, p_lst, idx):
        area_lst = [np.sum(im_data[:, :, zz]) * np.cos(self.angles[zz]) * p_lst[0] * p_lst[1] for zz in range(im_data.shape[2])]
        diameter_cur = 2 * np.sqrt(max(area_lst) / (4 * np.pi))
        self.measure_pd.loc[idx, 'max_equivalent_diameter [mm]'] = diameter_cur
        printv('  Max. equivalent diameter : ' + str(np.round(diameter_cur, 2)) + ' mm', self.verbose, type='info')

    def ___pve_weighted_avg(self, im_mask_data, im_atlas_data):
        return im_mask_data * im_atlas_data

    def __relative_ROIvol_in_mask(self, im_mask_data, im_atlas_roi_data, p_lst, im_template_vert_data=None, vert_level=None):
        #
        #   Goal:
        #         This function computes the percentage of ROI occupied by binary mask
        #         --> ie volume of the intersection between {im_mask and im_roi} divided by the volume of roi
        #         If im_template_vert and vert are specified, the ROI is restricted to the given vert_level
        #         The PVE is handled by the method 'weighted_average'
        #
        #   Inputs:
        #           - im_mask_data - type=NumPyArray - binary mask (eg lesions)
        #           - im_atlas_roi_data - type=NumPyArray - ROI in the same space as im_mask
        #                        - p_lst - type=list of float
        #           - im_template_vert_data - type=NumPyArray - vertebral template in the same space as im_mask
        #           - vert_level - type=int - vertebral level ID to restrict the ROI
        #

        if im_template_vert_data is not None and vert_level is not None:
            im_atlas_roi_data[np.where(im_template_vert_data != vert_level)] = 0.0
            im_mask_data[np.where(im_template_vert_data != vert_level)] = 0.0

        im_mask_roi_data_wa = self.___pve_weighted_avg(im_mask_data=im_mask_data, im_atlas_data=im_atlas_roi_data)
        vol_tot_roi = np.sum(im_atlas_roi_data) * p_lst[0] * p_lst[1] * p_lst[2]
        vol_mask_roi_wa = np.sum(im_mask_roi_data_wa) * p_lst[0] * p_lst[1] * p_lst[2]

        return vol_mask_roi_wa, vol_tot_roi

    def _measure_eachLesion_distribution(self, lesion_id, atlas_data, im_vert, im_lesion, p_lst):
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

    def __regroup_per_tracts(self, vol_dct, tract_limit):
        res_mask = [vol_dct[t][0] for t in vol_dct if t >= tract_limit[0] and t <= tract_limit[1]]
        res_tot = [vol_dct[t][1] for t in vol_dct if t >= tract_limit[0] and t <= tract_limit[1]]
        return np.sum(res_mask) * 100.0 / np.sum(res_tot)

    def _measure_totLesion_distribution(self, im_lesion, atlas_data, im_vert, p_lst):

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

    def measure(self):
        im_lesion = Image(self.fname_label)
        im_lesion_data = im_lesion.data
        p_lst = im_lesion.dim[4:7] # voxel size

        label_lst = [l for l in np.unique(im_lesion_data) if l]  # lesion label IDs list

        if self.path_template is not None:
            if os.path.isfile(self.path_levels):
                img_vert = Image(self.path_levels)
                im_vert_data = img_vert.data
                self.vert_lst = [v for v in np.unique(im_vert_data) if v]  # list of vertebral levels available in the input image

            else:
                im_vert_data = None
                printv('ERROR: the file ' + self.path_levels + ' does not exist. Please make sure the template was correctly registered and warped (sct_register_to_template or sct_register_multimodal and sct_warp_template)', type='error')

            # In order to open atlas images only one time
            atlas_data_dct = {}  # dict containing the np.array of the registrated atlas
            for fname_atlas_roi in self.atlas_roi_lst:
                tract_id = int(fname_atlas_roi.split('_')[-1].split('.nii.gz')[0])
                img_cur = Image(fname_atlas_roi)
                img_cur_copy = img_cur.copy()
                atlas_data_dct[tract_id] = img_cur_copy.data
                del img_cur

        self.volumes = np.zeros((im_lesion.dim[2], len(label_lst)))

        # iteration across each lesion to measure statistics
        for lesion_label in label_lst:
            im_lesion_data_cur = np.copy(im_lesion_data == lesion_label)
            printv('\nMeasures on lesion #' + str(lesion_label) + '...', self.verbose, 'normal')

            label_idx = self.measure_pd[self.measure_pd.label == lesion_label].index
            self._measure_volume(im_lesion_data_cur, p_lst, label_idx)
            self._measure_length(im_lesion_data_cur, p_lst, label_idx)
            self._measure_diameter(im_lesion_data_cur, p_lst, label_idx)

            # compute lesion distribution for each lesion
            if self.path_template is not None:
                self._measure_eachLesion_distribution(lesion_id=lesion_label,
                                                      atlas_data=atlas_data_dct,
                                                      im_vert=im_vert_data,
                                                      im_lesion=im_lesion_data_cur,
                                                      p_lst=p_lst)

        if self.path_template is not None:
            # compute total lesion distribution
            self._measure_totLesion_distribution(im_lesion=np.copy(im_lesion_data > 0),
                                                 atlas_data=atlas_data_dct,
                                                 im_vert=im_vert_data,
                                                 p_lst=p_lst)

        if self.fname_ref is not None:
            # Compute mean and std value in each labeled lesion
            self._measure_within_im(im_lesion=im_lesion_data, im_ref=Image(self.fname_ref).data, label_lst=label_lst)

    def _normalize(self, vect):
        norm = np.linalg.norm(vect)
        return vect / norm

    def angle_correction(self):
        im_seg = Image(self.fname_sc)
        nx, ny, nz, nt, px, py, pz, pt = im_seg.dim
        data_seg = im_seg.data

        # fit centerline, smooth it and return the first derivative (in physical space)
        _, arr_ctl, arr_ctl_der, _ = get_centerline(im_seg, param=ParamCenterline(), verbose=1)
        x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = arr_ctl_der

        self.angles = np.full_like(np.empty(nz), np.nan, dtype=np.double)

        # loop across x_centerline_deriv (instead of [min_z_index, max_z_index], which could vary after interpolation)
        for iz in range(x_centerline_deriv.shape[0]):
            # normalize the tangent vector to the centerline (i.e. its derivative)
            tangent_vect = self._normalize(np.array(
                [x_centerline_deriv[iz] * px, y_centerline_deriv[iz] * py, pz]))

            # compute the angle between the normal vector of the plane and the vector z
            angle = np.arccos(np.vdot(tangent_vect, np.array([0, 0, 1])))
            self.angles[iz] = math.degrees(angle)

    def label_lesion(self):
        printv('\nLabel connected regions of the masked image...', self.verbose, 'normal')
        im = Image(self.fname_mask)
        im_2save = im.copy()
        im_2save.data = label(im.data, connectivity=2)
        im_2save.save(self.fname_label)

        self.measure_pd['label'] = [l for l in np.unique(im_2save.data) if l]
        printv('Lesion count = ' + str(len(self.measure_pd['label'])), self.verbose, 'info')

    def _orient(self, fname, orientation):
        return Image(fname).change_orientation(orientation).save(fname, mutable=True)

    def orient2rpi(self):
        # save input image orientation
        self.orientation = Image(self.fname_mask).orientation

        if not self.orientation == 'RPI':
            printv('\nOrient input image(s) to RPI orientation...', self.verbose, 'normal')
            self._orient(self.fname_mask, 'RPI')

            if self.fname_sc is not None:
                self._orient(self.fname_sc, 'RPI')
            if self.fname_ref is not None:
                self._orient(self.fname_ref, 'RPI')
            if self.path_template is not None:
                self._orient(self.path_levels, 'RPI')
                for fname_atlas in self.atlas_roi_lst:
                    self._orient(fname_atlas, 'RPI')

    def ifolder2tmp(self):
        # copy input image
        if self.fname_mask is not None:
            sct.copy(self.fname_mask, self.tmp_dir)
            self.fname_mask = ''.join(extract_fname(self.fname_mask)[1:])
        else:
            printv('ERROR: No input image', self.verbose, 'error')

        # copy seg image
        if self.fname_sc is not None:
            sct.copy(self.fname_sc, self.tmp_dir)
            self.fname_sc = ''.join(extract_fname(self.fname_sc)[1:])

        # copy ref image
        if self.fname_ref is not None:
            sct.copy(self.fname_ref, self.tmp_dir)
            self.fname_ref = ''.join(extract_fname(self.fname_ref)[1:])

        # copy registered template
        if self.path_template is not None:
            sct.copy(self.path_levels, self.tmp_dir)
            self.path_levels = ''.join(extract_fname(self.path_levels)[1:])

            self.atlas_roi_lst = []
            for fname_atlas_roi in os.listdir(self.path_atlas):
                if fname_atlas_roi.endswith('.nii.gz'):
                    tract_id = int(fname_atlas_roi.split('_')[-1].split('.nii.gz')[0])
                    if tract_id < 36:  # Not interested in CSF
                        sct.copy(os.path.join(self.path_atlas, fname_atlas_roi), self.tmp_dir)
                        self.atlas_roi_lst.append(fname_atlas_roi)

        os.chdir(self.tmp_dir)  # go to tmp directory


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

    fname_mask = arguments.m
    fname_sc = arguments.s
    fname_ref = arguments.i

    # Path to template
    path_template = arguments.f
    # TODO: check this in the parser
    # if not os.path.isdir(path_template) and os.path.exists(path_template):
    #     path_template = None
    #     printv("ERROR output directory %s is not a valid directory" % path_template, 1, 'error')

    # Output Folder
    path_results = arguments.ofolder
    # if not os.path.isdir(path_results) and os.path.exists(path_results):
    #     printv("ERROR output directory %s is not a valid directory" % path_results, 1, 'error')
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    # Remove temp folder
    if arguments.r is not None:
        rm_tmp = bool(arguments.r)
    else:
        rm_tmp = True

    # Verbosity
    verbose = arguments.v
    sct.init_sct(log_level=verbose, update=True)  # Update log level

    # create the Lesion constructor
    lesion_obj = AnalyzeLeion(fname_mask=fname_mask,
                              fname_sc=fname_sc,
                              fname_ref=fname_ref,
                              path_template=path_template,
                              path_ofolder=path_results,
                              verbose=verbose)

    # run the analyze
    lesion_obj.analyze()

    # remove tmp_dir
    if rm_tmp:
        sct.rmtree(lesion_obj.tmp_dir)

    printv('\nDone! To view the labeled lesion file (one value per lesion), type:', verbose)
    if fname_ref is not None:
        printv('fsleyes ' + fname_mask + ' ' + os.path.join(path_results, lesion_obj.fname_label) + ' -cm red-yellow -a 70.0 & \n', verbose, 'info')
    else:
        printv('fsleyes ' + os.path.join(path_results, lesion_obj.fname_label) + ' -cm red-yellow -a 70.0 & \n', verbose, 'info')


if __name__ == "__main__":
    sct.init_sct()
    main()
