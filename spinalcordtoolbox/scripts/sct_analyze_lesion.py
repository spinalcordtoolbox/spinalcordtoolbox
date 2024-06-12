#!/usr/bin/env python
#
# Analyze lesions
#
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import os
import sys
import pickle
from typing import Sequence

import numpy as np
from skimage.measure import label

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, ActionCreateFolder, display_viewer_syntax
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel, LazyLoader
from spinalcordtoolbox.utils.fs import tmp_create, extract_fname, copy, rmtree

pd = LazyLoader("pd", globals(), "pandas")
plt = LazyLoader("plt", globals(), "matplotlib.pyplot")

def get_parser():
    parser = SCTArgumentParser(
        description='Compute statistics on segmented lesions. The function assigns an ID value to each lesion (1, 2, '
                    '3, etc.) and then outputs morphometric measures for each lesion:\n'
                    '- volume [mm^3]\n'
                    '- length [mm]: length along the Superior-Inferior axis\n'
                    '- max_equivalent_diameter [mm]: maximum diameter of the lesion, when approximating the lesion as '
                    'a circle in the axial plane\n'
                    '- max_axial_damage_ratio []: maximum ratio of the lesion area divided by the spinal cord area\n\n'
                    'If the proportion of lesion in each region (e.g. WM and GM) does not sum up to 100%, it means '
                    'that the registered template does not fully cover the lesion. In that case you might want to '
                    'check the registration results.'
                    '- dorsal_bridge_width [mm]: width of spared tissue dorsal to the spinal cord lesion\n'
                    '- ventral_bridge_width [mm]: width of spared tissue ventral to the spinal cord lesion\n\n'
    )

    mandatory_arguments = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory_arguments.add_argument(
        "-m",
        required=True,
        help='Binary mask of lesions (lesions are labeled as "1").',
        metavar=Metavar.file,
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="show this help message and exit")
    optional.add_argument(
        "-s",
        required=False,
        help="Spinal cord centerline or segmentation file, which will be used to correct morphometric measures with "
             "cord angle with respect to slice. (e.g. 't2_seg.nii.gz')\n"
             "If provided, then the lesion volume, length, diameter, axial damage ratio, and tissue bridges will be "
             "computed. "
             "Otherwise, if not provided, then only the lesion volume will be computed.",
        metavar=Metavar.file)
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
             "function computes:\n"
             "\n"
             "  a. for each lesion, the proportion of that lesion within each vertebral level and each region "
             "of the template (e.g. GM, WM, WM tracts). Each cell contains a percentage value representing how much of "
             "the lesion volume exists within the region indicated by the row/column (rows represent vertebral levels, "
             "columns represent ROIs). The percentage values are summed to totals in both the bottom row and the right "
             "column, and the sum of all cells is 100 (i.e. 100 percent of the lesion), found in the bottom-right.\n"
             "  b. the proportions of each ROI (e.g. vertebral level, GM, WM) occupied by lesions.\n"
             "\n"
             "These percentage values are stored in different pages of the output `lesion_analysis.xls` spreadsheet;"
             "one page for each lesion (a.) plus a final page summarizing the total ROI occupation of all lesions (b.)",
        metavar=Metavar.str,
        default=None,
        required=False)
    optional.add_argument(
        "-ofolder",
        help='Output folder (e.g. ".")',
        metavar=Metavar.folder,
        action=ActionCreateFolder,
        default='.',
        required=False)
    optional.add_argument(
        "-r",
        type=int,
        help="Remove temporary files.",
        required=False,
        default=1,
        choices=(0, 1))
    optional.add_argument(
        '-v',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode")

    return parser


class AnalyzeLesion:
    def __init__(self, fname_mask, fname_sc, fname_ref, path_template, path_ofolder, verbose):
        self.fname_mask = fname_mask

        self.fname_sc = fname_sc
        self.fname_ref = fname_ref
        self.path_template = path_template
        self.path_ofolder = path_ofolder
        self.verbose = verbose
        self.wrk_dir = os.getcwd()
        # NOTE: the tissue bridges are NOT included in self.measure_keys because we do not want to average them
        self.measure_keys = ['volume [mm3]', 'length [mm]', 'max_equivalent_diameter [mm]', 'max_axial_damage_ratio []']

        if not set(np.unique(Image(fname_mask).data)) == set([0.0, 1.0]):
            if set(np.unique(Image(fname_mask).data)) == set([0.0]):
                printv('WARNING: Empty masked image', self.verbose, 'warning')
            else:
                printv("ERROR input file %s is not binary file with 0 and 1 values" % fname_mask, 1, 'error')

        if fname_sc is not None:
            if not Image(fname_mask).dim[0:3] == Image(fname_sc).dim[0:3]:
                printv("ERROR: Lesion and spinal cord images must have the same dimensions", 1, 'error')

        # create tmp directory
        self.tmp_dir = tmp_create(basename="analyze-lesion")  # path to tmp directory

        # lesion file where each lesion has a different value
        self.fname_label = extract_fname(self.fname_mask)[1] + '_label' + extract_fname(self.fname_mask)[2]

        # initialization of measure sheet
        measure_lst = ['label'] + self.measure_keys
        if self.fname_ref is not None:
            for measure in ['mean', 'std']:
                measure_lst.append(measure + '_' + extract_fname(self.fname_ref)[1])
        measure_dct = {}
        for column in measure_lst:
            measure_dct[column] = None
        self.measure_pd = pd.DataFrame(data=measure_dct, index=range(0), columns=measure_lst)

        # orientation of the input image
        self.orientation = None

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
        self.pickle_name = extract_fname(self.fname_mask)[1] + '_analysis.pkl'
        self.excel_name = extract_fname(self.fname_mask)[1] + '_analysis.xls'
        self.tissue_bridges_png_name = extract_fname(self.fname_mask)[1] + '_tissue_bridges.png'

    def analyze(self):
        self.ifolder2tmp()

        # Orient input image(s) to RPI
        self.orient2rpi()

        # Label connected regions of the masked image
        self.label_lesion()

        # Compute angle for CSA correction if spinal cord segmentation provided
        # NB: If segmentation is not provided, then we will only compute volume, so
        #     no angle correction is needed
        if self.fname_sc is not None:
            self.angle_correction()

        # Compute lesion volume, equivalent diameter, (S-I) length, max axial nominal diameter, and tissue bridges
        # if registered template provided: across vertebral level, GM, WM, within WM/GM tracts...
        # if ref image is provided: Compute mean and std value in each labeled lesion
        self.measure()

        # reorient data to RPI if needed
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
            copy(os.path.join(self.tmp_dir, file_), os.path.join(self.path_ofolder, file_))
        # Save tissue bridges plot
        # TODO: make the plotting below optional --> move the code below under `if self.verbose == 2`?
        #  or include the figure into QC report?
        if os.path.isfile(os.path.join(self.tmp_dir, self.tissue_bridges_png_name)):
            printv('\n... tissue bridges plot saved in the file:', self.verbose, 'normal')
            printv('\n  - ' + os.path.join(self.path_ofolder, self.tissue_bridges_png_name), self.verbose, 'normal')
            copy(os.path.join(self.tmp_dir, self.tissue_bridges_png_name),
                 os.path.join(self.path_ofolder, self.tissue_bridges_png_name))

    def pack_measures(self):
        with pd.ExcelWriter(self.excel_name, engine='xlsxwriter') as writer:
            self.measure_pd.to_excel(writer, sheet_name='measures', index=False, engine='xlsxwriter')

            # Add the total column and row
            if self.path_template is not None:
                for sheet_name in self.distrib_matrix_dct:
                    if '#' in sheet_name:
                        df = self.distrib_matrix_dct[sheet_name].copy()
                        df = df.append(df.sum(numeric_only=True, axis=0), ignore_index=True)
                        df['total % (all tracts)'] = df.sum(numeric_only=True, axis=1)
                        df.iloc[-1, df.columns.get_loc('vert')] = 'total % (all vert)'
                        df.to_excel(writer, sheet_name=sheet_name, index=False, engine='xlsxwriter')
                    else:
                        self.distrib_matrix_dct[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False, engine='xlsxwriter')

            # Save pickle
            self.distrib_matrix_dct['measures'] = self.measure_pd
            with open(self.pickle_name, 'wb') as handle:
                pickle.dump(self.distrib_matrix_dct, handle)

    def show_total_results(self):
        """
        Print total results to CLI
        """

        printv('\n\nAveraged measures across all lesions...', self.verbose, 'normal')

        for key in self.measure_keys:
            mean_value = np.round(np.mean(self.measure_pd[key]), 2)
            std_value = np.round(np.std(self.measure_pd[key]), 2)
            measure_info = f'  {key} = {mean_value} +/- {std_value}'
            printv(measure_info, self.verbose, type='info')

        total_volume = np.round(np.sum(self.measure_pd['volume [mm3]']), 2)
        lesion_count = len(self.measure_pd['volume [mm3]'].values)

        printv('\nTotal volume = ' + str(total_volume) + ' mm^3', self.verbose, 'info')
        printv('Lesion count = ' + str(lesion_count), self.verbose, 'info')

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
            file_ref = extract_fname(self.fname_ref)[1]
            printv(
                f'Mean+/-std of lesion #{lesion_label} in {file_ref} file: {mean_cur:.2f}+/-{std_cur:.2f}',
                self.verbose,
                type='info')

    def _measure_volume(self, im_data, p_lst, idx):
        """
        Measure the volume of the lesion
        """
        volume = np.sum(im_data) * p_lst[0] * p_lst[1] * p_lst[2]
        self.measure_pd.loc[idx, 'volume [mm3]'] = volume
        printv(f'  Volume : {round(volume, 2)} mm^3', self.verbose, type='info')

    def _measure_axial_damage_ratio(self, im_data, p_lst, idx):
        """
        Measure the maximum axial damage ratio
        The axial damage ratio is calculated as the ratio of lesion area divided by spinal cord area
        The axial damage ratio is calculated for each slice and then the maximum value is retained
        REF: Smith, A. C. et al. (2021). Axial MRI biomarkers of spinal cord damage to predict future walking and motor
        function: A retrospective study. Spinal Cord, 59(6), 693-699. https://doi.org/10.1038/s41393-020-00561-w

        :param im_data: 3D numpy array, binary mask of the lesion
        :param p_lst: list, pixel size of the lesion
        :param idx: int, index of the lesion
        """

        # Load the spinal cord image
        im_sc = Image(self.fname_sc)
        im_sc_data = im_sc.data
        p_lst_sc = im_sc.dim[4:7]   # voxel size

        axial_damage_ratio_dict = {}
        # Get axial slices with lesion
        lesion_slices = np.unique(np.where(im_data)[2])
        for slice in lesion_slices:
            # Lesion area
            lesion_area = np.sum(im_data[:, :, slice]) * p_lst[0] * p_lst[1]
            # Spinal cord area
            sc_area = np.sum(im_sc_data[:, :, slice]) * p_lst_sc[0] * p_lst_sc[1]
            # Compute the axial damage ratio slice by slice
            axial_damage_ratio_dict[slice] = lesion_area / sc_area

        # Get the maximum axial damage ratio
        maximum_axial_damage_ratio = np.max(list(axial_damage_ratio_dict.values()))

        # Save the maximum axial damage ratio
        self.measure_pd.loc[idx, 'max_axial_damage_ratio []'] = maximum_axial_damage_ratio
        printv('  Maximum axial damage ratio : ' + str(np.round(maximum_axial_damage_ratio, 2)),
               self.verbose, type='info')

    def _measure_tissue_bridges(self, im_lesion_data, p_lst, idx):
        """
        Measure the tissue bridges (widths of spared tissue ventral and dorsal to the spinal cord lesion).
        Tissue bridges are quantified as the width of spared tissue at the **minimum** distance from cerebrospinal fluid
        (i.e., the spinal cord boundary) to the lesion boundary.
        NOTE: we compute the tissue bridges for all sagittal slices containing the lesion (i.e., for the midsagittal and
        parasagittal slices).
        REF: Huber E, Lachappelle P, Sutter R, Curt A, Freund P. Are midsagittal tissue bridges predictive of outcome
        after cervical spinal cord injury? Ann Neurol. 2017 May;81(5):740-748. doi: 10.1002/ana.24932.

        :param im_lesion_data: 3D numpy array: mask of the lesion. The orientation is assumed to be RPI (because we
            reoriented the image to RPI using orient2rpi())
        :param p_lst: list, pixel size of the lesion
        :param idx: int, index of the lesion
        """

        # Load the spinal cord segmentation mask
        # The orientation is assumed to be RPI (because we reoriented the image to RPI using orient2rpi())
        im_sc = Image(self.fname_sc)
        im_sc_data = im_sc.data

        # --------------------------------------
        # Get slices with the lesion
        # --------------------------------------
        # We decided to use all sagittal slices containing the lesion to compute the tissue bridges
        # In other words, we compute the tissue bridges from the midsagittal slice and also from all parasagittal slices

        # Get slices with lesion
        sagittal_lesion_slices = np.unique(np.where(im_lesion_data)[0])     # as the orientation is RPI, [0] is the R-L direction
        if self.verbose == 2:
            printv('  Slices with lesion: ' + str(sagittal_lesion_slices), self.verbose, type='info')

        # --------------------------------------
        # Compute tissue bridges for each sagittal slice containing the lesion
        # --------------------------------------
        tissue_bridges_dict = {}
        # Loop across sagittal slices
        for sagittal_slice in sagittal_lesion_slices:
            # Get all axial slices (S-I direction) with the lesion for the selected sagittal slice
            # In other words, we will iterate through the lesion in S-I direction and compute tissue bridges for each
            # axial slice with the lesion
            axial_lesion_slices = np.unique(np.where(im_lesion_data[sagittal_slice, :, :])[1])     # as the orientation is RPI, [1] is the S-I direction
            # Iterate across axial slices to compute tissue bridges
            for axial_slice in axial_lesion_slices:
                # Get the lesion segmentation mask of the selected 2D axial slice
                slice_lesion_data = im_lesion_data[sagittal_slice, :, axial_slice]
                # Get the spinal cord segmentation mask of the selected 2D axial slice
                slice_sc_data = im_sc_data[sagittal_slice, :, axial_slice]
                # Get the indices of the lesion mask for the selected axial slice
                lesion_indices = np.where(slice_lesion_data)[0]
                # Get the indices of the spinal cord mask for the selected axial slice
                sc_indices = np.where(slice_sc_data)[0]

                # Compute ventral and dorsal tissue bridges
                dorsal_bridge_width = lesion_indices[0] - sc_indices[0]         # [0] returns the most dorsal elements
                # if the lesion extends the spinal cord, the dorsal bridge is set to 0
                if dorsal_bridge_width < 0:
                    dorsal_bridge_width = 0
                ventral_bridge_width = sc_indices[-1] - lesion_indices[-1]      # [-1] returns the most ventral elements
                # if the lesion extends the spinal cord, the ventral bridge is set to 0
                if ventral_bridge_width < 0:
                    ventral_bridge_width = 0

                tissue_bridges_dict[sagittal_slice, axial_slice] = \
                    {'dorsal_bridge_width': dorsal_bridge_width,
                     'ventral_bridge_width': ventral_bridge_width,
                     'lesion_indices': lesion_indices}       # we store lesion indices for plotting

        # --------------------------------------
        # Get minimal tissue bridges
        # --------------------------------------
        # Convert the dictionary to a DataFrame (for easier manipulation)
        # 1. Create a MultiIndex from the dictionary keys
        index = pd.MultiIndex.from_tuples(tissue_bridges_dict.keys(), names=['sagittal_slice', 'axial_slice'])
        # 2. Create the DataFrame using the MultiIndex and the dictionary values
        tissue_bridges_df = pd.DataFrame(list(tissue_bridges_dict.values()), index=index)
        # 3. Reset the index to make 'sagittal_slice' and 'axial_slice' as columns
        tissue_bridges_df.reset_index(inplace=True)

        # Get slices of minimum dorsal and ventral tissue bridges for each sagittal slice
        # NOTE: we get minimum because tissue bridges are quantified as the width of spared tissue at the minimum
        # distance from cerebrospinal fluid to the lesion boundary
        for sagittal_slice in sagittal_lesion_slices:
            # Get df for the selected sagittal slice
            df_temp = tissue_bridges_df[tissue_bridges_df['sagittal_slice'] == sagittal_slice]
            # Get the axial slice with the minimum dorsal tissue bridge for the selected sagittal slice
            min_dorsal_bridge_width_slice = df_temp.loc[df_temp['dorsal_bridge_width'].idxmin(), 'axial_slice']
            # Get the axial slice with the minimum ventral tissue bridge for the selected sagittal slice
            min_ventral_bridge_width_slice = df_temp.loc[df_temp['ventral_bridge_width'].idxmin(), 'axial_slice']

            # Add a new column with value True to tissue_bridges_df; this information is needed for plotting
            tissue_bridges_df.loc[(tissue_bridges_df['sagittal_slice'] == sagittal_slice) &
                                  (tissue_bridges_df['axial_slice'] == min_dorsal_bridge_width_slice), 'min_dorsal_bridge_axial_slice'] = True
            tissue_bridges_df.loc[(tissue_bridges_df['sagittal_slice'] == sagittal_slice) &
                                  (tissue_bridges_df['axial_slice'] == min_ventral_bridge_width_slice), 'min_ventral_bridge_axial_slice'] = True
            # Replace NaN with False
            tissue_bridges_df.fillna(False, inplace=True)

            # Get the width of the tissue bridges in mm
            # NOTE: the orientation is RPI (because we reoriented the image to RPI using orient2rpi()); therefore
            # p_lst[0] is the pixel size in the R-L direction, p_lst[1] is the pixel size in the A-P direction, and
            # p_lst[2] is the pixel size in the S-I direction.
            # Since we are computing dorsal and ventral tissue bridges, we use p_lst[1] (A-P direction)
            dorsal_bridge_width_mm = float(df_temp[df_temp['axial_slice'] ==
                                                   min_dorsal_bridge_width_slice]['dorsal_bridge_width'] * p_lst[1])
            ventral_bridge_width_mm = float(df_temp[df_temp['axial_slice'] ==
                                                    min_ventral_bridge_width_slice]['ventral_bridge_width'] * p_lst[1])

            # Save the minimum tissue bridges
            self.measure_pd.loc[idx, f'slice_{str(sagittal_slice)}_dorsal_bridge_width [mm]'] = dorsal_bridge_width_mm
            self.measure_pd.loc[idx, f'slice_{str(sagittal_slice)}_ventral_bridge_width [mm]'] = ventral_bridge_width_mm
            printv(f'  Sagittal slice {sagittal_slice}, Minimum dorsal tissue bridge width: '
                   f'{np.round(dorsal_bridge_width_mm, 2)} mm (axial slice {min_dorsal_bridge_width_slice})',
                   self.verbose, type='info')
            printv(f'  Sagittal slice {sagittal_slice}, Minimum ventral tissue bridge width: '
                   f'{np.round(ventral_bridge_width_mm, 2)} mm (axial slice {min_ventral_bridge_width_slice})',
                   self.verbose, type='info')

        # --------------------------------------
        # Plot all sagittal slices with lesions using matplotlib and save it as png
        # --------------------------------------
        # TODO: make the plotting below optional --> move the code below under `if self.verbose == 2`?
        #  or include the figure into QC report?

        #  Create a figure with num of subplots equal to the num of sagittal slices containing the lesion
        fig, axes = plt.subplots(1, len(sagittal_lesion_slices), figsize=(len(sagittal_lesion_slices)*5, 5))
        # Flatten 2D array into 1D to allow iteration by loop
        axs = axes.ravel()
        # Loop across subplots (one subplot per sagittal slice)
        for index in range(0, len(axs)):
            # Get sagittal slice
            sagittal_slice = sagittal_lesion_slices[index]
            # Get spinal cord and lesion masks data for the selected sagittal slice
            im_sc_mid_sagittal = im_sc_data[sagittal_slice]
            im_mid_sagittal = im_lesion_data[sagittal_slice]

            # Plot spinal cord and lesion masks
            axs[index].imshow(np.swapaxes(im_sc_mid_sagittal, 1, 0), cmap='gray', origin="lower")
            axs[index].imshow(np.swapaxes(im_mid_sagittal, 1, 0), cmap='jet', alpha=0.8,
                              interpolation='nearest', origin="lower")

            # Crop around the lesion
            axs[index].set_xlim(np.min(np.where(im_lesion_data)[1]) - 20, np.max(np.where(im_lesion_data)[1]) + 20)
            axs[index].set_ylim(np.min(np.where(im_lesion_data)[2]) - 20, np.max(np.where(im_lesion_data)[2]) + 20)

            # --------------------------------------
            # Add horizontal lines for the tissue bridges
            # --------------------------------------
            # x1_dorsal is ndarray: the indices of the lesion mask
            # Note: we use [0] because .values returns a numpy array
            x1_dorsal = tissue_bridges_df[(tissue_bridges_df['sagittal_slice'] == sagittal_slice) &
                                          (tissue_bridges_df['min_dorsal_bridge_axial_slice'])]['lesion_indices'].values[0]
            # x2_dorsal is int: the width of the tissue bridge
            x2_dorsal = tissue_bridges_df[(tissue_bridges_df['sagittal_slice'] == sagittal_slice) &
                                          (tissue_bridges_df['min_dorsal_bridge_axial_slice'])]['dorsal_bridge_width'].values[0]
            # y_dorsal is int: the axial slice with the minimum dorsal tissue bridge width
            y_dorsal = tissue_bridges_df[(tissue_bridges_df['sagittal_slice'] == sagittal_slice) &
                                         (tissue_bridges_df['min_dorsal_bridge_axial_slice'])]['axial_slice'].values[0]
            axs[index].plot([x1_dorsal[0] - 1, x1_dorsal[0] - x2_dorsal], [y_dorsal] * 2,  'r--', linewidth=1)

            # x1_ventral is ndarray: the indices of the lesion mask
            # Note: we use [0] because .values returns a numpy array
            x1_ventral = tissue_bridges_df[(tissue_bridges_df['sagittal_slice'] == sagittal_slice) &
                                           (tissue_bridges_df['min_ventral_bridge_axial_slice'])]['lesion_indices'].values[0]
            # x2_ventral is int: the width of the tissue bridge
            x2_ventral = tissue_bridges_df[(tissue_bridges_df['sagittal_slice'] == sagittal_slice) &
                                           (tissue_bridges_df['min_ventral_bridge_axial_slice'])]['ventral_bridge_width'].values[0]
            # y_dorsal is int: the axial slice with the minimum dorsal tissue bridge width
            y_ventral = tissue_bridges_df[(tissue_bridges_df['sagittal_slice'] == sagittal_slice) &
                                         (tissue_bridges_df['min_ventral_bridge_axial_slice'])]['axial_slice'].values[0]
            axs[index].plot([x1_ventral[-1] + 1, x1_ventral[-1] + x2_ventral], [y_ventral] * 2,  'r--', linewidth=1)

            # --------------------------------------
            # Add text with the width of the tissue bridges above the tissue bridges
            # --------------------------------------
            dorsal_bridge_width_mm = float(x2_dorsal * p_lst[1])
            axs[index].text(x1_dorsal[0] - x2_dorsal / 2,
                            y_dorsal + 1,
                            f'{np.round(dorsal_bridge_width_mm, 2)} mm',
                            color='red', fontsize=12, ha='right', va='bottom')
            ventral_bridge_width_mm = float(x2_ventral * p_lst[1])
            axs[index].text(x1_ventral[-1] + x2_ventral / 2,
                            y_ventral + 1,
                            f'{np.round(ventral_bridge_width_mm, 2)} mm',
                            color='red', fontsize=12, ha='left', va='bottom')

            axs[index].set_title(f'Sagittal slice #{sagittal_slice}')
            axs[index].set_ylabel('Inferior-Superior')
            axs[index].set_xlabel('Posterior-Anterior')

        plt.savefig(self.tissue_bridges_png_name)
        plt.close()

    def _measure_length(self, im_data, p_lst, idx):
        """
        Measure the length of the lesion along the superior-inferior axis when taking into account the angle correction
        """
        length_cur = np.sum([p_lst[2] / np.cos(self.angles[zz]) for zz in np.unique(np.where(im_data)[2])])
        self.measure_pd.loc[idx, 'length [mm]'] = length_cur
        printv('  (S-I) length : ' + str(np.round(length_cur, 2)) + ' mm', self.verbose, type='info')

    def _measure_diameter(self, im_data, p_lst, idx):
        """
        Measure the max. equivalent diameter of the lesion when taking into account the angle correction
        """
        area_lst = [np.sum(im_data[:, :, zz]) * np.cos(self.angles[zz]) * p_lst[0] * p_lst[1] for zz in range(im_data.shape[2])]
        diameter_cur = 2 * np.sqrt(max(area_lst) / np.pi)
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
        self.distrib_matrix_dct[sheet_name] = pd.DataFrame.from_dict({'vert': [str(v) for v in self.vert_lst] + ['total % (all vert)']})

        # initialized to 0 for each vertebral level and each PAM50 tract
        for tract_id in atlas_data:
            self.distrib_matrix_dct[sheet_name]['PAM50_' + str(tract_id).zfill(2)] = [0] * len(self.vert_lst + ['total % (all vert)'])

        for vert in self.vert_lst + ['total % (all vert)']:  # loop over the vertebral levels
            if vert != 'total % (all vert)':
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
        p_lst = im_lesion.dim[4:7]  # voxel size

        label_lst = [label for label in np.unique(im_lesion_data) if label]  # lesion label IDs list

        if self.path_template is not None:
            if os.path.isfile(self.path_levels):
                img_vert = Image(self.path_levels)
                im_vert_data = img_vert.data
                self.vert_lst = [v for v in np.unique(im_vert_data) if v]  # list of vertebral levels available in the input image

            else:
                im_vert_data = None
                printv(
                    f"ERROR: the file {self.path_levels} does not exist. "
                    f"Please make sure the template was correctly registered and warped "
                    f"(sct_register_to_template or sct_register_multimodal and sct_warp_template)",
                    type='error',
                )

            # In order to open atlas images only one time
            atlas_data_dct = {}  # dict containing the np.array of the registrated atlas
            for fname_atlas_roi in self.atlas_roi_lst:
                tract_id = int(fname_atlas_roi.split('_')[-1].split('.nii.gz')[0])
                img_cur = Image(fname_atlas_roi)
                img_cur_copy = img_cur.copy()
                atlas_data_dct[tract_id] = img_cur_copy.data
                del img_cur

        # iteration across each lesion to measure statistics
        for lesion_label in label_lst:
            im_lesion_data_cur = np.copy(im_lesion_data == lesion_label)
            printv('\nMeasures on lesion #' + str(lesion_label) + '...', self.verbose, 'normal')

            label_idx = self.measure_pd[self.measure_pd.label == lesion_label].index
            # For the lesion length and diameter, we need the spinal cord segmentation for angle correction
            # For the axial damage ratio, we need the spinal cord segmentation to compute the ratio between lesion area
            # and spinal cord area
            # For the tissue bridges, we need the spinal cord segmentation to compute the width of spared tissue ventral
            # and dorsal to the spinal cord lesion
            if self.fname_sc is not None:
                self._measure_length(im_lesion_data_cur, p_lst, label_idx)
                self._measure_diameter(im_lesion_data_cur, p_lst, label_idx)
                self._measure_axial_damage_ratio(im_lesion_data_cur, p_lst, label_idx)
                self._measure_tissue_bridges(im_lesion_data_cur, p_lst, label_idx)
            self._measure_volume(im_lesion_data_cur, p_lst, label_idx)

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

        # fit centerline, smooth it and return the first derivative (in physical space)
        # We set minmax=False to prevent cropping and ensure that `self.angles[iz]` covers all z slices of `im_seg`
        _, arr_ctl, arr_ctl_der, _ = get_centerline(im_seg, param=ParamCenterline(minmax=False), verbose=1)
        x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = arr_ctl_der

        self.angles = np.full_like(np.empty(nz), np.nan, dtype=np.double)

        # loop across x_centerline_deriv (instead of [min_z_index, max_z_index], which could vary after interpolation)
        for iz in range(x_centerline_deriv.shape[0]):
            # normalize the tangent vector to the centerline (i.e. its derivative)
            tangent_vect = self._normalize(np.array(
                [x_centerline_deriv[iz] * px, y_centerline_deriv[iz] * py, pz]))

            # compute the angle between the normal vector of the plane and the vector z
            self.angles[iz] = np.arccos(np.vdot(tangent_vect, np.array([0, 0, 1])))

    def label_lesion(self):
        printv('\nLabel connected regions of the masked image...', self.verbose, 'normal')
        im = Image(self.fname_mask)
        im_2save = im.copy()
        im_2save.data = label(im.data, connectivity=2)
        im_2save.save(self.fname_label)

        self.measure_pd['label'] = [label for label in np.unique(im_2save.data) if label]
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
            copy(self.fname_mask, self.tmp_dir)
            self.fname_mask = ''.join(extract_fname(self.fname_mask)[1:])
        else:
            printv('ERROR: No input image', self.verbose, 'error')

        # copy seg image
        if self.fname_sc is not None:
            copy(self.fname_sc, self.tmp_dir)
            self.fname_sc = ''.join(extract_fname(self.fname_sc)[1:])

        # copy ref image
        if self.fname_ref is not None:
            copy(self.fname_ref, self.tmp_dir)
            self.fname_ref = ''.join(extract_fname(self.fname_ref)[1:])

        # copy registered template
        if self.path_template is not None:
            copy(self.path_levels, self.tmp_dir)
            self.path_levels = ''.join(extract_fname(self.path_levels)[1:])

            self.atlas_roi_lst = []
            for fname_atlas_roi in os.listdir(self.path_atlas):
                if fname_atlas_roi.endswith('.nii.gz'):
                    tract_id = int(fname_atlas_roi.split('_')[-1].split('.nii.gz')[0])
                    if tract_id < 36:  # Not interested in CSF
                        copy(os.path.join(self.path_atlas, fname_atlas_roi), self.tmp_dir)
                        self.atlas_roi_lst.append(fname_atlas_roi)

        os.chdir(self.tmp_dir)  # go to tmp directory


def main(argv: Sequence[str]):
    """
    Main function
    :param argv:
    :return:
    """
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

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

    # create the Lesion constructor
    lesion_obj = AnalyzeLesion(fname_mask=fname_mask,
                               fname_sc=fname_sc,
                               fname_ref=fname_ref,
                               path_template=path_template,
                               path_ofolder=path_results,
                               verbose=verbose)

    # run the analyze
    lesion_obj.analyze()

    # remove tmp_dir
    if rm_tmp:
        rmtree(lesion_obj.tmp_dir)

    if fname_ref is not None:
        display_viewer_syntax(
            files=[fname_mask, os.path.join(path_results, lesion_obj.fname_label)],
            im_types=['anat', 'softseg'],
            opacities=['1.0', '0.7'],
            verbose=verbose
        )
    else:
        display_viewer_syntax(
            files=[os.path.join(path_results, lesion_obj.fname_label)],
            im_types=['softseg'],
            opacities=['0.7'],
            verbose=verbose
        )


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
