#!/usr/bin/env python
#
# Analyze lesions
#
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import os
import sys
import pickle
import warnings
from typing import Sequence

import numpy as np
from skimage.measure import label

from spinalcordtoolbox.image import Image, rpi_slice_to_orig_orientation
from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline
from spinalcordtoolbox.metadata import read_label_file
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, ActionCreateFolder, display_viewer_syntax
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel, LazyLoader, sct_progress_bar
from spinalcordtoolbox.utils.fs import tmp_create, extract_fname, copy, rmtree
from spinalcordtoolbox.reports.qc2 import sct_analyze_lesion

pd = LazyLoader("pd", globals(), "pandas")


def get_parser():
    parser = SCTArgumentParser(
        description='Compute statistics on segmented lesions. The function assigns an ID value to each lesion (1, 2, '
                    '3, etc.) and then outputs morphometric measures for each lesion:\n'
                    '- volume [mm^3]\n'
                    '- length [mm]: length along the Superior-Inferior axis\n'
                    '- max_equivalent_diameter [mm]: maximum diameter of the lesion, when approximating the lesion as '
                    'a circle in the axial plane\n'
                    '- max_axial_damage_ratio []: maximum ratio of the lesion area divided by the spinal cord area\n'
                    '- midsagittal_spinal_cord_slice: midsagittal slice number of the spinal cord\n'
                    '- length_midsagittal_slice [mm]: length along the Superior-Inferior axis in the **midsagittal '
                    'slice**\n'
                    '- width_midsagittal_slice [mm]: width along the Anterior-Posterior axis the **midsagittal '
                    'slice**\n'
                    '- dorsal_bridge_width [mm]: width of spared tissue dorsal to the spinal cord lesion '
                    '(i.e. towards the posterior direction of the AP axis)\n'
                    '- ventral_bridge_width [mm]: width of spared tissue ventral to the spinal cord lesion '
                    '(i.e. towards the anterior direction of the AP axis)\n\n'
                    'If the proportion of lesion in each region (e.g. WM and GM) does not sum up to 100%, it means '
                    'that the registered template does not fully cover the lesion. In that case you might want to '
                    'check the registration results.'
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
        "-perslice",
        help="Specify whether to aggregate atlas metrics ('-f' option) per slice (`-perslice 1`) or per vertebral "
             "level (default behavior).",
        metavar=Metavar.int,
        type=int,
        choices=(0, 1),
        default=0,
        required=False
    )
    optional.add_argument(
        "-ofolder",
        help='Output folder (e.g. "."). Default is the current folder (".").',
        metavar=Metavar.folder,
        action=ActionCreateFolder,
        default='.',
        required=False)
    optional.add_argument(
        '-qc',
        metavar=Metavar.folder,
        action=ActionCreateFolder,
        help="The path where the quality control generated content will be saved."
    )
    optional.add_argument(
        '-qc-dataset',
        metavar=Metavar.str,
        help="If provided, this string will be mentioned in the QC report as the dataset the process was run on."
    )
    optional.add_argument(
        '-qc-subject',
        metavar=Metavar.str,
        help="If provided, this string will be mentioned in the QC report as the subject the process was run on."
    )
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
    def __init__(self, fname_mask, fname_sc, fname_ref, path_template, path_ofolder, perslice, verbose):
        self.fname_mask = fname_mask
        self.midsagittal_sc_slice = None
        self.midsagittal_sc_slice_rpi = None
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
        self.rows = {}
        self.row_name = "slice" if perslice else "vert"
        self.atlas_roi_lst = None
        self.atlas_combinedlabels = {}
        self.distrib_matrix_dct = {}

        # output names
        self.pickle_name = extract_fname(self.fname_mask)[1] + '_analysis.pkl'
        self.excel_name = extract_fname(self.fname_mask)[1] + '_analysis.xls'

    def analyze(self):
        self.ifolder2tmp()

        # Orient input image(s) to RPI
        self.orient2rpi()

        # Label connected regions of the masked image
        self.label_lesion()

        # Compute angles for CSA correction and tissue bridge computations if
        # spinal cord segmentation is provided.
        # NB: If segmentation is not provided, then we will only compute volume, so
        #     no angle correction is needed
        if self.fname_sc is not None:
            self.angle_correction()
            self.get_midsagittal_slice()

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

    def pack_measures(self):

        with pd.ExcelWriter(self.excel_name, engine='xlsxwriter') as writer:
            self.measure_pd.to_excel(writer, sheet_name='measures', index=False, engine='xlsxwriter')

            # Save spreadsheet (for -f option)
            if self.path_template is not None:
                for sheet_name in self.distrib_matrix_dct:
                    self.distrib_matrix_dct[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False,
                                                                 engine='xlsxwriter')

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

        # For the tissue bridges, we get the minimum bridges across all lesions for the midsagittal slice
        if self.fname_sc is not None:
            midsagittal_dorsal_bridges = list()
            midsagittal_ventral_bridges = list()
            # Iterate across lesions to get the bridges for the midsagittal slice
            for idx, row in self.measure_pd.iterrows():
                if row['midsagittal_spinal_cord_slice'] is not None:        # just safety check
                    # Get the midsagittal slice number for the selected lesion
                    # Note that the midsagittal slice is the same for all lesions as it is based on the spinal cord
                    # segmentation
                    midsagittal_slice = str(int(row['midsagittal_spinal_cord_slice']))
                    if idx == 0:        # Print only once, not for each lesion
                        printv(f'\nMinimum tissue bridges across all lesions for the midsagittal slice '
                               f'(sagittal slice {midsagittal_slice})...', self.verbose, 'normal')
                    # Check whether the lesion has bridges in the midsagittal slice, if not, set the bridge width to NaN
                    if f'slice_{midsagittal_slice}_dorsal_bridge_width [mm]' in row and \
                            f'slice_{midsagittal_slice}_ventral_bridge_width [mm]' in row:
                        # Get dorsal and ventral tissue bridges for the mid-sagittal slice
                        dorsal_tissue_bridge = row[f'slice_{midsagittal_slice}_dorsal_bridge_width [mm]']
                        ventral_tissue_bridge = row[f'slice_{midsagittal_slice}_ventral_bridge_width [mm]']
                    # Note: the following else is for the case when all the lesions are parasagittal and there is thus
                    # no 'slice_{midsagittal_slice}_dorsal_bridge_width [mm]' column
                    else:
                        dorsal_tissue_bridge = np.nan
                        ventral_tissue_bridge = np.nan
                    # If there are NaN values, print a warning
                    # Note: for multiple lesions, there might one midsagittal lesion and another parasagittal lesion.
                    # In such a case, 'slice_{midsagittal_slice}_dorsal_bridge_width [mm]' column exists for both
                    # lesions (parasagittal lesion contains NaNs) and the previous 'if' is True for both lesions.
                    # This is why we cannot include the following printv into the previous 'else' statement because it
                    # would not be printed for the parasagittal lesion.
                    if np.isnan(dorsal_tissue_bridge) or np.isnan(ventral_tissue_bridge):
                        printv(f'WARNING: Lesion #{idx+1} does not exist in the midsagittal slice',
                               self.verbose, type='warning')

                    # Store the bridges for the midsagittal slice for the selected lesion
                    midsagittal_dorsal_bridges.append(dorsal_tissue_bridge)
                    midsagittal_ventral_bridges.append(ventral_tissue_bridge)

            # Compute the minimum bridges across all lesions for the midsagittal slice
            # Note: lesion(s) can be located on the parasagittal slices meaning that they do not have bridges in the
            # midsagittal slice, in such case the bridge width is NaN --> use np.nanmin to get the minimum value
            # Suppress the 'RuntimeWarning for All-NaN axis encountered' warning
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN axis encountered")
            min_dorsal_bridge = np.nanmin(midsagittal_dorsal_bridges)
            min_ventral_bridge = np.nanmin(midsagittal_ventral_bridges)
            printv(f'  Minimum dorsal bridge width [mm]: {np.round(min_dorsal_bridge, 2)}', self.verbose, type='info')
            printv(f'  Minimum ventral bridge width [mm]: {np.round(min_ventral_bridge, 2)}', self.verbose, type='info')

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
        printv(f'  Volume: {round(volume, 2)} mm^3', self.verbose, type='info')

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
        printv('  Maximum axial damage ratio: ' + str(np.round(maximum_axial_damage_ratio, 2)),
               self.verbose, type='info')

    def _measure_tissue_bridges(self, im_lesion_data, p_lst, idx):
        """
        Measure the tissue bridges (widths of spared tissue ventral and dorsal to the spinal cord lesion).
        Tissue bridges are quantified as the width of spared tissue at the **minimum** distance from cerebrospinal fluid
        (i.e., the spinal cord boundary) to the lesion boundary.

        NOTE: we compute the tissue bridges for all sagittal slices containing the lesion (i.e., for the midsagittal and
        parasagittal slices).

        Since we assume the input is in RPI orientation, then bridge widths are computed across the Y axis
        (AP axis), with dorsal == posterior (-Y) and ventral == anterior (+Y).

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

        # Restrict the lesion mask to the spinal cord mask (from anatomical level, it does not make sense to have lesion
        # outside the spinal cord mask)
        im_lesion_data = im_lesion_data * im_sc_data

        # Get the dimensions of the lesion mask
        dim = im_lesion_data.shape

        # --------------------------------------
        # Get slices with the lesion
        # --------------------------------------
        # We decided to use all sagittal slices containing the lesion to compute the tissue bridges
        # In other words, we compute the tissue bridges from the midsagittal slice and also from all parasagittal slices

        # Get slices with lesion
        # Note: we use [0] for the R-L direction as the orientation is RPI
        sagittal_lesion_slices = np.unique(np.where(im_lesion_data)[0])
        if self.verbose == 2:
            # Convert the sagittal slice numbers from RPI to the original orientation
            # '0' because of the R-L direction (first in RPI)
            sagittal_lesion_slices_print = [rpi_slice_to_orig_orientation(dim, self.orientation, slice, 0)
                                            for slice in sagittal_lesion_slices]
            # Reverse ordering
            sagittal_lesion_slices_print = sagittal_lesion_slices_print[::-1]
            printv('  Slices with lesion: ' + str(sagittal_lesion_slices_print), self.verbose, type='info')

        # --------------------------------------
        # Compute tissue bridges for each sagittal slice containing the lesion
        # --------------------------------------
        tissue_bridges_dict = {}
        # Loop across sagittal slices
        for sagittal_slice in sagittal_lesion_slices:
            # Get all axial slices (S-I direction) with the lesion for the selected sagittal slice
            # In other words, we will iterate through the lesion in S-I direction and compute tissue bridges for each
            # axial slice with the lesion
            # Note: we use [1] for the S-I direction as the orientation is RPI
            axial_lesion_slices = np.unique(np.where(im_lesion_data[sagittal_slice, :, :])[1])
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
                     'ventral_bridge_width': ventral_bridge_width}

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
            df_temp = tissue_bridges_df[tissue_bridges_df['sagittal_slice'] == sagittal_slice].copy()

            # Get the width of the tissue bridges in mm (by multiplying by p_lst[1]) and use np.cos(self.angles_sagittal[SLICE])
            # to correct for the angle of the spinal cord with respect to the axial slice
            # NOTE: the orientation is RPI (because we reoriented the image to RPI using orient2rpi()); therefore
            # p_lst[0] is the pixel size in the R-L direction, p_lst[1] is the pixel size in the A-P direction, and
            # p_lst[2] is the pixel size in the S-I direction.
            # Since we are computing dorsal and ventral tissue bridges, we use p_lst[1] (A-P direction)
            dorsal_bridge_width_mm = df_temp.apply(lambda row:
                                                   row['dorsal_bridge_width'] * p_lst[1] *
                                                   np.cos(self.angles_sagittal[row['axial_slice']]), axis=1)
            ventral_bridge_width_mm = df_temp.apply(lambda row:
                                                    row['ventral_bridge_width'] * p_lst[1] *
                                                    np.cos(self.angles_sagittal[row['axial_slice']]), axis=1)

            # Add the columns to the DataFrame
            # For some reason I need to add the columns one by one. When I tried to write directly to the DataFrame,
            # I got the following error:
            #   "IndexError: only integers, slices (:), ellipsis (...), numpy.newaxis (None) and integer or boolean
            #   arrays are valid indices"
            df_temp['dorsal_bridge_width_mm'] = dorsal_bridge_width_mm
            df_temp['ventral_bridge_width_mm'] = ventral_bridge_width_mm

            # Get the axial slices corresponding to the minimum bridge widths
            # This information is printed to terminal
            min_dorsal_bridge_width_slice = df_temp.loc[df_temp['dorsal_bridge_width_mm'].idxmin(), 'axial_slice']
            min_ventral_bridge_width_slice = df_temp.loc[df_temp['ventral_bridge_width_mm'].idxmin(), 'axial_slice']

            # Get the minimum dorsal and ventral bridge widths
            min_dorsal_bridge_width_mm = float(df_temp['dorsal_bridge_width_mm'].min())
            min_ventral_bridge_width_mm = float(df_temp['ventral_bridge_width_mm'].min())
            min_total_bridge_width_mm = min_dorsal_bridge_width_mm + min_ventral_bridge_width_mm

            # Convert the sagittal and axial slice numbers from RPI to the original orientation
            # '0' because of the R-L direction (first in RPI)
            sagittal_slice = rpi_slice_to_orig_orientation(dim, self.orientation, sagittal_slice, 0)
            # '2' because of the S-I direction (third in RPI)
            min_dorsal_bridge_width_slice = rpi_slice_to_orig_orientation(dim, self.orientation,
                                                                          min_dorsal_bridge_width_slice, 2)
            min_ventral_bridge_width_slice = rpi_slice_to_orig_orientation(dim, self.orientation,
                                                                           min_ventral_bridge_width_slice, 2)

            # Save the minimum tissue bridges
            self.measure_pd.loc[idx, f'slice_{sagittal_slice}_dorsal_bridge_width [mm]'] = min_dorsal_bridge_width_mm
            self.measure_pd.loc[idx, f'slice_{sagittal_slice}_ventral_bridge_width [mm]'] = min_ventral_bridge_width_mm
            self.measure_pd.loc[idx, f'slice_{sagittal_slice}_total_bridge_width [mm]'] = min_total_bridge_width_mm
            printv(f'  Sagittal slice {sagittal_slice}, Minimum dorsal tissue bridge width: '
                   f'{np.round(min_dorsal_bridge_width_mm, 2)} mm (axial slice {min_dorsal_bridge_width_slice})',
                   self.verbose, type='info')
            printv(f'  Sagittal slice {sagittal_slice}, Minimum ventral tissue bridge width: '
                   f'{np.round(min_ventral_bridge_width_mm, 2)} mm (axial slice {min_ventral_bridge_width_slice})',
                   self.verbose, type='info')
            printv(f'  Sagittal slice {sagittal_slice}, Total tissue bridge width: '
                   f'{np.round(min_total_bridge_width_mm, 2)} mm', self.verbose, type='info')

    def _measure_length(self, im_data, p_lst, idx):
        """
        Measure the length of the lesion along the superior-inferior axis when taking into account the angle correction
        The length is computed across all sagittal lesion (i.e., midsagittal and parasagittal) slices meaning that the
        measurement is 3D.
        For lesion length for the midsagittal slice only, see _measure_length_midsagittal_slice().
        """
        length_cur = np.sum([p_lst[2] / np.cos(self.angles_3d[zz]) for zz in np.unique(np.where(im_data)[2])])
        self.measure_pd.loc[idx, 'length [mm]'] = length_cur
        printv('  (S-I) length: ' + str(np.round(length_cur, 2)) + ' mm', self.verbose, type='info')

    def _measure_length_midsagittal_slice(self, im_lesion_data, p_lst, idx):
        """
        Measure the length of the lesion along the superior-inferior axis in the **midsagittal slice** when taking into
        account the angle correction.

        :param im_lesion_data: 3D numpy array: mask of the lesion. The orientation is assumed to be RPI (because we
        reoriented the image to RPI using orient2rpi())
        :param p_lst: list, pixel size of the lesion
        :param idx: int, index of the lesion
        """
        # Fetch a list of axial slice numbers that are nonzero in the mid-sagittal slice (RPI)
        im_data_midsagittal = im_lesion_data[self.midsagittal_sc_slice_rpi, :, :]  # 3D -> 2D, dim=[AP, SI]
        nonzero_axial_slices = np.unique(np.where(im_data_midsagittal)[1])  # [1] -> SI

        # Compute the length of the lesion along the superior-inferior axis in the midsagittal slice
        # The length is computed as the sum of the angle corrected axial slice thicknesses
        # Note: if there is no lesion in the midsagittal slice, the length will be 0 (because np.sum([]) = 0.0)
        length_cur = np.sum([p_lst[2] / np.cos(self.angles_sagittal[s])  # p_lst[2] -> pixel size of SI axis
                             for s in nonzero_axial_slices])

        self.measure_pd.loc[idx, 'length_midsagittal_slice [mm]'] = length_cur
        printv(f'  (S-I) length in the midsagittal slice: {(np.round(length_cur, 2))} mm',
               self.verbose, type='info')

    def _measure_width_midsagittal_slice(self, im_lesion_data, p_lst, idx):
        """
        Measure the width of the lesion along the anterior-posterior axis in the **midsagittal slice** when taking into
        account the angle correction.
        The width is defined as the maximum lesion width in the A-P axis across all axial slices with the lesion in
        the midsagittal slice.

        :param im_lesion_data: 3D numpy array: mask of the lesion. The orientation is assumed to be RPI (because we
        reoriented the image to RPI using orient2rpi())
        :param p_lst: list, pixel size of the lesion
        :param idx: int, index of the lesion
        """
        # Fetch a list of axial slice numbers that are nonzero in the mid-sagittal slice (RPI)
        im_data_midsagittal = im_lesion_data[self.midsagittal_sc_slice_rpi, :, :]  # 3D -> 2D, dim=[AP, SI]
        nonzero_axial_slices = np.unique(np.where(im_data_midsagittal)[1])  # [1] -> SI

        # Iterate across axial slices to compute lesion width
        lesion_width_dict = {}
        for axial_slice in nonzero_axial_slices:
            # Get the lesion segmentation mask of the selected 2D axial slice
            slice_lesion_data = im_lesion_data[self.midsagittal_sc_slice_rpi, :, axial_slice]
            # Get the indices of the lesion mask for the selected axial slice to compute the lesion width.
            # The lesion width is defined as max - min + 1
            # Note: we intentionally use 'max - min + 1' instead of 'len(np.where(slice_lesion_data)[0])' because the
            # 'len' approach would return the number of elements, which could be influenced, for example, by the
            # presence of holes in the lesion mask.
            # Context: https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4617#discussion_r1744031056
            slice_min = np.where(slice_lesion_data)[0][0]     # [0] returns the most dorsal elements
            slice_max = np.where(slice_lesion_data)[0][-1]    # [-1] returns the most ventral elements
            lesion_width_dict[axial_slice] = slice_max - slice_min + 1

        # Get the width in mm and apply the angle correction
        width_cur_dict = {axial_slice: p_lst[1] * np.cos(self.angles_sagittal[axial_slice]) * lesion_width  # p_lst[1] -> pixel size of AP axis
                          for axial_slice, lesion_width in lesion_width_dict.items()}

        # Get the maximum width across all axial slices
        # Check if width_cur_dict is empty, if so, it means that the lesion does not exist in the midsagittal slice.
        # In this case, set the width to 0
        width_cur = max(width_cur_dict.values()) if width_cur_dict else 0

        # Save the width of the lesion along the anterior-posterior axis in the midsagittal slice
        self.measure_pd.loc[idx, 'width_midsagittal_slice [mm]'] = width_cur
        printv(f'  (A-P) width in the midsagittal slice: {str(np.round(width_cur, 2))} mm',
               self.verbose, type='info')

    def _measure_diameter(self, im_data, p_lst, idx):
        """
        Measure the max. equivalent diameter of the lesion when taking into account the angle correction
        """
        area_lst = [np.sum(im_data[:, :, zz]) * np.cos(self.angles_3d[zz]) * p_lst[0] * p_lst[1] for zz in range(im_data.shape[2])]
        diameter_cur = 2 * np.sqrt(max(area_lst) / np.pi)
        self.measure_pd.loc[idx, 'max_equivalent_diameter [mm]'] = diameter_cur
        printv('  Max. equivalent diameter: ' + str(np.round(diameter_cur, 2)) + ' mm', self.verbose, type='info')

    def ___pve_weighted_avg(self, im_mask_data, im_atlas_data):
        return im_mask_data * im_atlas_data

    def __keep_only_indices(self, image, indices):
        """Keep values defined by indices, and set all other coordinates to zero."""
        image_out = np.zeros_like(image)
        image_out[indices] = image[indices]
        return image_out

    def __relative_ROIvol_in_mask(self, im_mask_data, im_atlas_roi_data, p_lst, indices_to_keep):
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
        #           - p_lst - type=list of float
        #           - indices_to_keep - type=(anything that can be used to index numpy arrays)
        #                               anything outside this mask will be set to 0
        #
        im_atlas_roi_data = self.__keep_only_indices(im_atlas_roi_data, indices_to_keep)
        im_mask_data = self.__keep_only_indices(im_mask_data, indices_to_keep)

        im_mask_roi_data_wa = self.___pve_weighted_avg(im_mask_data=im_mask_data, im_atlas_data=im_atlas_roi_data)
        vol_tot_roi = np.sum(im_atlas_roi_data) * p_lst[0] * p_lst[1] * p_lst[2]
        vol_mask_roi_wa = np.sum(im_mask_roi_data_wa) * p_lst[0] * p_lst[1] * p_lst[2]

        return vol_mask_roi_wa, vol_tot_roi

    def _measure_eachLesion_distribution(self, lesion_id, atlas_data, im_vert, im_lesion, p_lst):
        sheet_name = 'lesion#' + str(lesion_id) + '_distribution'
        self.distrib_matrix_dct[sheet_name] = pd.DataFrame.from_dict({'row': [str(v) for v in self.rows.keys()]})

        # initialized to 0 for each vertebral level and each PAM50 tract
        for tract_id in atlas_data:
            self.distrib_matrix_dct[sheet_name]['PAM50_' + str(tract_id).zfill(2)] = [0] * len(self.rows)

        vol_mask_tot = 0.0  # vol tot of this lesion through the vertebral levels and PAM50 tracts
        im_vert_and_lesion = im_vert * im_lesion  # to check which vertebral levels have lesions
        # Loop over slices or vertebral levels
        for row, indices_to_keep in sct_progress_bar(self.rows.items(), unit=self.row_name,
                                                     desc="  Computing lesion distribution (volume values)"):
            if np.count_nonzero(im_vert_and_lesion[indices_to_keep]):  # if there is lesion in this vertebral level
                idx = self.distrib_matrix_dct[sheet_name][self.distrib_matrix_dct[sheet_name].row == str(row)].index
                for tract_id in atlas_data:  # Loop over PAM50 tracts
                    res_lst = self.__relative_ROIvol_in_mask(im_mask_data=im_lesion,
                                                             im_atlas_roi_data=atlas_data[tract_id],
                                                             p_lst=p_lst,
                                                             indices_to_keep=indices_to_keep)
                    self.distrib_matrix_dct[sheet_name].loc[idx, 'PAM50_' + str(tract_id).zfill(2)] = res_lst[0]
                    vol_mask_tot += res_lst[0]

        # convert the volume values in distrib_matrix_dct to percentage values
        for row in sct_progress_bar(self.rows.keys(), unit=self.row_name,
                                    desc="  Converting volume values into percentage values"):
            idx = self.distrib_matrix_dct[sheet_name][self.distrib_matrix_dct[sheet_name].row == str(row)].index
            for tract_id in atlas_data:
                val = self.distrib_matrix_dct[sheet_name].loc[idx, 'PAM50_' + str(tract_id).zfill(2)].values[0]
                self.distrib_matrix_dct[sheet_name].loc[idx, 'PAM50_' + str(tract_id).zfill(2)] = val * 100.0 / vol_mask_tot

        # Add the total column
        self.distrib_matrix_dct[sheet_name]['total % (all tracts)'] = \
            self.distrib_matrix_dct[sheet_name].sum(numeric_only=True, axis=1)

        # Add additional columns for the "CombinedLabels" defined by info_label.txt
        for label_name, sublabels in self.atlas_combinedlabels.items():
            column_names_to_sum = [f"PAM50_{subid:02}" for subid in sublabels]
            self.distrib_matrix_dct[sheet_name][label_name] = \
                self.distrib_matrix_dct[sheet_name][column_names_to_sum].sum(axis=1)

        # Add the total row
        self.distrib_matrix_dct[sheet_name] = self.distrib_matrix_dct[sheet_name].append(
            self.distrib_matrix_dct[sheet_name].sum(numeric_only=True, axis=0),
            ignore_index=True
        )
        self.distrib_matrix_dct[sheet_name].iloc[
            -1, self.distrib_matrix_dct[sheet_name].columns.get_loc('row')
        ] = f'total % (all {self.row_name})'

    def __regroup_per_tracts(self, vol_dct, tracts):
        res_mask = [vol_dct[t][0] for t in vol_dct if t in tracts]
        res_tot = [vol_dct[t][1] for t in vol_dct if t in tracts]
        return np.sum(res_mask) * 100.0 / np.sum(res_tot)

    def _measure_totLesion_distribution(self, im_lesion, atlas_data, im_vert, p_lst):

        sheet_name = 'ROI_occupied_by_lesion'
        total_row = f'total % (all {self.row_name})'
        rows_with_total = {
            **self.rows,
            # numpy array index equivalent to [:, :, :]
            total_row: (slice(None), slice(None), slice(None)),
        }
        self.distrib_matrix_dct[sheet_name] = pd.DataFrame.from_dict({'row': [str(r) for r in rows_with_total]})

        # initialized to 0 for each vertebral level and each PAM50 tract
        for tract_id in atlas_data:
            self.distrib_matrix_dct[sheet_name][f"PAM50_{tract_id:02}"] = [0] * len(rows_with_total)

        im_vert_and_lesion = im_vert * im_lesion  # to check which vertebral levels have lesions
        # loop over slices/vertlevels
        for row, indices_to_keep in sct_progress_bar(rows_with_total.items(), unit=self.row_name,
                                                     desc="  Computing ROI distribution for all lesions"):
            if row == total_row or np.count_nonzero(im_vert_and_lesion[indices_to_keep]):
                res_perTract_dct = {}  # for each tract compute the volume occupied by lesion and the volume of the tract
                idx = self.distrib_matrix_dct[sheet_name][self.distrib_matrix_dct[sheet_name].row == str(row)].index
                for tract_id in atlas_data:  # loop over the tracts
                    res_perTract_dct[tract_id] = self.__relative_ROIvol_in_mask(im_mask_data=im_lesion,
                                                                                im_atlas_roi_data=atlas_data[tract_id],
                                                                                p_lst=p_lst,
                                                                                indices_to_keep=indices_to_keep)

                # group tracts to compute involvement in CombinedLabels (GM, WM, DC, VF, LF)
                for label_name, sublabels in self.atlas_combinedlabels.items():
                    self.distrib_matrix_dct[sheet_name].loc[idx, label_name] = \
                        self.__regroup_per_tracts(vol_dct=res_perTract_dct, tracts=sublabels)

                # save involvement in each PAM50 tracts
                for tract_id in atlas_data:
                    self.distrib_matrix_dct[sheet_name].loc[idx, 'PAM50_' + str(tract_id).zfill(2)] = res_perTract_dct[tract_id][0] * 100.0 / res_perTract_dct[tract_id][1]

    def measure(self):
        im_lesion = Image(self.fname_label)
        im_lesion_data = im_lesion.data
        p_lst = im_lesion.dim[4:7]  # voxel size

        label_lst = [label for label in np.unique(im_lesion_data) if label]  # lesion label IDs list

        # Print warning if there is no lesion (label_lst is empty list)
        if not label_lst:
            printv(f'WARNING: No lesion found in {self.fname_label}.', self.verbose, 'warning')

        if self.path_template is not None:
            if os.path.isfile(self.path_levels):
                img_vert = Image(self.path_levels)
                im_vert_data = img_vert.data
                if self.row_name == "vert":
                    # list of vertebral levels available in the input image
                    # precompute the list of indices for each vertebral level
                    # these indices are used to zero out certain levels (to compute the volume of the remaining levels)
                    self.rows = {
                        vert: np.where(im_vert_data == vert)
                        for vert in np.unique(im_vert_data) if vert
                    }
                else:
                    assert self.row_name == "slice"
                    # Keep the same vert image, but uses slices instead
                    self.rows = {
                        z: (slice(None), slice(None), z)  # numpy array index equivalent to [:, :, z]
                        for z in range(im_vert_data.shape[2])
                    }

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
                self.measure_pd.loc[label_idx, 'midsagittal_spinal_cord_slice'] = self.midsagittal_sc_slice
                self._measure_length(im_lesion_data_cur, p_lst, label_idx)
                self._measure_diameter(im_lesion_data_cur, p_lst, label_idx)
                self._measure_axial_damage_ratio(im_lesion_data_cur, p_lst, label_idx)
                self._measure_length_midsagittal_slice(im_lesion_data_cur, p_lst, label_idx)
                self._measure_width_midsagittal_slice(im_lesion_data_cur, p_lst, label_idx)
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
            print("\nROI percentage taken up by all lesions...")
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
        # We set minmax=False to prevent cropping and ensure that `self.angles_3d[iz]` covers all z slices of `im_seg`
        _, arr_ctl, arr_ctl_der, _ = get_centerline(im_seg, param=ParamCenterline(minmax=False), verbose=1)
        x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = arr_ctl_der

        self.angles_3d = np.full(nz, np.nan, dtype=np.double)
        self.angles_sagittal = np.full(nz, np.nan, dtype=np.double)

        # loop across x_centerline_deriv (instead of [min_z_index, max_z_index], which could vary after interpolation)
        for iz in range(x_centerline_deriv.shape[0]):
            # normalize the tangent vector to the centerline (i.e. its derivative)
            tangent_vect = self._normalize(np.array(
                [x_centerline_deriv[iz] * px, y_centerline_deriv[iz] * py, pz]))
            # compute the angle between the normal vector of the plane and the vector z
            self.angles_3d[iz] = np.arccos(np.vdot(tangent_vect, np.array([0, 0, 1])))

            # this assumes RPI orientation, and computes the angle of the centerline
            # when projected onto the sagittal plane
            tangent_vect = self._normalize(np.array([0, y_centerline_deriv[iz] * py, pz]))
            # compute the angle between the normal vector of the plane and the vector z
            self.angles_sagittal[iz] = np.arccos(np.vdot(tangent_vect, np.array([0, 0, 1])))

    def get_midsagittal_slice(self):
        """Get the midsagittal slice based on the lesion segmentation mask."""
        # Get the RPI-oriented midsagittal slice number (to be reused in other functions)
        im_lesion_data = Image(self.fname_label).data
        # Get the slice with the largest lesion area
        nonzero_slices = np.unique(np.where(im_lesion_data)[0])     # RPI image: [0] -> RL (sagittal)
        self.midsagittal_sc_slice_rpi = nonzero_slices[np.argmax([np.sum(im_lesion_data[slice, :, :]) for slice in nonzero_slices])]
        # Convert the RPI-oriented slice number to the original orientation (for outputting to user)
        self.midsagittal_sc_slice = rpi_slice_to_orig_orientation(dim=im_lesion_data.shape,
                                                                  orig_orientation=self.orientation,
                                                                  slice_number=self.midsagittal_sc_slice_rpi,
                                                                  axis=0)  # 0 = RL
        printv(f'Midsagittal slice of the lesion cord: {self.midsagittal_sc_slice}', self.verbose, type='info')

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
            tract_ids = []
            for fname_atlas_roi in sorted(os.listdir(self.path_atlas)):
                if fname_atlas_roi.endswith('.nii.gz'):
                    tract_id = int(fname_atlas_roi.split('_')[-1].split('.nii.gz')[0])
                    if tract_id < 36:  # Not interested in CSF
                        tract_ids.append(tract_id)
                        copy(os.path.join(self.path_atlas, fname_atlas_roi), self.tmp_dir)
                        self.atlas_roi_lst.append(fname_atlas_roi)

            # fetch "CombinedLabels" from atlas info_label.txt
            # NB: We have to do this here (rather than in tmp_dir) because `read_label_file` will fail unless all files
            # are present, and we skip copying the CSF to the tmp dir in the lines above.
            printv("\nLoading CombinedLabels from `info_label.txt`...")
            if os.path.isfile(os.path.join(self.path_atlas, "info_label.txt")):
                _, _, _, _, combinedlabel_names, label_groups, _ = read_label_file(self.path_atlas, "info_label.txt")
                combined_labels = {}
                for label_name, sublabels in zip(combinedlabel_names, label_groups):
                    # If one of the CombinedLabels matches the total set of all atlas labels, discard it
                    # In practice, this will cause the 'spinal cord' label to be discarded (spanning tracts 0:35).
                    if set(tract_ids) == set(sublabels):
                        printv(f"WARNING: CombinedLabel '{label_name}' is identical to the 'total' column that sums "
                               f"all atlas labels ([{sublabels[0]}:{sublabels[-1]}]). "
                               f"The '{label_name}' column will not be added to the output spreadsheet.",
                               self.verbose, type="warning")
                    else:
                        combined_labels[label_name] = sublabels
                self.atlas_combinedlabels = combined_labels

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
    path_results = os.path.expanduser(arguments.ofolder)        # expand '~' to user home directory
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
                               perslice=arguments.perslice,
                               verbose=verbose)

    # run the analyze
    lesion_obj.analyze()

    # Create QC report for tissue bridges (only if SC is provided)
    if arguments.qc is not None:
        if fname_sc is not None:
            sct_analyze_lesion(
                fname_input=fname_mask,
                fname_label=lesion_obj.fname_label,
                fname_sc=fname_sc,
                measure_pd=lesion_obj.measure_pd,
                argv=argv,
                path_qc=arguments.qc,
                dataset=arguments.qc_dataset,
                subject=arguments.qc_subject,
            )
        else:
            printv("WARNING: Spinal cord segmentation not provided, skipping QC. "
                   "(SC seg is required to show tissue bridges).",
                   verbose=verbose, type="warning")

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
