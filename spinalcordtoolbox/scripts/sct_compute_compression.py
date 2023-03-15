#!/usr/bin/env python
#########################################################################################
#
# Compute maximum spinal cord compression using AP diameter or other morphometrics.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Sandrine Bédard, Jan Valosek, Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################
# TODO: add option to normalize or not
# TODO: maybe create an API or move some functions
import sys
import os
import numpy as np
import csv
import logging
from typing import Sequence
import pandas as pd
from spinalcordtoolbox.utils import SCTArgumentParser, Metavar, init_sct, set_loglevel
from spinalcordtoolbox.utils.fs import get_absolute_path, extract_fname, printv
from spinalcordtoolbox.image import Image
from spinalcordtoolbox import __data_dir__


logger = logging.getLogger(__name__)


# PARSER
# ==========================================================================================
def get_parser():
    parser = SCTArgumentParser(
        description='Compute spinal cord compression morphometrics such as the Maximum Spinal Cord Compression [1], '
                    'the antero-posterior (AP) diameter, or other relevant spinal cord morphometrics that are output'
                    ' by the function sct_process_segmentation (CSA, RL diameter, eccentricity, solidity, etc.).'
                    ' Metrics are normalized using a database of spinal cord morphometrics built from healthy control'
                    ' subjects. This database uses the PAM50 template as an anatomical reference system.'

                    '[1]: Miyanji F, Furlan JC, Aarabi B, Arnold PM, Fehlings MG. Acute cervical traumatic spinal cord injury:'
                    ' MR imaging findings correlated with neurologic outcome--prospective study with 100 consecutive'
                    ' patients. Radiology 2007;243(3):820-827.'
    )

    mandatoryArguments = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatoryArguments.add_argument(
        '-i',
        metavar=Metavar.file,
        required=True,
        help='CSV morphometric file obtained with sct_process_segmentation. Example: csa.csv.'
    )
    mandatoryArguments.add_argument(
        '-l',
        metavar=Metavar.file,
        required=True,
        help='NIfTI file that includes labels at the compression sites. Each compression site is denoted by a single voxel of value `1`. '
             'Example: sub-001_T2w_compression_labels.nii.gz'
    )
    mandatoryArguments.add_argument(
        '-i-PAM50',
        metavar=Metavar.file,
        required=True,
        help='CSV morphometric file in the PAM50 space, obtained by running: '
        'sct_process_segmentation -normalize-PAM50.'
    )
    mandatoryArguments.add_argument(
        '-metric',
        required=False,
        help='Metric to normalize.'
        'Choices: area, diameter_AP, diameter_RL, eccentricity, solidity\n',
        default='diameter_AP',
        choices=['diameter_AP', 'area', 'diameter_RL', 'eccentricity', 'solidity'],
        metavar=Metavar.file,
    )
    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        '-file-participants',
        metavar=Metavar.file,
        default='participants.tsv',
        help='participants.tsv file of healthy controls.'
    )
    optional.add_argument(
        '-sex',
        type=str,
        choices=['F', 'M'],
        help='Sex of healthy subject to use for the normalization. Requires the flag "-file-participants".'
        ' By default, both sexes are used.'
    )
    optional.add_argument(
        '-age',
        type=int,
        nargs=2,
        metavar="[0 100]",
        help='Age range of healthy subjects to use for the normalization. Requires the flag "-file-participants".'
        'Example: "-age 60 80". By default, all ages are considered.'
    )
    optional.add_argument(
        '-o',
        metavar=Metavar.file,
        help='Output CSV file name. If not provided, the suffix "_compression_metrics" is added to the file name provided by the flag "-i"'
    )
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")
    optional.add_argument(
        '-v',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode")

    return parser


# Functions for Step 2 (Processing healthy controls from `PAM50_normalized_metrics` dataset)
# ==========================================================================================
def select_HC(fname_participants, sex=None, age=None):
    """
    Selects healthy controls to use for normalization based on sex and age range specified by the user.
    :param fname_participants: Filename of participants.tsv
    :param sex: Either F for female or M for male.
    :param age: list: List of age range to select subjects.
    :return list_to_include: list: List of participant_id
    """
    # Load participants information
    data = pd.read_csv(fname_participants, sep="\t")
    # Initialize lists
    list_sub_sex = []
    list_sub_age = []
    # select subject with same sex
    if sex:
        list_sub_sex = data.loc[data['sex'] == sex, 'participant_id'].to_list()
        if not age:
            list_to_include = list_sub_sex
    # select subjects within age range
    if age:
        list_sub_age = data.loc[data['age'].between(age[0], age[1]), 'participant_id'].to_list()
        if not sex:
            list_to_include = list_sub_age
    if age and sex:
        list_to_include = set(list_sub_age).intersection(list_sub_sex)
    if not age and not sex:
        list_to_include = data['participant_id'].to_list()
    printv(f'{len(list_to_include)} healthy controls are used for normalization')
    return list(list_to_include)


def average_HC(ref_folder, metric, list_HC):
    """
    Gets metrics of healthy controls in PAM50 anatomical dimensions and averages across subjects.
    :param ref_folder: path to folder where .csv fiels of healthy controls are.
    :param metric: str: metric to perform normalization
    :param list_HC: list: List of healthy controls to include
    :return df:
    """
    # Initialize empty dataframe
    df = pd.DataFrame()
    # Create empty dict to put dataframe of each healthy control
    d = {}
    # Iterator to count number of healthy subjects
    i = 0
    # Loop through .csv files of healthy controls
    for file in os.listdir(ref_folder):
        if 'PAM50.csv' in file:
            subject = os.path.basename(file).split('_')[0]
            if subject in list_HC:
                d[file] = pd.read_csv(os.path.join(ref_folder, file)).astype({metric: float})
                i = i+1
    first_key = next(iter(d))
    # Create an empty dataframe with same columns
    df = pd.DataFrame(columns=d[first_key].columns)
    df['VertLevel'] = d[first_key]['VertLevel']
    df['Slice (I->S)'] = d[first_key]['Slice (I->S)']
    # Loop through all HC
    for key, values in d.items():
        for column in d[key].columns:
            if 'MEAN' in column:
                if df[column].isnull().values.all():
                    df[column] = d[key][column]
                else:
                    # Sum all columns that have MEAN key
                    df[column] = df[column] + d[key][column].tolist()
    # Divide by number of HC
    for column in df.columns:
        if 'MEAN' in column:
            df[column] = df[column]/i
    return df


# Functions for Step 3 (Fetching the compressed levels from the subject and PAM50 space)
# ==========================================================================================
def get_verterbral_level_from_slice(slices, df_metrics):
    """
    From slices, gets the coresponding vertebral level and creates a dict fo level and corresponding slice(s).
    :param slices: list: list of slices number.
    :param df_metrics: pandas.DataFrame: dataframe of metrics (output of sct_process_segmentation).
    :return level_slice_dict: dict:
    """
    idx = df_metrics['Slice (I->S)'].isin(slices).tolist()
    level_compression = df_metrics.loc[idx, ['VertLevel', 'Slice (I->S)']]
    if level_compression.empty:
        raise ValueError(f"Slice {slices} doesn't have a computed metric")
    level_slice_dict = {}
    for level in np.unique(level_compression['VertLevel']):
        level_slice_dict[level] = level_compression.loc[level_compression['VertLevel'] == level, 'Slice (I->S)'].to_list()
    return level_slice_dict


def get_up_lw_levels(levels, df, metric):
    """
    Get most upper level from all compressed levels and lowest level from all compressed levels
    :param levels: list: Compressed levels.
    :param df: pandas.DataFrame: Metrics output of sct_process_segmentation.
    :return upper_level: int: Smallest level (closest to superior)
    :return lower_level: int: Highest level (closest to inferior)
    """
    upper_level = min(levels) - 1
    lower_level = max(levels) + 1
    # Check if lower and upper levels are available:
    upper_empty = df.loc[df['VertLevel'] == upper_level, metric].empty
    lower_empty = df.loc[df['VertLevel'] == lower_level, metric].empty
    if not lower_empty and upper_empty:
        # Set upper level to lower level. Only normalize using the lower level instead of an average of upper and lower level.
        upper_level = lower_level
    elif not upper_empty and lower_empty:
        # Set lower level to upper level. Only normalize using the lower level instead of an average of upper and lower level.
        lower_level = upper_level

    elif lower_empty and upper_empty:
        ValueError('No levels above nor below all compressions are available.')
    return upper_level, lower_level


def get_slices_in_PAM50(compressed_level_dict, df_metrics, df_metrics_PAM50):
    """
    Get corresponding slice of compression in PAM50 space.
    :param compressed_level_dict: dict: Dictionary of levels and corresponding slice(s).
    :param df_metrics: pandas.DataFrame: Metrics output of sct_process_segmentation.
    :param df_metrics_PAM50: pandas.DataFrame: Metrics output of sct_process_segmentation in PAM50 anatomical dimensions.
    :return compression_level_dict_PAM50:
    """
    # TODO - won't be ok for most upper and lowest levels if they are not complete...
    compression_level_dict_PAM50 = {}
    # Loop across slices and levels with compression
    for level, slices in compressed_level_dict.items():
        # Number of slices in native image
        nb_slices_level = len(df_metrics.loc[df_metrics['VertLevel'] == level, 'VertLevel'].to_list())
        # Number of slices in PAM50
        nb_slices_PAM50 = len(df_metrics_PAM50.loc[df_metrics_PAM50['VertLevel'] == level, 'VertLevel'].to_list())
        # Do interpolation from native space to PAM50
        x_PAM50 = np.arange(0, nb_slices_PAM50, 1)
        x = np.linspace(0, nb_slices_PAM50 - 1, nb_slices_level)
        new_slices_coord = np.interp(x_PAM50, x, df_metrics.loc[df_metrics['VertLevel'] == level, 'Slice (I->S)'].to_list())
        # find nearest index
        slices_PAM50 = []
        for slice in slices:
            # get index corresponding to the min value
            idx = np.abs(new_slices_coord - slice).argmin()
            new_slice = df_metrics_PAM50.loc[df_metrics_PAM50['VertLevel'] == level, 'Slice (I->S)'].to_list()[idx]
            slices_PAM50.append(new_slice)
        compression_level_dict_PAM50[level] = slices_PAM50
    return compression_level_dict_PAM50


# Functions for Step 4 (Computing MSCC using spinal cord morphometrics.)
# ==========================================================================================
def average_compression_PAM50(slice_thickness, slice_thickness_PAM50, metric, df_metrics_PAM50, upper_level, lower_level, slice):
    """
    Defines slices to average metric at compression level following slice thickness and averages metric at
    compression, across the entire level above and below compression.
    :param slice_thickness: float: slice thickness of native image space.
    :param slice_thickness_PAM50: float: slice thickness of the PAM50.
    :param metric: str: metric to perform normalization
    :param df_metrics_PAM50: pandas.DataFrame: Metrics of sct_process_segmentation in PAM50 anatomical dimensions.
    :param upper_level: int: level above compression.
    :param lower_level: int: level below compression.
    :param slice: int: slice of spinal cord compression.
    :return upper_AP_mean:
    :retrun lower_AP_mean:
    :retrun compressed_AP_mean:
    :retrun slices_avg: Slices in PAM50 space to average metric.

    """
    # If resolution of image is higher than PAM50 template, get slices equivalent to native slice thickness
    nb_slice = slice_thickness//slice_thickness_PAM50
    if nb_slice > 1:
        slices_avg = np.arange(min(slice) - nb_slice//2, max(slice) + nb_slice//2, 1)
    # If more than one slice has compression, get all slices from that range
    if len(slice) > 1:
        slices_avg = np.arange(min(slice), max(slice), 1)
    else:
        slices_avg = slice
    return average_metric(df_metrics_PAM50, metric, upper_level, lower_level, slices_avg), slices_avg


def average_metric(df, metric, upper_level, lower_level, slices_avg):
    """
    Average metric at compression level, at level above and below.
    :param df: pandas.DataFrame: Metrics output of sct_process_segmentation.
    :param metric: str: metric to perform normalization
    :param upper_level: int: level above compression.
    :param lower_level: int: level below compression.
    :param slices_avg: Slices in PAM50 space to average metrics.
    :return: ma: float64: Metric above the compression
    :retrun: mb: float64: Metric below the compression
    :retrun: mi: float64: Metric at the compression level
    """
    # find index of slices to average
    idx = df['Slice (I->S)'].isin(slices_avg).tolist()
    ma = df.loc[df['VertLevel'] == upper_level, metric].mean()
    mb = df.loc[df['VertLevel'] == lower_level, metric].mean()
    mi = df.loc[idx, metric].mean()
    return ma, mb, mi


def metric_ratio(ma, mb, mi):
    """
    Compute MSCC (Maximum Spinal Cord Compression) using the chosen metric of compression and of levels
    above and bellow.
    :param float: ma: metric of level above compression level.
    :param float: mb: metric of level above compression level.
    :param float: mi: metric at compression level.
    :return float: metric ratio in %
    """
    return (1 - float(mi) / ((ma + mb) / float(2))) * 100


def metric_ratio_norm(ap, ap_HC):
    """
    Compute normalized MSCC (Maximum Spinal Cord Compression) using the chosen metric at the compression and
    levels above and bellow.
    Each metric is divided by the corresponding value in healthy controls.
    :param list: ap: list metric value of level above, below and at compression of patient.
    :param list: ap_HC: list metric value of level above, below and at compression of healthy
    controls.
    :return float: MSCC normalized in %
    """
    ma = ap[0]/ap_HC[0]
    mb = ap[1]/ap_HC[1]
    mi = ap[2]/ap_HC[2]
    return metric_ratio(ma, mb, mi)


def save_csv(fname_out, level, metric, metric_ratio, metric_ratio_nrom, filename):
    """
    Save .csv file of MSCC results.
    :param fname_out:
    :param level: int: Level of compression.
    :param metric: str: metric to perform normalization
    :param metric_ratio: float:
    :param metric_ratio_nrom:
    :param filename: str: input filename
    :retrun:
    """
    if not os.path.isfile(fname_out):
        with open(fname_out, 'w') as csvfile:
            header = ['filename', 'Compression Level', metric + ' ratio', 'Normalized ' + metric + ' ratio']
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
    with open(fname_out, 'a') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        line = [filename, level, metric_ratio, metric_ratio_nrom]
        csv_writer.writerow(line)


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)    # values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG]

    # Step 0: Argument validation
    # ---------------------------
    # Load input and output filenames
    fname_labels = arguments.l
    fname_metrics = get_absolute_path(arguments.i)
    fname_metrics_PAM50 = get_absolute_path(arguments.i_PAM50)
    if arguments.o is not None:
        fname_out = arguments.o
    else:
        path, file_name, ext = extract_fname(get_absolute_path(arguments.i))
        fname_out = os.path.join(path, file_name + '_compression_metrics' + ext)
    # Load normalization-related arguments
    metric = 'MEAN(' + arguments.metric + ')'  # Adjust for csv file columns name
    sex = arguments.sex
    age = arguments.age
    if age:
        age.sort()
        if any(n < 0 for n in age):
            parser.error('Age range needs to be positive, {} was specified'.format(age))

    # Step 1. Load subject input files (label image, metric CSVs)
    # -----------------------------------------------------------
    im_levels = Image(fname_labels).change_orientation('RPI')
    slice_thickness = im_levels.dim[5]
    slice_compressed = [int(coord.z) for coord in im_levels.getNonZeroCoordinates(sorting='z')]
    if not slice_compressed:
        raise ValueError('No compression labels found.')
    df_metrics = pd.read_csv(fname_metrics).astype({metric: float})
    df_metrics_PAM50 = pd.read_csv(fname_metrics_PAM50).astype({metric: float})

    # Step 2. Load reference input files (PAM50, healthy controls)
    # ------------------------------------------------------------
    # Get PAM50 slice thickness
    fname_PAM50 = os.path.join(__data_dir__, 'PAM50', 'template', 'PAM50_t2s.nii.gz')
    slice_thickness_PAM50 = Image(fname_PAM50).change_orientation('RPI').dim[5]
    # Get data from healthy control and average them
    path_ref = os.path.join(__data_dir__, 'PAM50_normalized_metrics')
    fname_partcipants = get_absolute_path(os.path.join(path_ref, arguments.file_participants))
    list_HC = select_HC(fname_partcipants, sex, age)
    df_avg_HC = average_HC(path_ref, metric, list_HC)

    # Step 3. Determine compressed levels for both subject and PAM50 space
    # --------------------------------------------------------------------
    # Get vertebral level corresponding to the slice with the compression
    compressed_levels_dict = get_verterbral_level_from_slice(slice_compressed, df_metrics)
    # Get vertebral level above and below the compression
    upper_level, lower_level = get_up_lw_levels(compressed_levels_dict.keys(), df_metrics, metric)
    # Get slices corresponding in PAM50 space
    compressed_levels_dict_PAM50 = get_slices_in_PAM50(compressed_levels_dict, df_metrics, df_metrics_PAM50)

    # Step 4. Compute MSCC metrics for each compressed level
    # ------------------------------------------------------
    for level in compressed_levels_dict_PAM50.keys():
        # Get metric of patient with compression
        ap, slices_avg = average_compression_PAM50(slice_thickness, slice_thickness_PAM50, metric, df_metrics_PAM50,
                                                   upper_level, lower_level, compressed_levels_dict_PAM50[level])
        # Get metrics of healthy controls
        ap_HC = average_metric(df_avg_HC, metric, upper_level, lower_level, slices_avg)
        logger.debug('\nmetric_a_HC =  {}, metric_b_HC = {}, betric_i_HC = {}'.format(ap_HC[0], ap_HC[1], ap_HC[2]))
        logger.debug('metric_a =  {}, metric_b = {}, metric_i = {}'.format(ap[0], ap[1], ap[2]))

        # Compute MSCC
        metric_ratio_norm_result = metric_ratio_norm(ap, ap_HC)
        metric_ratio_result = metric_ratio(ap[0], ap[1], ap[2])
        save_csv(fname_out, level, arguments.metric, metric_ratio_result, metric_ratio_norm_result, arguments.i)

        # Display results
        printv('\nLevel: {}'.format(level), verbose=verbose, type='info')
        printv('\n{} ratio norm = {}'.format(metric, metric_ratio_norm_result), verbose=verbose, type='info')
        printv('\n{} ratio = {}\n'.format(metric, metric_ratio_result), verbose=verbose, type='info')

    printv(f'Saved: {fname_out}')


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
