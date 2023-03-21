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
from spinalcordtoolbox.utils.fs import get_absolute_path, check_file_exist, extract_fname, printv
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.centerline.core import get_centerline, ParamCenterline
from spinalcordtoolbox.types import Centerline
from spinalcordtoolbox import __data_dir__


logger = logging.getLogger(__name__)


NEAR_ZERO_THRESHOLD = 1e-6


# PARSER
# ==========================================================================================
def get_parser():
    parser = SCTArgumentParser(
        description='Compute spinal cord compression morphometrics such as the Maximum Spinal Cord Compression [1], '
                    'the antero-posterior (AP) diameter, or other relevant spinal cord morphometrics that are output'
                    ' by the function sct_process_segmentation (CSA, RL diameter, eccentricity, solidity, etc.).'
                    ' Metrics are normalized using a database of spinal cord morphometrics built from healthy control'
                    ' subjects. This database uses the PAM50 template as an anatomical reference system.'
                    ' \nEquation:    ratio = (1 - mi/((ma+mb)/2))\n'
                    'mi: metric at compression level. ma: metric of level above compression level. mb: metric of level below compression level.'
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
        '-i-PAM50',
        metavar=Metavar.file,
        help='CSV morphometric file in the PAM50 space, obtained by running: '
        'sct_process_segmentation -normalize-PAM50.'
    )
    optional.add_argument(
        '-s',
        metavar=Metavar.file,
        help='NIfTI file of spinal cord segmentation.'
    )
    optional.add_argument(
        '-extent',
        type=float,
        metavar=Metavar.float,
        default=20.0,
        help='Extent (in mm) to average metrics of healthy levels in superior-inferior direction.'
    )
    optional.add_argument(
        '-distance',
        type=float,
        metavar=Metavar.float,
        default=10.0,
        help='Distance (in mm) in the superior-inferior direction from the compression to average healthy slices.'
    )
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
    printv(f'{len(list_to_include)} healthy controls are used for normalization')
    return list(list_to_include)


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


def csv2dataFrame(filename, metric):
    """
    Loads a .csv file and builds a pandas dataFrame of the data
    :param filename: str: filename of the .csv file
    :return data :pd.DataFrame: pandas dataframe of the .csv file's data
    """
    check_file_exist(filename, verbose=0)
    data = pd.read_csv(filename)
    # Ensure that the chosen metric is in float
    data.astype({metric: float})
    return data


def get_slice_thickness(img):
    """
    Get slice thickness from the input image.
    :param img: Image: source image
    :return float: slice thickness in mm.
    """
    return img.dim[5]


def get_compressed_slice(img, verbose):
    """
    Get all the compression labels (voxels of value: '1') that are contained in the input image.
    :param img: Image: source image
    :return list: list of slices number
    """
    # Get all coordinates
    coordinates = img.getNonZeroCoordinates(sorting='z')
    logger.debug('Compression labels coordinates: {}'.format(coordinates))
    # Check it coordinates is empty
    if not coordinates:
        raise ValueError('No compression labels found.')
    # Return only slices number
    return [int(coordinate.z) for coordinate in coordinates]


def get_centerline_object(im_seg, verbose):
    """
    Get centerline object in physical dimensions
    """
    # Compute spinalcordtoolbox.types.Centerline class from get_centerline wit physical coordinates
    param_centerline = ParamCenterline(
                       algo_fitting='bspline',  # TODO add as default arg
                       smooth=30,  # TODO add as default arg
                       minmax=True)  # Check if we want min max or not

    _, arr_ctl_phys, arr_ctl_der_phys, _ = get_centerline(im_seg, param_centerline,
                                                          verbose=verbose, space="phys")
    ctl_seg = Centerline(*arr_ctl_phys, *arr_ctl_der_phys)
    return ctl_seg


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
        raise ValueError('Slice {} doesn\'t have a computed metric'.format(slices))
    level_slice_dict = {}
    for level in np.unique(level_compression['VertLevel']):
        level_slice_dict[level] = level_compression.loc[level_compression['VertLevel'] == level, 'Slice (I->S)'].to_list()
    return level_slice_dict


def average_compression_PAM50(slice_thickness, slice_thickness_PAM50, metric, df_metrics_PAM50, z_range_above, z_range_below, slice):
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
    return get_mean_metric(df_metrics_PAM50, metric, z_range_above, z_range_below, slices_avg), slices_avg


def average_hc(ref_folder, metric, list_HC):
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
            if list_HC:
                # Check if subject is in list to include
                if subject in list_HC:
                    d[file] = csv2dataFrame(os.path.join(ref_folder, file), metric)  # TODO change verbose for arg
                    i = i+1
            else:
                d[file] = csv2dataFrame(os.path.join(ref_folder, file), metric)  # TODO change verbose for arg
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


def get_mean_metric(df, metric, z_range_above, z_range_below, slices_avg):
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
    idx_compression = df['Slice (I->S)'].isin(slices_avg).tolist()
    idx_above = df['Slice (I->S)'].isin(z_range_above).tolist()
    idx_below = df['Slice (I->S)'].isin(z_range_below).tolist()
    ma = df.loc[idx_above, metric].mean()
    mb = df.loc[idx_below, metric].mean()
    mi = df.loc[idx_compression, metric].mean()
    return ma, mb, mi


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


def get_slices_upper_lower_level_from_centerline(centerline, distance, extent, z_compressions, z_ref):
    """
    Get slices to average for the level above the highest compression and below the lowest compression from the centerline.
    (If arg i-PAM50 is not used; meaning no normalization)
    : param centerline: Centerline(): Spinal cord centerline object
    : param distance: float: distance (mm) from the compression from where to average healthy slices.
    : param extent: float: extent (mm) to average healthy slices.
    : param z_compressions: list: list of slice that have a compression.
    : param z_ref: list: z index corresponding to the segmentation since the centerline only includes slices of the segmentation.
    : return
    """
    length = centerline.incremental_length_inverse
    # Get z index of lowest (min) and highest (max) compression
    z_compression_below = min(z_compressions)
    z_compression_above = max(z_compressions)
    # Get slices range for level below lowest compression
    idx = np.argwhere(z_ref == z_compression_below)[0][0]
    length_0 = length[idx]
    zmax_below = z_ref[np.argmin(np.array([np.abs(i - length_0 + distance) for i in length]))]
    zmin_below = z_ref[np.argmin(np.array([np.abs(i - length_0 + distance + extent) for i in length]))]

    # Get slices range for level above aboveest compression
    idx = np.argwhere(z_ref == z_compression_above)[0][0]
    length_0 = length[idx]
    zmin_above = z_ref[np.argmin(np.array([np.abs(i - length_0 - distance) for i in length]))]
    zmax_above = z_ref[np.argmin(np.array([np.abs(i - length_0 - distance - extent) for i in length]))]
    print(zmin_below, zmax_below)
    # If zmin is equal to zmax, the range is not available, use the other level above/below
    if zmin_above == zmax_above:
        zmin_above = zmin_below
        zmax_above = zmax_below
    if zmin_below == zmax_below:
        zmin_below = zmin_above
        zmax_below = zmax_above
    if zmin_above == zmax_above and zmin_below == zmax_below:
        raise ValueError("No slices of level above of below with a distance of "
                         + str(distance) + " mm and extent of " + str(extent)
                         + " mm. Please provide another distance and extent.")
    slices_above = np.arange(zmin_above, zmax_above, 1)
    slices_below = np.arange(zmin_below, zmax_below, 1)
    print(slices_above, slices_below)
    return slices_below, slices_above


def get_slices_upper_lower_level_from_PAM50(compression_level_dict_PAM50, df_metrics_PAM50, distance, extent, slice_thickness_PAM50):
    """
    Get slices to average for the level above the highest compression and below the lowest compression from the PAM50.
    : param compression_level_dict_PAM50: dict: Dictionary of levels and corresponding slice(s) in the PAM50 space.
    : param df_metrics_PAM50: pandas.DataFrame: Metrics output of sct_process_segmentation in PAM50 anatomical dimensions.
    : param distance: float: distance (mm) from the compression from where to average healthy slices.
    : param extent: float: extent (mm) to average healthy slices.
    : param slice_thickness_PAM50: float: Slice thickness of the PAM50.
    : return slices_below:
    : return slices_above:
    """
    level_above = min([level for level, slices in compression_level_dict_PAM50.items()])
    level_below = max([level for level, slices in compression_level_dict_PAM50.items()])
    # Get slices to average at distance across the chosen extent for the aboveest level
    zmin_above = int(max(compression_level_dict_PAM50[level_above]) + distance/slice_thickness_PAM50)
    zmax_above = int(max(compression_level_dict_PAM50[level_above]) + distance/slice_thickness_PAM50 + extent/slice_thickness_PAM50)
    # Get slices to average at distance across the chosen extent for the lowest level
    zmin_below = int(min(compression_level_dict_PAM50[level_below]) - distance/slice_thickness_PAM50 - extent/slice_thickness_PAM50)
    zmax_below = int(min(compression_level_dict_PAM50[level_below]) - distance/slice_thickness_PAM50)
    # Check if slices have available metrics
    df_metrics_PAM50_short = df_metrics_PAM50.drop(columns=['DistancePMJ', 'SUM(length)'])
    df_metrics_PAM50_short.dropna(inplace=True)
    # If above/below not available, only take the level below/above
    if zmax_below not in df_metrics_PAM50_short['Slice (I->S)'].to_list():
        zmax_below = zmax_above
        zmin_below = zmin_above
        not_below = True
    if zmin_above not in df_metrics_PAM50_short['Slice (I->S)'].to_list():
        zmax_above = zmax_below
        zmin_above = zmin_below
        not_above = True
    if not_above and not_below:
        raise ValueError("No metrics of level above and below all compressions are available with a distance of "
                         + str(distance) + " mm and extent of " + str(extent)
                         + " mm. Please provide another distance and extent.")
    # Take last available slice if extent is out of range
    if zmin_below not in df_metrics_PAM50_short['Slice (I->S)'].to_list():
        zmin_below = min(df_metrics_PAM50_short['Slice (I->S)'].to_list())
    if zmax_above not in df_metrics_PAM50_short['Slice (I->S)'].to_list():
        zmin_below = min(df_metrics_PAM50_short['Slice (I->S)'].to_list())
    slices_above = np.arange(zmin_above, zmax_above, 1)
    slices_below = np.arange(zmin_below, zmax_below, 1)
    return slices_below, slices_above


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
        spamwriter = csv.writer(csvfile, delimiter=',')
        line = [filename, level, metric_ratio, metric_ratio_nrom]
        spamwriter.writerow(line)


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)    # values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG]
    fname_labels = arguments.l
    img = Image(fname_labels)
    img.change_orientation('RPI')
    path_ref = os.path.join(__data_dir__, 'PAM50_normalized_metrics')
    if arguments.o is not None:
        fname_out = arguments.o
    else:
        path, file_name, ext = extract_fname(get_absolute_path(arguments.i))
        fname_out = os.path.join(path, file_name + '_compression_metrics' + ext)
    fname_metrics = get_absolute_path(arguments.i)
    metric = 'MEAN(' + arguments.metric + ')'  # Adjust for csv file columns name
    # Fetch distance and extent and segmentation
    fname_segmentation = arguments.s
    distance = arguments.distance
    extent = arguments.extent
    # Fetch metrics of subject
    df_metrics = csv2dataFrame(fname_metrics, metric)
    # Get vertebral level corresponding to the slice with the compression
    slice_thickness = get_slice_thickness(img)
    slice_compressed = get_compressed_slice(img, verbose)
    compressed_levels_dict = get_verterbral_level_from_slice(slice_compressed, df_metrics)
    # Initialize variables if normalization with
    if arguments.i_PAM50:
        fname_metrics_PAM50 = get_absolute_path(arguments.i_PAM50)
        sex = arguments.sex
        age = arguments.age
        if age:
            if any(n < 0 for n in age):
                parser.error('Age range needs to be positive, {} was specified'.format(age))
            # Put age range in order
            else:
                age.sort()
        if sex or age:
            if not os.path.isfile(get_absolute_path(os.path.join(path_ref, arguments.file_participants))):
                raise FileNotFoundError('participants.tsv file must exists to select sex or age.')
            else:
                fname_partcipants = get_absolute_path(os.path.join(path_ref, arguments.file_participants))
                list_HC = select_HC(fname_partcipants, sex, age)
        else:
            list_HC = None
        # Select healthy controls based on sex and/or age range

        # Get PAM50 slice thickness
        fname_PAM50 = os.path.join(__data_dir__, 'PAM50', 'template', 'PAM50_t2.nii.gz')
        img_pam50 = Image(fname_PAM50)
        img_pam50.change_orientation('RPI')
        slice_thickness_PAM50 = get_slice_thickness(img_pam50)
        # Fetch metrics of PAM50 template
        df_metrics_PAM50 = csv2dataFrame(fname_metrics_PAM50, metric)

        # Get slices corresponding in PAM50 space
        compressed_levels_dict = get_slices_in_PAM50(compressed_levels_dict, df_metrics, df_metrics_PAM50)
        z_range_below, z_range_above = get_slices_upper_lower_level_from_PAM50(compressed_levels_dict, df_metrics_PAM50, distance, extent, slice_thickness_PAM50)
        # Get data from healthy control and average them
        df_avg_HC = average_hc(path_ref, metric, list_HC)
    else:
        # Get spinal cord centerline object
        im_seg = Image(get_absolute_path(fname_segmentation)).change_orientation('RPI')
        # Get max and min index of the segmentation with pmj
        _, _, Z = (im_seg.data > NEAR_ZERO_THRESHOLD).nonzero()
        min_z_index, max_z_index = min(Z), max(Z)
        # Get the z index corresponding to the segmentation since the centerline only includes slices of the segmentation.
        z_ref = np.array(range(min_z_index.astype(int), max_z_index.max().astype(int) + 1))

        centerline = get_centerline_object(im_seg, verbose=verbose)
        z_range_above, z_range_below = get_slices_upper_lower_level_from_centerline(centerline, distance, extent, slice_compressed, z_ref)

    # Loop through all compressed levels (compute one MSCC per compressed level)
    for level in compressed_levels_dict.keys():
        # Get metric of patient with compression
        if arguments.i_PAM50:
            ap, slices_avg = average_compression_PAM50(slice_thickness, slice_thickness_PAM50, metric, df_metrics_PAM50,
                                                       z_range_above, z_range_below, compressed_levels_dict[level])
            # Get metrics of healthy controls
            ap_HC = get_mean_metric(df_avg_HC, metric, z_range_above, z_range_below, slices_avg)
            logger.debug('\nmetric_a_HC =  {}, metric_b_HC = {}, metric_i_HC = {}'.format(ap_HC[0], ap_HC[1], ap_HC[2]))
            # Compute Normalized Ratio
            metric_ratio_norm_result = metric_ratio_norm(ap, ap_HC)
        else:
            slices_avg = compressed_levels_dict[level]
            ap = get_mean_metric(df_metrics, metric, z_range_above, z_range_below, slices_avg)
            metric_ratio_norm_result = None
        logger.debug('metric_a =  {}, metric_b = {}, metric_i = {}'.format(ap[0], ap[1], ap[2]))
        # Compute Ratio
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
