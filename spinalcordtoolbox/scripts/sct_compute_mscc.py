#!/usr/bin/env python
#########################################################################################
#
# Compute maximum spinal cord compression.
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
from spinalcordtoolbox import __data_dir__


logger = logging.getLogger(__name__)


# PARSER
# ==========================================================================================
def get_parser():
    parser = SCTArgumentParser(
        description='Compute Maximum Spinal Cord Compression (MSCC) as in: Miyanji F, Furlan JC, Aarabi B, Arnold PM, '
                    'Fehlings MG. Acute cervical traumatic spinal cord injury: MR imaging findings correlated with '
                    'neurologic outcome--prospective study with 100 consecutive patients. Radiology 2007;243(3):820-'
                    '827.'
        # TODO - update this description
    )

    mandatoryArguments = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatoryArguments.add_argument(
        '-i',
        metavar=Metavar.file,
        required=True,
        help='Input csv file name in native space. Example: csa.csv.'
    )
    mandatoryArguments.add_argument(
        '-l',
        metavar=Metavar.file,
        required=True,
        help='.nii file with compression labels. Each compression is denoted by a single voxel of value `1`. '
             'Example: sub-001_T2w_compression_labels.nii.gz'
    )
    mandatoryArguments.add_argument(  # TODO: to remove, fetch dataset, add age, sex, height ...
        '-ref',
        required=True,
        help='Folder with .csv files (in PAM50 space) of HC control to use for normalization.',
        metavar=Metavar.folder,
    )
    mandatoryArguments.add_argument(
        '-i-PAM50',
        metavar=Metavar.file,
        required=False,
        help='Input file name (add extension). Example: csa_PAM50.csv.'
    )
    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        '-metric',
        required=True,
        help='Metric name to normalize in .csv file output from sct_process_segmentation. Default = MEAN(diameter_AP)',
        default='MEAN(diameter_AP)',
        choices=['MEAN(area)', 'MEAN(diameter_RL)', 'MEAN(eccentricity)', 'MEAN(solidity)'],
        metavar=Metavar.file,
    )
    optional.add_argument(
        '-o',
        metavar=Metavar.file,
        help="Output csv file name. If not provided, _mscc suffix is added to the file name provided by -i flag."
    )
    optional.add_argument(
        '-subject',
        metavar=Metavar.file,
        help="Name of subject. Default: filename of -i"
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


def mscc(da, db, di):
    """
    Compute MSCC (Maximum Spinal Cord Compression) using anterior-posterior (AP) diameter of compression and of levels
    above and bellow.
    :param float: da: diameter of level above compression level.
    :param float: db: diameter of level above compression level.
    :param float: di: diameter at compression level.
    :return float: MSCC in %
    """
    return (1 - float(di) / ((da + db) / float(2))) * 100


def mscc_norm(ap, ap_HC):
    """
    Compute normalized MSCC (Maximum Spinal Cord Compression) using anterior-posterior (AP) diameter of compression and
    of levels above and bellow.
    Each AP diameter is divided by the value of healthy controls.
    :param list: ap: list anterior posterior diameter value of level above, below and at compression of patient.
    :param list: ap_HC: list anterior posterior diameter value of level above, below and at compression of healthy
    controls.
    :return float: MSCC normalized in %
    """
    da = ap[0]/ap_HC[0]
    db = ap[1]/ap_HC[1]
    di = ap[2]/ap_HC[2]
    return mscc(da, db, di)


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
    if verbose == 2:
        print('Compression labels coordinates: {}'.format(coordinates))
    # Check it coordinates is empty
    if not coordinates:
        raise ValueError('No compression labels found.')
    # Return only slices number
    return [int(coordinate.z) for coordinate in coordinates]


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


def average_compression_PAM50(slice_thickness, slice_thickness_PAM50, df_metrics_PAM50, upper_level, lower_level, slice):
    """
    Defines slices to average AP diameter at compression level following slice thickness and averages AP diameter at
    compression, across the entire level above and below compression.
    :param slice_thickness: float: slice thickness of native image space.
    :param slice_thickness_PAM50: float: slice thickness of the PAM50.
    :param df_metrics_PAM50: pandas.DataFrame: Metrics of sct_process_segmentation in PAM50 anatomical dimensions.
    :param upper_level: int: level above compression.
    :param lower_level: int: level below compression.
    :param slice: int: slice of spinal cord compression.
    :return upper_AP_mean:
    :retrun lower_AP_mean:
    :retrun compressed_AP_mean:
    :retrun slices_avg: Slices in PAM50 space to average AP diameter.

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
    return get_mean_AP_diameter(df_metrics_PAM50, upper_level, lower_level, slices_avg), slices_avg


def average_hc(ref_folder, metric, upper_level, lower_level, slices_avg):
    """
    Gets AP diameter of healthy controls in PAM50 anatomical dimensions and avrages across subjects.
    Averages AP diameter at compression, across the entire level above and below compression.
    :param ref_folder: path to folder where .csv fiels of healthy controls are.
    :param upper_level: int: level above compression.
    :param lower_level: int: level below compression.
    :param slices_avg: Slices in PAM50 space to average AP diameter.
    :return: upper_AP_mean
    :retrun: lower_AP_mean
    :retrun: compressed_AP_mean
    """
    # Initialize empty dataframe
    df = pd.DataFrame()
    # Create empty dict to put dataframe of each healthy control
    d = {}
    # Iterator to count number of healthy subjects
    i = 0
    # Loop through .csv files of healthy controls
    for file in os.listdir(ref_folder):
        if 'PAM50' in file:
            d[file] = csv2dataFrame(os.path.join(ref_folder, file), metric)  # TODO change verbose for arg
            i = i+1
    first_key = next(iter(d))
    # Create an empty dataframe with ame columns
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

    return get_mean_AP_diameter(df, upper_level, lower_level, slices_avg)


def get_mean_AP_diameter(df, metric, upper_level, lower_level, slices_avg):
    """
    Average AP diameter at compression level, at level above and below.
    :param df: pandas.DataFrame: Metrics output of sct_process_segmentation.
    :param metric: str: metric to perform normalization
    :param upper_level: int: level above compression.
    :param lower_level: int: level below compression.
    :param slices_avg: Slices in PAM50 space to average AP diameter.
    :return: da: float64: AP diameter above the compression
    :retrun: db: float64: AP diameter below the compression
    :retrun: di: float64: AP diameter at the compression level
    """
    # find index of slices to average
    idx = df['Slice (I->S)'].isin(slices_avg).tolist()
    da = df.loc[df['VertLevel'] == upper_level, metric].mean()
    db = df.loc[df['VertLevel'] == lower_level, metric].mean()
    di = df.loc[idx, metric].mean()
    return da, db, di


def get_up_lw_levels(levels):
    """
    Get most upper level from all compressed levels and lowest level from all compressed levels
    :param levels: list: Compressed levels.
    :return upper_level: int: Smallest level (closest to superior)
    :return lower_level: int: Highest level (closest to inferior)
    """
    upper_level = min(levels) - 1
    lower_level = max(levels) + 1
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


def save_csv(fname_out, level, mscc, mscc_norm, subject):
    """
    Save .csv file of MSCC results.
    :param fname_out:
    :param level: int: Level of compression.
    :param mscc: float:
    :param mscc_norm:
    :param subject: str: subject id
    :retrun:
    """
    if not os.path.isfile(fname_out):
        with open(fname_out, 'w') as csvfile:
            header = ['Subject', 'Compression Level', 'MSCC', 'Normalized MSCC']
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
    with open(fname_out, 'a') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        line = [subject, level, mscc, mscc_norm]
        spamwriter.writerow(line)


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)    # values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG]

    fname_labels = arguments.l

    img = Image(fname_labels)
    img.change_orientation('RPI')

    path_ref = get_absolute_path(arguments.ref)
    if arguments.o is not None:
        fname_out = arguments.o
    else:
        path, file_name, ext = extract_fname(get_absolute_path(arguments.i))
        fname_out = os.path.join(path, file_name + '_mscc' + ext)
    fname_metrics = get_absolute_path(arguments.i)
    if arguments.i_PAM50 is None:
        path, file_name, ext = extract_fname(fname_metrics)
        fname_metrics_PAM50 = os.path.join(path, file_name + '_PAM50' + ext)
    else:
        fname_metrics_PAM50 = arguments.i_PAM50
    if arguments.subject is None:
        subject = arguments.i
    else:
        subject = arguments.subject
    metric = arguments.metric

    slice_thickness = get_slice_thickness(img)
    slice_compressed = get_compressed_slice(img, verbose)

    # Get PAM50 slice thickness
    fname_PAM50 = os.path.join(__data_dir__, 'PAM50', 'template', 'PAM50_t2.nii.gz')
    img_pam50 = Image(fname_PAM50)
    img_pam50.change_orientation('RPI')
    slice_thickness_PAM50 = get_slice_thickness(img_pam50)

    # Fetch metrics of subject
    df_metrics = csv2dataFrame(fname_metrics, metric)
    df_metrics_PAM50 = csv2dataFrame(fname_metrics_PAM50)

    # Get vertebral level corresponding to the slice with the compression
    compressed_levels_dict = get_verterbral_level_from_slice(slice_compressed, df_metrics)
    # Get vertebral level above and below the compression
    upper_level, lower_level = get_up_lw_levels(compressed_levels_dict.keys())
    # Get slices corresponding in PAM50 space
    compressed_levels_dict_PAM50 = get_slices_in_PAM50(compressed_levels_dict, df_metrics, df_metrics_PAM50)

    # Loop through all compressed levels (compute one MSCC per compressed level)
    for level in compressed_levels_dict_PAM50.keys():
        # Get anterior-posterior (AP) diameter of patient with compression
        ap, slices_avg = average_compression_PAM50(slice_thickness, slice_thickness_PAM50,  df_metrics_PAM50,
                                                   upper_level, lower_level, compressed_levels_dict_PAM50[level])
        # Get AP diameter of healthy controls
        ap_HC = average_hc(path_ref, metric, upper_level, lower_level, slices_avg)
        logger.debug('\nda_HC =  {}, db_HC = {}, di_HC = {}'.format(ap_HC[0], ap_HC[1], ap_HC[2]))
        logger.debug('da =  {}, db = {}, di = {}'.format(ap[0], ap[1], ap[2]))

        # Compute MSCC
        mscc_result_norm = mscc_norm(ap, ap_HC)
        mscc_result = mscc(ap[0], ap[1], ap[2])
        save_csv(fname_out, level, mscc_result, mscc_result_norm, subject)

        # Display results
        printv('\nLevel: {}'.format(level), verbose=verbose, type='info')
        printv('\nMSCC norm = {}'.format(mscc_result_norm), verbose=verbose, type='info')
        printv('\nMSCC = {}\n'.format(mscc_result), verbose=verbose, type='info')

    printv(f'Saved: {fname_out}')


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
