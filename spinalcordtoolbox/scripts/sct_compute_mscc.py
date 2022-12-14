#!/usr/bin/env python
#########################################################################################
#
# Compute maximum spinal cord compression.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Sandrine Bédard, Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sys
import os
import numpy as np
from typing import Sequence
import pandas as pd
from spinalcordtoolbox.utils import SCTArgumentParser, Metavar, init_sct, printv, set_loglevel
from spinalcordtoolbox.utils.fs import get_absolute_path, check_file_exist, extract_fname
from spinalcordtoolbox.image import Image

# PARSER
# ==========================================================================================
def get_parser():
    parser = SCTArgumentParser(
        description='Compute Maximum Spinal Cord Compression (MSCC) as in: Miyanji F, Furlan JC, Aarabi B, Arnold PM, '
                    'Fehlings MG. Acute cervical traumatic spinal cord injury: MR imaging findings correlated with '
                    'neurologic outcome--prospective study with 100 consecutive patients. Radiology 2007;243(3):820-'
                    '827.'
    )

    mandatoryArguments = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatoryArguments.add_argument(
        '-i',
        metavar=Metavar.file,
        required=True,
        help='Input file name (add extension). Example: csa.csv.'
    )
    mandatoryArguments.add_argument(
        '-s',
        metavar=Metavar.file,
        required=True,
        help='Spinal cord segmentation to fetch slice thickness.'
    )
    mandatoryArguments.add_argument(
        '-compression_level',
        metavar=Metavar.file,
        required=True,
        help='.txt file with slice(s) (in RPI) of compressed level'
    )
    mandatoryArguments.add_argument(  # TODO: to remove, fetch dataset, add age, sex, height ...
        '-ref',
        required=True,
        help='Folder with .csv files of HC control to use for normalization.',
        metavar=Metavar.folder,
    )
    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        '-i-PAM50',
        metavar=Metavar.file,
        required=False,
        help='Input file name (add extension). Example: csa_PAM50.csv.'
    )
    optional.add_argument(
        '-o',
        metavar=Metavar.file,
        help="Name of output file. Example: src_reg.nii.gz"
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
    return (1 - float(di) / ((da + db) / float(2))) * 100


def mscc_norm(ap, ap_HC):
    da = ap[0]/ap_HC[0]
    db = ap[1]/ap_HC[1]
    di = ap[2]/ap_HC[2]
    return mscc(da, db, di)


def csv2dataFrame(filename):
    """
    Loads a .csv file and builds a pandas dataFrame of the data
    Args:
        filename (str): filename of the .csv file
    Returns:
        data (pd.dataFrame): pandas dataframe of the .csv file's data
    """
    check_file_exist(filename)
    data = pd.read_csv(filename)
    data.astype({"MEAN(diameter_AP)": float})
    return data


def get_slice_thickness(fname_seg):
    im_seg = Image(fname_seg)
    im_seg.change_orientation('RPI')
    pz = im_seg.dim[5]
    print('slice thickness', pz)
    return pz


def get_compressed_slice(fname_slice):
    with open(fname_slice) as f:
        slices = f.readlines()
    slices = [l.strip('\n\r') for l in slices]
    return [int(i) for i in slices]


def get_verterbral_level_from_slice(slices, df_metrics):
    idx = df_metrics['Slice (I->S)'].isin(slices).tolist()
    level_compression = df_metrics.loc[idx, ['VertLevel', 'Slice (I->S)']]
    level_slice_dict =  {}
    slices = np.array(slices)
    for level in np.unique(level_compression['VertLevel']):
        level_slice_dict[level] = level_compression.loc[level_compression['VertLevel']==level,'Slice (I->S)'].to_list()
    return level_slice_dict


def average_compression_PAM50(slice_thickness, df_metrics_PAM50, upper_level, lower_level, slice):
    # slice 759
    # TODO average on equ slice
    nb_slice = slice_thickness//0.5
    if nb_slice > 1 :
        slices_avg = np.arange(min(slice) - nb_slice//2, max(slice) + nb_slice//2,1)
    if len(slice)>1:
        slices_avg = np.arange(min(slice), max(slice), 1)
    else:
        slices_avg = slice
    # find index of slices to average    
    idx = df_metrics_PAM50['Slice (I->S)'].isin(slices_avg).tolist()
    # TODO: function for 3 next rows
    upper_AP_mean = df_metrics_PAM50.loc[df_metrics_PAM50['VertLevel']==upper_level, 'MEAN(diameter_AP)'].mean()
    lower_AP_mean = df_metrics_PAM50.loc[df_metrics_PAM50['VertLevel']==lower_level, 'MEAN(diameter_AP)'].mean()
    compressed_AP_mean = df_metrics_PAM50.loc[idx, 'MEAN(diameter_AP)'].mean()
    return upper_AP_mean, lower_AP_mean, compressed_AP_mean, slices_avg
    # How to get equivalent in PAM50 space ??


def average_hc(ref_folder, upper_level, lower_level, slices_avg, slice_thickness):
    df = pd.DataFrame()
    d = {}
    i = 0
    # Loop through .csv files of healthy controls
    for file in os.listdir(ref_folder):
        if 'PAM50' in file:
            print(os.path.join(ref_folder,file))
            d[file] = csv2dataFrame(os.path.join(ref_folder,file))
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
                    df[column]  = d[key][column]
                else:
                    # Sum all columns that have MEAN key
                    df[column]= df[column] + d[key][column].tolist()
    # Divide by number of HC
    for column in df.columns:
        if 'MEAN' in column:
            df[column] = df[column]/i

    # find index of slices to average    
    idx = df['Slice (I->S)'].isin(slices_avg).tolist()    
    upper_AP_mean = df.loc[df['VertLevel']==upper_level, 'MEAN(diameter_AP)'].mean()
    lower_AP_mean = df.loc[df['VertLevel']==lower_level, 'MEAN(diameter_AP)'].mean()
    compressed_AP_mean = df.loc[idx, 'MEAN(diameter_AP)'].mean()
    return upper_AP_mean, lower_AP_mean, compressed_AP_mean


def get_up_lw_levels(levels):
    """
    Get must upper level from all compressed levels and lowest level from all compressed levels
    :param: levels: list: Compressed levels.
    :return: upper_level: int: Smallest level (closest to superior)
    :return: lower_level: int: Highest level (closest to inferior)
    """
    upper_level = min(levels) - 1 
    lower_level = max(levels) + 1
    return upper_level, lower_level

def get_slices_in_PAM50(compressed_level_dict, df_metrics, df_metrics_PAM50):
    # TODO maybe use function in metrics to PAM50?
    compression_level_dict_PAM50 = {}
    for level, slices in compressed_level_dict.items():
        nb_slices_level = len(df_metrics.loc[df_metrics['VertLevel']==level, 'VertLevel'].to_list())
        nb_slices_PAM50 = len(df_metrics_PAM50.loc[df_metrics_PAM50['VertLevel']==level, 'VertLevel'].to_list())
        x_PAM50 = np.arange(0, nb_slices_PAM50, 1)
        x = np.linspace(0, nb_slices_PAM50 - 1, nb_slices_level)
        new_slices_coord = np.interp(x_PAM50, x, df_metrics.loc[df_metrics['VertLevel']==level, 'Slice (I->S)'].to_list())
        # find nearest index
        slices_PAM50 = []
        for slice in slices:
            idx = np.abs(new_slices_coord - slice).argmin()
            new_slice= df_metrics_PAM50.loc[df_metrics_PAM50['VertLevel']==level, 'Slice (I->S)'].to_list()[idx]
            slices_PAM50.append(new_slice)
        compression_level_dict_PAM50[level] = slices_PAM50
    return compression_level_dict_PAM50


# MAIN
# ==========================================================================================
def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)

    # Get parser info
    fname_segmentation = get_absolute_path(arguments.s)
    path_ref = get_absolute_path(arguments.ref)
    if arguments.o is not None:
        fname_out = arguments.o
    else:
        path, file_name, ext = extract_fname(get_absolute_path(arguments.compression_level))
        fname_out = file_name + '_mscc' + ext
    # TODO: read slice
    fname_slice_compressed = arguments.compression_level
    fname_metrics = get_absolute_path(arguments.i)
    if arguments.i_PAM50 is None:
        path, file_name, ext = extract_fname(fname_metrics)
        fname_metrics_PAM50 = os.path.join(path, file_name + '_PAM50' + ext)
    else:
        fname_metrics_PAM50 = arguments.i_PAM50

    slice_thickness = get_slice_thickness(fname_segmentation)
    slice_compressed = get_compressed_slice(fname_slice_compressed)

    df_metrics = csv2dataFrame(fname_metrics)
    df_metrics_PAM50 = csv2dataFrame(fname_metrics_PAM50)

    compressed_levels_dict = get_verterbral_level_from_slice(slice_compressed, df_metrics)
    up_level, lw_level = get_up_lw_levels(compressed_levels_dict.keys())
    compressed_levels_dict_PAM50 = get_slices_in_PAM50(compressed_levels_dict, df_metrics, df_metrics_PAM50)

    for level in compressed_levels_dict_PAM50.keys():
        ap = average_compression_PAM50(slice_thickness, df_metrics_PAM50, up_level, lw_level, compressed_levels_dict_PAM50[level])
        slices_avg = ap[3]
        ap_HC = average_hc(path_ref, up_level, lw_level, slices_avg, slice_thickness)
        print('Upper HC', ap_HC[0], 'Lower HC', ap_HC[1], 'Compressed HC', ap_HC[2])
        print('Upper', ap[0], 'Lower', ap[1], 'Compressed', ap[2])

        # Compute MSCC
        mscc_result_norm= mscc_norm(ap, ap_HC)
        mscc_result= mscc(ap[0], ap[1], ap[2])

    # Input 
        #.txt file with compression slice nb
        # .csv file normalized and not normalize
        # slice thickness
        # path to .csv for norm
    # steps:
        # fetch vertebral level of compressed level 
        # average compression on slice thickness eqc
        # Average on entire 3 levels of HC
        # Compute MSCC


    # Display results
    # TODO save in txt file
        printv('Level: ' + str(level)+ '\n', verbose, 'info')
        printv('\nMSCC norm = ' + str(mscc_result_norm) + '\n', verbose, 'info')
        printv('\nMSCC = ' + str(mscc_result) + '\n', verbose, 'info')

if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
