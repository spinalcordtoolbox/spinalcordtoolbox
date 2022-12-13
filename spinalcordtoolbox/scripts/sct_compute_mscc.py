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
        help='.txt file with slice (in RPI) of compressed level'
    )
    mandatoryArguments.add_argument(
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
    # TODO add option for multiple compressed levels/ slices
    with open(fname_slice) as f:
        slice = f.read()
    print('slice', slice)
    return int(slice)


def get_verterbral_level_from_slice(slice, df_metrics):
    # TODO add if multiple levels or multiple slices
    level_compression = df_metrics.loc[df_metrics['Slice (I->S)']==slice, 'VertLevel'].to_list()
    print('Compressed level', level_compression[0])
    return level_compression[0]


def average_compression_PAM50(slice_thickness, df_metrics_PAM50, level, slice):
    # slice 759
    # TODO average on equ slice
    upper_level = level - 1 # TODO change if more than one level
    lower_level = level + 1
    nb_slice = slice_thickness//0.5
    if nb_slice > 1:
        slices_avg = np.arange(slice - nb_slice//2, slice + nb_slice//2,1)
        print(slices_avg)
    else:
        slices_avg = [slice]
    # find index of slices to average    
    idx = df_metrics_PAM50['Slice (I->S)'].isin(slices_avg).tolist()
    # TODO: function for 3 next rows
    upper_AP_mean = df_metrics_PAM50.loc[df_metrics_PAM50['VertLevel']==upper_level, 'MEAN(diameter_AP)'].mean()
    lower_AP_mean = df_metrics_PAM50.loc[df_metrics_PAM50['VertLevel']==lower_level, 'MEAN(diameter_AP)'].mean()
    compressed_AP_mean = df_metrics_PAM50.loc[idx, 'MEAN(diameter_AP)'].mean()
    return upper_AP_mean, lower_AP_mean, compressed_AP_mean
    # How to get equivalent in PAM50 space ??


def average_hc(ref_folder, level, slices_avg):
    upper_level = level - 1 # TODO change if more than one level
    lower_level = level + 1
    df = pd.DataFrame()
    d = {}
    d = {}
    i = 0
    for file in os.listdir(ref_folder):
        if 'PAM50' in file:
            print(os.path.join(ref_folder,file))
            d[file] = csv2dataFrame(os.path.join(ref_folder,file))
            i = i+1
    first_key = next(iter(d))
    df = pd.DataFrame(columns=d[first_key].columns)
    df['VertLevel'] = d[first_key]['VertLevel']
    df['Slice (I->S)'] = d[first_key]['Slice (I->S)']
    for key, values in d.items():
        for column in d[key].columns:
            if 'MEAN' in column:
                if df[column].isnull().values.all():
                    df[column]  = d[key][column]
                else:
                    df[column]= df[column] + d[key][column].tolist()
    for column in df.columns:
        if 'MEAN' in column:
            df[column] = df[column]/i
    upper_AP_mean = df.loc[df['VertLevel']==upper_level, 'MEAN(diameter_AP)'].mean()
    lower_AP_mean = df.loc[df['VertLevel']==lower_level, 'MEAN(diameter_AP)'].mean()
    compressed_AP_mean = df.loc[df['Slice (I->S)']==slices_avg, 'MEAN(diameter_AP)'].mean()
    return upper_AP_mean, lower_AP_mean, compressed_AP_mean



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

    compressed_level = get_verterbral_level_from_slice(slice_compressed, df_metrics)
    ap = average_compression_PAM50(slice_thickness, df_metrics_PAM50, compressed_level, 759)
    ap_HC = average_hc(path_ref, compressed_level, 759)


    print('Upper HC', ap_HC[0], 'Lower HC', ap_HC[1], 'Compressed HC', ap_HC[2])
    print('Upper HC', ap[0], 'Lower HC', ap[1], 'Compressed HC', ap[2])

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
    printv('\nMSCC norm = ' + str(mscc_result_norm) + '\n', verbose, 'info')
    printv('\nMSCC = ' + str(mscc_result) + '\n', verbose, 'info')

if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
