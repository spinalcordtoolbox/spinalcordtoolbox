#!/usr/bin/env python
#
# Compute adapted Spinal Cord Occupation Ratio (aSCOR) from spinal cord and spinal canal masks.
# morphometrics.
#
# Copyright (c) 2025 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import sys
import os
import textwrap
import logging
from typing import Sequence
from spinalcordtoolbox.utils.fs import get_absolute_path, TempFolder
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel
from spinalcordtoolbox.utils.shell import display_open, Metavar
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils.sys import LazyLoader
from spinalcordtoolbox.scripts import sct_process_segmentation

pd = LazyLoader("pd", globals(), "pandas")

logger = logging.getLogger(__name__)

INDEX_COLUMNS = ['Filename SC', 'Filename canal', 'Slice (I->S)', 'VertLevel', 'DistancePMJ', 'aSCOR']


# PARSER
# ==========================================================================================


def get_parser():
    """
    :return: Returns the parser with the command line documentation contained in it.
    """
    # Because `sct_compute_ascor` is a light wrapper for `sct_process_segmentation`, we want to re-use most of the arguments.
    parser = sct_process_segmentation.get_parser(ascor=True)
    parser.description = textwrap.dedent("""
        Compute adapted Spinal Cord Occupation Ratio (aSCOR) from spinal cord and spinal canal segmentation masks.
        The aSCOR is defined as the ratio between the cross-sectional area of the spinal cord and the cross-sectional area of the spinal canal
    """)  # noqa: E501 (line too long)

    # In `sct_process_segmentation`, we normally use the `-i` argument. But because we set `ascor=True` when calling
    # `get_parser()`, `-i` won't be created (to allow us to add `-i-SC` and `-i-canal` instead).
    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        '-i-SC',
        metavar=Metavar.file,
        help=textwrap.dedent("""
            Spinal cord segmentation mask to compute morphometrics from. Example: `sub-001_T2w_seg.nii.gz`
            Spinal cord segmentation can be obtained with `sct_deepseg spinalcord`.
        """),  # noqa: E501 (line too long)
    )
    mandatory.add_argument(
        '-i-canal',
        metavar=Metavar.file,
        help=textwrap.dedent("""
            Spinal canal segmentation mask to compute morphometrics from. Example: `sub-001_T2w_canal_seg.nii.gz`
            Canal segmentation can be obtained with `sct_deepseg sc_canal_t2`.
        """),  # noqa: E501 (line too long)
    )

    return parser


def compute_ascor(csa_sc, csa_canal):
    """
    Computes the aSCOR (spinal cord area to canal area ratio) for each row in the provided CSV files and saves the results to a CSV file.

    :param str csa_sc: Path to the CSV file containing spinal cord area measurements.
    :param str csa_canal: Path to the CSV file containing spinal canal area measurements.
    :return pandas.DataFrame: DataFrame containing the aSCOR results."""
    df_sc = pd.read_csv(csa_sc)
    df_canal = pd.read_csv(csa_canal)
    df_ascor = pd.DataFrame()
    # Loop across rows in dataframe
    for idx in range(len(df_sc)):
        ascor_value = df_sc['MEAN(area)'].iloc[idx] / df_canal['MEAN(area)'].iloc[idx]
        row = [df_sc['Filename'].iloc[idx],
               df_canal['Filename'].iloc[idx],
               df_sc['Slice (I->S)'].iloc[idx],
               df_sc['VertLevel'].iloc[idx],
               df_sc['DistancePMJ'].iloc[idx],
               ascor_value]
        df_ascor = pd.concat([df_ascor, pd.DataFrame([row], columns=INDEX_COLUMNS)], ignore_index=True)
    printv(f"Computed aSCOR for {len(df_ascor)} rows.", 1, 'normal')
    # Save to csv
    return df_ascor


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)    # values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG]

    # Load input and output filenames (while passing the remaining args to `sct_process_segmentation`)
    fname_sc_segmentation = get_absolute_path(arguments.i_SC)
    fname_canal_segmentation = get_absolute_path(arguments.i_canal)
    fname_out = arguments.o
    process_seg_argv = [a for i, a in enumerate(argv) if argv[i-1] not in ['-i-SC', '-o', '-i-canal'] and a not in ['-i-SC', '-o', '-i-canal']]

    # Validate the inputs
    img_sc = Image(fname_sc_segmentation).change_orientation('RPI')
    img_canal = Image(fname_canal_segmentation).change_orientation('RPI')
    if not img_sc.data.shape == img_canal.data.shape:
        raise ValueError(f"Shape mismatch between spinal cord segmentation [{img_sc.data.shape}],"
                         f" and canal segmentation [{img_canal.data.shape}]). "
                         f"Please verify that your spinal cord and canal segmentations were done in the same space.")

    # Run sct_process_segmentation twice: 1) SC seg 2) canal seg
    temp_folder = TempFolder(basename="process-segmentation")
    path_tmp = temp_folder.get_path()
    printv("Running sct_process_segmentation on spinal cord segmentation...", verbose, 'normal')
    sct_process_segmentation.main(
                                  ['-i', fname_sc_segmentation,
                                   '-o', os.path.join(path_tmp, "sc.csv"),
                                   ] + process_seg_argv)
    printv("Running sct_process_segmentation on spinal canal segmentation...", verbose, 'normal')
    sct_process_segmentation.main(
                                  ['-i', fname_canal_segmentation,
                                   '-o', os.path.join(path_tmp, "canal.csv"),
                                   ] + process_seg_argv)

    # Compute aSCOR
    printv("Computing aSCOR...", verbose, 'normal')
    df_ascor = compute_ascor(os.path.join(path_tmp, "sc.csv"), os.path.join(path_tmp, "canal.csv"))
    # Save aSCOR to csv
    if arguments.append and os.path.exists(fname_out):
        dataframe_old = pd.read_csv(fname_out, index_col=INDEX_COLUMNS)
        df_ascor = pd.concat([dataframe_old.reset_index(), df_ascor], ignore_index=True)
    df_ascor.to_csv(fname_out, index=False)
    printv(f'\nSaved: {os.path.abspath(fname_out)}')
    display_open(os.path.abspath(fname_out))

    # Clean up temp
    if arguments.r and temp_folder is not None:
        logger.info("\nRemove temporary files...")
        temp_folder.cleanup()


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
