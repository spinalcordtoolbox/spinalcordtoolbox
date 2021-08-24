#!/usr/bin/env python
#
# Script to perform manual correction of segmentations and vertebral labeling.
#
# Authors: Jan Valosek, Julien Cohen-Adad and Sandrine Bédard

import glob
import json
import logging
import os
import sys
import shutil
from textwrap import dedent
import time
import yaml
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar
from spinalcordtoolbox.utils import fs
from spinalcordtoolbox.image import add_suffix, remove_suffix

# TODO: adapt for GM seg


def get_parser():
    """
    parser function
    """
    parser = SCTArgumentParser(
        description='Manual correction of spinal cord segmentation, vertebral and pontomedullary junction labeling. '
                    'Manually corrected files are saved under path-segmanual/ folder.',
        prog=os.path.basename(__file__).strip('.py')
    )
    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        '-config',
        metavar=Metavar.file,
        required=True,
        help=
        "R|Config yaml file listing images that require manual corrections for segmentation and vertebral "
        "labeling. 'FILES_SEG' lists images associated with spinal cord segmentation "
        ",'FILES_LABEL' lists images associated with vertebral labeling "
        "and 'FILES_PMJ' lists images associated with pontomedullary junction labeling"
        "You can validate your .yml file at this website: http://www.yamllint.com/."
        " If you want to correct segmentation only, ommit 'FILES_LABEL' in the list. Below is an example .yml file:\n"
        + dedent(
            """
            FILES_SEG:
            - sub-1000032_T1w.nii.gz
            - sub-1000083_T2w.nii.gz
            FILES_LABEL:
            - sub-1000032_T1w.nii.gz
            - sub-1000710_T1w.nii.gz
            FILES_PMJ:
            - sub-1000032_T1w.nii.gz
            - sub-1000710_T1w.nii.gz\n
            """)
    )
    mandatory.add_argument(
        '-path-in',
        metavar=Metavar.folder,
        required=True,
        help='Path to the processed data. Example: ~/ukbiobank_results/data_processed',
        default='./'
    )
    optional = parser.add_argument_group("OPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")
    optional.add_argument(
        '-path-out',
        metavar=Metavar.folder,
        help="Path to the BIDS dataset where the corrected labels will be generated. Note: if the derivatives/ folder "
             "does not already exist, it will be created."
             "Example: ~/data-ukbiobank",
        default='./')
    optional.add_argument(
        '-path-segmanual',
        metavar=Metavar.folder,
        default='./derivatives/labels'
    )
    optional.add_argument(
        '-software',
        default='fsleyes',
        metavar=Metavar.str,
        choices=['fsleyes', 'itksnap']
    )
    optional.add_argument(
        '-qc-only',
        help="Only output QC report based on the manually-corrected files already present in the derivatives folder. "
             "Skip the copy of the source files, and the opening of the manual correction pop-up windows.",
        action='store_true'
    )
    optional.add_argument(  # TODO: to remove
        '-add-seg-only',
        help="Only copy the source files (segmentation) that aren't in -config list to the derivatives/ folder. "
             "Use this flag to add manually QC-ed automatic segmentations to the derivatives folder.",
        action='store_true'
    )
    optional.add_argument(
        '-v', '--verbose',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode")

    return parser


def get_function(task):
    if task == 'FILES_SEG':
        return 'sct_deepseg_sc'
    elif task == 'FILES_LABEL':
        return 'sct_label_utils'
    elif task == 'FILES_PMJ':
        return 'sct_detect_pmj'
    else:
        raise ValueError("This task is not recognized: {}".format(task))


def get_suffix(task, suffix=''):
    if task == 'FILES_SEG':
        return '_seg' + suffix
    elif task == 'FILES_LABEL':
        return '_labels' + suffix
    elif task == 'FILES_PMJ':
        return '_pmj' + suffix

    else:
        raise ValueError("This task is not recognized: {}".format(task))


def correct_segmentation(fname, fname_seg_out):
    """
    Copy fname_seg in fname_seg_out, then open ITK-SNAP with fname and fname_seg_out.
    :param fname:
    :param fname_seg:
    :param fname_seg_out:
    :param name_rater:
    :return:
    """
    # launch ITK-SNAP
    # Note: command line differs for macOs/Linux and Windows
    print("In ITK-SNAP, correct the segmentation, then save it with the same name (overwrite).")
    if shutil.which('itksnap') is not None:  # Check if command 'itksnap' exists
        os.system('itksnap -g ' + fname + ' -s ' + fname_seg_out)  # for macOS and Linux
    elif shutil.which('ITK-SNAP') is not None:  # Check if command 'ITK-SNAP' exists
        os.system('ITK-SNAP -g ' + fname + ' -s ' + fname_seg_out)  # For windows
    else:
        sys.exit("ITK-SNAP not found. Please install it before using this program or check if it was added to PATH variable. Exit program.")


def correct_vertebral_labeling(fname, fname_label):
    """
    Open sct_label_utils to manually label vertebral levels.
    :param fname:
    :param fname_label:
    :param name_rater:
    :return:
    """
    message = "Place labels at the posterior tip of each inter-vertebral disc. E.g. Label 3: C2/C3, Label 4: C3/C4, etc., then click 'Save and Quit'." 
    os.system('sct_label_utils -i {} -create-viewer 2:23 -o {} -msg "{}"'.format(fname, fname_label, message))


def correct_pmj_label(fname, fname_label):
    """
    Open sct_label_utils to manually label PMJ.
    :param fname:
    :param fname_label:
    :param name_rater:
    :return:
    """
    message = "Click at the posterior tip of the pontomedullary junction (PMJ) then click 'Save and Quit'."
    os.system('sct_label_utils -i {} -create-viewer 50 -o {} -msg "{}"'.format(fname, fname_label, message))


def create_json(fname_nifti, name_rater):
    """
    Create json sidecar with meta information
    :param fname_nifti: str: File name of the nifti image to associate with the json sidecar
    :param name_rater: str: Name of the expert rater
    :return:
    """
    metadata = {'Author': name_rater, 'Date': time.strftime('%Y-%m-%d %H:%M:%S')}
    fname_json = fname_nifti.rstrip('.nii').rstrip('.nii.gz') + '.json'
    with open(fname_json, 'w') as outfile:
        json.dump(metadata, outfile, indent=4)


def check_files_exist(dict_files):  # TODO: move to utils.fs
    """
    Check if all files listed in the input dictionary exist
    :param dict_files:
    :param path_data: folder where BIDS dataset is located
    :return:
    """
    missing_files = []
    for task, files in dict_files.items():
        if files is not None:
            for file in files:
                if not os.path.exists(file):
                    missing_files.append(file)
    if missing_files:
        logging.error("The following files are missing: \n{}. \nPlease check that the files listed "
                        "in the yaml file and the input path are correct.".format(missing_files))


def check_output_folder(path_manual):  #TODO: move to utils.fs
    """
    Make sure path exists, has writing permissions.
    :param path_bids:
    :return: path_bids_derivatives
    """
    
    #if not os.path.exists(path_manual):
    #    logging.error("Output path does not exist: {}".format(path_manual))
    os.makedirs(path_manual, exist_ok=True)


def main():

    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Logging level
    #if args.verbose:
    #    coloredlogs.install(fmt='%(message)s', level='DEBUG')
    #else:
    #    coloredlogs.install(fmt='%(message)s', level='INFO')

    # check if input yml file exists
    fs.check_file_exist(args.config)
    fname_yml = args.config

    # fetch input yml file as dict
    with open(fname_yml, 'r') as stream:
        try:
            dict_yml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Curate dict_yml to only have filenames instead of absolute path
    #dict_yml = utils.curate_dict_yml(dict_yml)

    # check for missing files before starting the whole process
    check_files_exist(dict_yml)

    # check that output folder exists and has write permission
    check_output_folder(args.path_segmanual)
    path_out = args.path_segmanual
    # Get name of expert rater (skip if -qc-only is true)
    if not args.qc_only:
        name_rater = input("Enter your name (Firstname Lastname). It will be used to generate a json sidecar with each "
                           "corrected file: ")

    # Build QC report folder name
    fname_qc = 'qc_corr_' + time.strftime('%Y%m%d%H%M%S')

    # Get list of segmentations files for all subjects in -path-in (if -add-seg-only)
    if args.add_seg_only:
        path_list = glob.glob(args.path_in + "/**/*_seg.nii.gz", recursive=True)  # TODO: add other extension
        # Get only filenames without suffix _seg  to match files in -config .yml list
        file_list = [remove_suffix(os.path.split(path)[-1], '_seg') for path in path_list]

    # TODO: address "none" issue if no file present under a key
    # Perform manual corrections
    for task, files in dict_yml.items():
        # Get the list of segmentation files to add to derivatives, excluding the manually corrrected files in -config.
        if args.add_seg_only and task == 'FILES_SEG':
            # Remove the files in the -config list
            for file in files:
                if file in file_list:
                    file_list.remove(file)
            files = file_list  # Rename to use those files instead of the ones to exclude
        if files is not None:
            for file in files:
                # build file names
                subject = file.split('_')[0]
                contrast = utils.get_contrast(file)
                fname = os.path.join(args.path_in, subject, contrast, file)
                fname_label = os.path.join(
                    path_out, subject, contrast, add_suffix(file, get_suffix(task, '-manual')))
                os.makedirs(os.path.join(path_out, subject, contrast), exist_ok=True)
                if not args.qc_only:
                    if os.path.isfile(fname_label):
                        # if corrected file already exists, asks user if they want to overwrite it
                        answer = None
                        while answer not in ("y", "n"):
                            answer = input("WARNING! The file {} already exists. "
                                           "Would you like to modify it? [y/n] ".format(fname_label))
                            if answer == "y":
                                do_labeling = True
                                overwrite = False
                            elif answer == "n":
                                do_labeling = False
                            else:
                                print("Please answer with 'y' or 'n'")
                    else:
                        do_labeling = True
                        overwrite = True
                    # Perform labeling for the specific task
                    if do_labeling:
                        if task in ['FILES_SEG']:
                            fname_seg = add_suffix(fname, get_suffix(task))
                            if overwrite:
                                shutil.copyfile(fname_seg, fname_label)
                            if not args.add_seg_only:
                                correct_segmentation(fname, fname_label)
                        elif task == 'FILES_LABEL':
                            correct_vertebral_labeling(fname, fname_label)
                        elif task == 'FILES_PMJ':
                            correct_pmj_label(fname, fname_label)
                        else:
                            sys.exit('Task not recognized from yml file: {}'.format(task))
                        # create json sidecar with the name of the expert rater
                        create_json(fname_label, name_rater)

                # generate QC report (only for vertebral labeling or for qc only)
                if args.qc_only:  #or task != 'FILES_SEG':
                    os.system('sct_qc -i {} -s {} -p {} -qc {} -qc-subject {}'.format(
                        fname, fname_label, get_function(task), fname_qc, subject))
                    # Archive QC folder
                    shutil.copy(fname_yml, fname_qc)
                    shutil.make_archive(fname_qc, 'zip', fname_qc)
                    print("Archive created:\n--> {}".format(fname_qc+'.zip'))


if __name__ == '__main__':
    main()
