#!/usr/bin/env python
"""
This command-line tool is the interface for the deepseg API that performs segmentation using deep learning from the
ivadomed package.
"""

# TODO: Add link to example image so users can decide wether their images look "close enough" to some of the proposed
#  models (e.g., mice, etc.).
# TODO: add test to make sure that all postprocessing flags match the core.DEFAULT dictionary items
# TODO: Fetch default value (and display) depending on the model that is used.
# TODO: accommodate multiclass segmentation

import os
import sys
import logging
from typing import Sequence
from pathlib import Path

from ivadomed import inference as imed_inference
import nibabel as nib
import numpy as np

from spinalcordtoolbox.deepseg import models
from spinalcordtoolbox.image import splitext
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, display_viewer_syntax
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel

logger = logging.getLogger(__name__)


def get_parser():
    parser = SCTArgumentParser(
        description="Segment an anatomical structure or pathologies according to the specified deep learning model."
    )

    input_output = parser.add_argument_group("\nINPUT/OUTPUT")
    input_output.add_argument(
        "-i",
        nargs="+",
        help="Image to segment. Can be multiple images (separated with space).",
        metavar=Metavar.file)
    input_output.add_argument(
        "-c",
        nargs="+",
        help="Type of image contrast. Indicates the order in which the images have been presented with -i. "
             "Optional if only one image is specified with -i. The contrasts should be separated by spaces "
             "(e.g., -c t1 t2).",
        choices=('t1', 't2', 't2star'),
        metavar=Metavar.file)
    input_output.add_argument(
        "-o",
        help="Output file name. In case of multi-class segmentation, class-specific suffixes will be added. By default,"
             "the suffix specified in the packaged model will be added and output extension will be .nii.gz.",
        metavar=Metavar.str)

    seg = parser.add_argument_group('\nTASKS')
    seg.add_argument(
        "-task",
        nargs="+",
        help="Task to perform. It could either be a pre-installed task, task that could be installed, or a custom task."
             " To list available tasks, run: sct_deepseg -list-tasks. To use a custom task, indicate the path to the "
             " ivadomed packaged model (see https://ivadomed.org/en/latest/pretrained_models.html#packaged-model-format for more details). "
             " More than one path can be indicated (separated with space) for cascaded application of the models.",
        metavar=Metavar.str)
    seg.add_argument(
        "-list-tasks",
        action='store_true',
        help="Display a list of tasks that can be achieved.")
    seg.add_argument(
        "-list-tasks-long",
        action='store_true',
        help="Display a list of tasks, along with detailed descriptions (including information on how the model was "
             "trained, what data it was trained on, any performance evaluations, associated papers, etc.)")
    seg.add_argument(
        "-install-task",
        help="Install models that are required for specified task.",
        choices=list(models.TASKS.keys()))

    misc = parser.add_argument_group('\nPARAMETERS')
    misc.add_argument(
        "-thr",
        type=float,
        dest='binarize_prediction',
        help="Binarize segmentation with specified threshold. Set to 0 for no thresholding (i.e., soft segmentation). "
             "Default value is model-specific and was set during optimization "
             "(more info at https://github.com/sct-pipeline/deepseg-threshold).",
        metavar=Metavar.float,
        default=None)
    misc.add_argument(
        "-r",
        type=int,
        help="Remove temporary files.",
        choices=(0, 1),
        default=1)
    misc.add_argument(
        "-largest",
        dest='keep_largest',
        type=int,
        help="Keep the largest connected-objects from the output segmentation. Specify the number of objects to keep."
             "To keep all objects, set to 0",
        default=None)
    misc.add_argument(
        "-fill-holes",
        type=int,
        help="Fill small holes in the segmentation.",
        choices=(0, 1),
        default=None)
    misc.add_argument(
        "-remove-small",
        type=str,
        nargs="+",
        help="Minimal object size to keep with unit (mm3 or vox). A single value can be provided or one value per "
             "prediction class. Single value example: 1mm3, 5vox. Multiple values example: 10 20 10vox (remove objects "
             "smaller than 10 voxels for class 1 and 3, and smaller than 20 voxels for class 2).",
        default=None)

    misc = parser.add_argument_group('\nMISC')
    misc.add_argument(
        '-v',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode")
    misc.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")

    return parser


def segment_and_average_volumes(model_paths, input_filenames, options):
    """
        Run `ivadomed.inference.segment_volume()` once per model, then average the outputs.

        :param model_paths: A list of folder paths. The folders must contain:
            (1) the model ('folder_model/folder_model.pt') to use
            (2) its configuration file ('folder_model/folder_model.json') used for the training,
            see https://github.com/neuropoly/ivadomed/wiki/configuration-file
        :param input_filenames: list of image filenames (e.g. .nii.gz) to segment. Multichannel models require multiple
            images to segment, i.e.., len(fname_images) > 1.
        :param options: A dictionary containing optional configuration settings, as specified by the
            ivadomed.inference.segment_volume function.

        :return: list, list: List of nibabel objects containing the soft segmentation(s), one per prediction class, \
            List of target suffix associated with each prediction
    """
    # Fetch the name of the model (to be used in logging)
    name_model = Path(model_paths[0]).parts[-1]
    logger.info(f"\nRunning inference for model '{name_model}'...")

    # Perform inference once per model
    nii_lsts, target_lsts = [], []
    for path_model in model_paths:
        if len(model_paths) > 1:  # We have an ensemble, so output messages to distinguish between seeds
            name_seed = Path(path_model).parts[-2]
            logger.info(f"\nUsing '{name_seed}'...")
        nii_lst, target_lst = imed_inference.segment_volume(path_model, input_filenames, options=options)
        nii_lsts.append(nii_lst)
        target_lsts.append(target_lst)

    # If we have a single model, skip averaging
    if len(model_paths) == 1:
        nii_lst = nii_lsts[0]
    # Otherwise, we have a model ensemble, so average the image data
    else:
        logger.info(f"\nAveraging outputs across the ensemble for '{name_model}'...")
        nii_lst = []
        # NB: `nii_lsts` is a list of lists, with each sublist being the *per-model* predictions. Example:
        #         [
        #             [m1_prediction_1.nii.gz, m1_prediction_2.nii.gz, ...],  # model 1 predictions
        #             [m2_prediction_1.nii.gz, m2_prediction_2.nii.gz, ...]   # model 2 predictions
        #         ]
        # So, we want to take: the average of "prediction_1", the average of "prediction_2", etc.
        # To do this, we unpack + zip `nii_lists`, so that "prediction_N" files are grouped as "predictions".
        for predictions in zip(*nii_lsts):
            # Average the data for each output in the ensemble
            data_mean = np.mean([pred.get_fdata() for pred in predictions])
            # Take the first image's header to reuse for the averaged image
            nii_header = predictions[0].header
            # Create a new Nifti1Image containing the averaged output
            nii_lst.append(nib.Nifti1Image(data_mean, header=nii_header, affine=nii_header.get_best_affine()))

    # The 'targets' should be identical for each model, so just take the first
    target_lst = target_lsts[0]

    return nii_lst, target_lst


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)

    if (arguments.list_tasks is False and arguments.list_tasks_long is False
            and arguments.install_task is None
            and (arguments.i is None or arguments.task is None)):
        parser.error("You must specify either '-list-tasks', '-list-tasks-long', '-install-task', "
                     "or both '-i' + '-task'.")

    # Deal with task
    if arguments.list_tasks:
        models.display_list_tasks()
    # Deal with task long description
    if arguments.list_tasks_long:
        models.display_list_tasks_long()

    if arguments.install_task is not None:
        for name_model in models.TASKS[arguments.install_task]['models']:
            models.install_model(name_model)
        exit(0)

    # Deal with input/output
    for file in arguments.i:
        if not os.path.isfile(file):
            parser.error("This file does not exist: {}".format(file))

    # Verify if the task is part of the "official" tasks, or if it is pointing to paths containing custom models
    if len(arguments.task) == 1 and arguments.task[0] in models.TASKS:
        # Check if all input images are provided
        required_contrasts = models.get_required_contrasts(arguments.task[0])
        n_contrasts = len(required_contrasts)
        # Get pipeline model names
        name_models = models.TASKS[arguments.task[0]]['models']
    else:
        n_contrasts = len(arguments.i)
        name_models = arguments.task

    if len(arguments.i) != n_contrasts:
        parser.error(
            "{} input files found. Please provide all required input files for the task {}, i.e. contrasts: {}."
            .format(len(arguments.i), arguments.task, ', '.join(required_contrasts)))

    # Check modality order
    if len(arguments.i) > 1 and arguments.c is None:
        parser.error(
            "Please specify the order in which you put the contrasts in the input images (-i) with flag -c, e.g., "
            "-c t1 t2")

    # Run pipeline by iterating through the models
    fname_prior = None
    output_filenames = None
    for name_model in name_models:
        # Check if this is an official model
        if name_model in list(models.MODELS.keys()):
            # If it is, check if it is installed
            path_model = models.folder(name_model)
            path_models = models.find_model_folder_paths(path_model)
            if not models.is_valid(path_models):
                printv("Model {} is not installed. Installing it now...".format(name_model))
                models.install_model(name_model)
        # If it is not, check if this is a path to a valid model
        else:
            path_model = os.path.abspath(name_model)
            path_models = models.find_model_folder_paths(path_model)
            if not models.is_valid(path_models):
                parser.error("The input model is invalid: {}".format(path_models))

        # Order input images
        if arguments.c is not None:
            input_filenames = []
            for required_contrast in models.MODELS[name_model]['contrasts']:
                for provided_contrast, input_filename in zip(arguments.c, arguments.i):
                    if required_contrast == provided_contrast:
                        input_filenames.append(input_filename)
        else:
            input_filenames = arguments.i

        # Call segment_nifti
        options = {**vars(arguments), "fname_prior": fname_prior}
        # NB: For single models, the averaging will have no effect.
        #     For model ensembles, this will average the output of the ensemble into a single set of outputs.
        nii_lst, target_lst = segment_and_average_volumes(path_models, input_filenames, options=options)

        # Delete intermediate outputs
        if fname_prior and os.path.isfile(fname_prior) and arguments.r:
            logger.info("Remove temporary files...")
            os.remove(fname_prior)

        output_filenames = []
        # Save output seg
        for nii_seg, target in zip(nii_lst, target_lst):
            if 'o' in options and options['o'] is not None:
                # To support if the user adds the extension or not
                extension = ".nii.gz" if ".nii.gz" in options['o'] else ".nii" if ".nii" in options['o'] else ""
                if extension == "":
                    fname_seg = options['o'] + target if len(target_lst) > 1 else options['o']
                else:
                    fname_seg = options['o'].replace(extension, target + extension) if len(target_lst) > 1 \
                        else options['o']
            else:
                fname_seg = ''.join([splitext(input_filenames[0])[0], target + '.nii.gz'])

            # If output folder does not exist, create it
            path_out = os.path.dirname(fname_seg)
            if not (path_out == '' or os.path.exists(path_out)):
                os.makedirs(path_out)

            nib.save(nii_seg, fname_seg)
            output_filenames.append(fname_seg)

        # Use the result of the current model as additional input of the next model
        fname_prior = fname_seg

    for output_filename in output_filenames:
        display_viewer_syntax([arguments.i[0], output_filename], colormaps=['gray', 'red'], opacities=['', '0.7'], verbose=verbose)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
