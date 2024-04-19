#!/usr/bin/env python
#
# This command-line tool is the interface for the deepseg API that performs
# segmentation using deep learning from the ivadomed package.
#
# Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

# TODO: Add link to example image so users can decide wether their images look "close enough" to some of the proposed
#  models (e.g., mice, etc.).
# TODO: add test to make sure that all postprocessing flags match the core.DEFAULT dictionary items
# TODO: Fetch default value (and display) depending on the model that is used.
# TODO: accommodate multiclass segmentation

import os
import sys
import logging
from typing import Sequence

from spinalcordtoolbox.reports import qc2
from spinalcordtoolbox.deepseg import models, inference
from spinalcordtoolbox.image import splitext, Image, check_image_kind
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, display_viewer_syntax, ActionCreateFolder
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel

logger = logging.getLogger(__name__)


def get_parser():
    parser = SCTArgumentParser(
        description="Segment an anatomical structure or pathologies according to the specified deep learning model.",
        epilog=models.list_tasks_string()
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
        '-qc',
        metavar=Metavar.folder,
        action=ActionCreateFolder,
        help="The path where the quality control generated content will be saved. Note: This flag requires the '-dseg' "
             "flag."
    )
    misc.add_argument(
        '-qc-dataset',
        metavar=Metavar.str,
        help="If provided, this string will be mentioned in the QC report as the dataset the process was run on."
    )
    misc.add_argument(
        '-qc-subject',
        metavar=Metavar.str,
        help="If provided, this string will be mentioned in the QC report as the subject the process was run on."
    )
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


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)

    if (arguments.list_tasks is False
            and arguments.install_task is None
            and (arguments.i is None or arguments.task is None)):
        parser.error("You must specify either '-list-tasks', '-install-task', "
                     "or both '-i' + '-task'.")

    # Deal with task long description
    if arguments.list_tasks:
        models.display_list_tasks()

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
                path_models = models.find_model_folder_paths(path_model)  # Re-parse to find newly downloaded folders
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

        # Segment the image based on the type of model present in the model folder
        try:
            model_type = models.check_model_software_type(path_models[0])  # NB: [0] -> Fetch first model from ensemble
        except ValueError:
            printv(f"Model type could not be determined. Directory '{path_model}' may be missing necessary files."
                   f"Please redownload the model using `sct_deepseg -install-task` before continuing.", type="error")

        if model_type == 'ivadomed':
            # NB: For single models, the averaging will have no effect.
            #     For model ensembles, this will average the output of the ensemble into a single set of outputs.
            im_lst, target_lst = inference.segment_and_average_volumes(path_models, input_filenames,
                                                                       options={**vars(arguments),
                                                                                "fname_prior": fname_prior})
        else:
            thr = (arguments.binarize_prediction if arguments.binarize_prediction
                   else models.MODELS[name_model]['thr'])  # Default `thr` value stored in model dict
            im_lst, target_lst = inference.segment_non_ivadomed(path_model, model_type, input_filenames, thr,
                                                                remove_temp_files=arguments.r)

        # Delete intermediate outputs
        if fname_prior and os.path.isfile(fname_prior) and arguments.r:
            logger.info("Remove temporary files...")
            os.remove(fname_prior)

        output_filenames = []
        # Save output seg
        for im_seg, target in zip(im_lst, target_lst):
            if hasattr(arguments, 'o') and arguments.o is not None:
                # To support if the user adds the extension or not
                extension = ".nii.gz" if ".nii.gz" in arguments.o else ".nii" if ".nii" in arguments.o else ""
                if extension == "":
                    fname_seg = arguments.o + target if len(target_lst) > 1 else arguments.o
                else:
                    fname_seg = arguments.o.replace(extension, target + extension) if len(target_lst) > 1 \
                        else arguments.o
            else:
                fname_seg = ''.join([splitext(input_filenames[0])[0], target + '.nii.gz'])

            # If output folder does not exist, create it
            path_out = os.path.dirname(fname_seg)
            if not (path_out == '' or os.path.exists(path_out)):
                os.makedirs(path_out)

            im_seg.save(fname_seg)
            output_filenames.append(fname_seg)

        # Use the result of the current model as additional input of the next model
        fname_prior = fname_seg

    if arguments.qc is not None:
        if len(input_filenames) > 1:
            # The only model that meets this case is :
            # It has 2 inputs -> 2 outputs (1 seg per input, no secondary seg)
            iterator = zip(input_filenames, output_filenames, [None] * len(input_filenames))  # [in, out1, out2=None]
        else:
            # The remaining models will have 1 input -> 1 OR 2 outputs.
            assert len(input_filenames) == 1
            iterator = zip([input_filenames[0]], [output_filenames[0]],                    # [in, out1
                           [None if len(output_filenames) == 1 else output_filenames[1]])  # [out2]

        # Create one QC report per input image, with one or two segs per image
        for fname_in, fname_seg1, fname_seg2 in iterator:
            qc2.sct_deepseg(
                fname_input=fname_in,
                fname_seg=fname_seg1,
                fname_seg2=fname_seg2,
                argv=argv,
                path_qc=os.path.abspath(arguments.qc),
                dataset=arguments.qc_dataset,
                subject=arguments.qc_subject,
            )

    for output_filename in output_filenames:
        img_kind = check_image_kind(Image(output_filename))
        display_viewer_syntax([arguments.i[0], output_filename],
                              im_types=['anat', img_kind],
                              opacities=['', '0.7'], verbose=verbose)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
