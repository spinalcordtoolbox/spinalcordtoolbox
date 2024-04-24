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

import torch

from spinalcordtoolbox.deepseg import models, inference
from spinalcordtoolbox.image import splitext, Image, check_image_kind
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, display_viewer_syntax
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel
from spinalcordtoolbox.utils.fs import tmp_create

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
        help="Contrast of the input. The `-c` option is only relevant for the following tasks:"
             "\n   - 'seg_tumor-edema-cavity_t1-t2': Specifies the contrast order of input images (e.g. -c t1 t2)"
             "\n   - 'seg_sc_ms_lesion_stir_psir': Specifies whether input should be inverted based on contrast "
             "(-c stir: no inversion, -c psir: inverted)"
             "\nBecause all other models have only a single input contrast, the '-c' option is ignored for them.",
        choices=('t1', 't2', 't2star', 'stir', 'psir'),
        metavar=Metavar.str)
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

    # Check if all input images have been specified (only relevant for 'seg_tumor-edema-cavity_t1-t2')
    # TODO: Fix contrast-related behavior as per https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/4445
    if 'seg_tumor-edema-cavity_t1-t2' in arguments.task[0] and len(arguments.i) != n_contrasts:
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

        # Order input images (only relevant for 'seg_tumor-edema-cavity_t1-t2')
        # TODO: Fix contrast-related behavior as per https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/4445
        if 'seg_tumor-edema-cavity_t1-t2' in arguments.task[0] and arguments.c is not None:
            input_filenames = []
            for required_contrast in models.MODELS[name_model]['contrasts']:
                for provided_contrast, input_filename in zip(arguments.c, arguments.i):
                    if required_contrast == provided_contrast:
                        input_filenames.append(input_filename)
        else:
            input_filenames = arguments.i.copy()

        # Inversion workaround for regular PSIR input to canproco STIR/PSIR model
        if 'seg_sc_ms_lesion_stir_psir' in arguments.task[0]:
            contrast = arguments.c[0] if arguments.c else None  # default is empty list
            if not contrast:
                parser.error(
                    "Task 'seg_sc_ms_lesion_stir_psir' requires the flag `-c` to identify whether the input is "
                    "STIR or PSIR. If `-c psir` is passed, the input will be inverted.")
            elif contrast == "psir":
                logger.warning("Inverting input PSIR image (multiplying data array by -1)...")
                tmpdir = tmp_create("sct_deepseg-inverted-psir")
                for i, fname_in in enumerate(input_filenames.copy()):
                    im_in = Image(fname_in)
                    im_in.data *= -1
                    path_img_tmp = os.path.join(tmpdir, os.path.basename(fname_in))
                    im_in.save(path_img_tmp)
                    input_filenames[i] = path_img_tmp
            else:
                if contrast != "stir":
                    parser.error("Task 'seg_sc_ms_lesion_stir_psir' requires the flag `-c` to be either psir or stir.")

        if 'seg_sc_epi' in arguments.task[0]:
            for image in arguments.i:
                image_shape = Image(image).data.shape
                if len(image_shape) == 4:
                    parser.error("Only 3D volumes are supported for this task. You can either provide a mean volume "
                                 "(using 'sct_maths -mean') or a single time point (using 'sct_image -split t'.")

        # Segment the image based on the type of model present in the model folder
        try:
            model_type = models.check_model_software_type(path_models[0])  # NB: [0] -> Fetch first model from ensemble
        except ValueError:
            printv(f"Model type could not be determined. Directory '{path_model}' may be missing necessary files."
                   f"Please redownload the model using `sct_deepseg -install-task` before continuing.", type="error")

        # Control GPU usage based on SCT-specific environment variable
        # NB: We use 'SCT_USE_GPU' as a "hidden option" to turn on GPU inference internally.
        # NB: Controlling which GPU(s) are used should be done by the environment variable 'CUDA_VISIBLE_DEVICES'.
        use_gpu = torch.cuda.is_available() and "SCT_USE_GPU" in os.environ

        if model_type == 'ivadomed':
            # NB: For single models, the averaging will have no effect.
            #     For model ensembles, this will average the output of the ensemble into a single set of outputs.
            im_lst, target_lst = inference.segment_and_average_volumes(path_models, input_filenames, use_gpu=use_gpu,
                                                                       options={**vars(arguments),
                                                                                "fname_prior": fname_prior})
        else:
            thr = (arguments.binarize_prediction if arguments.binarize_prediction
                   else models.MODELS[name_model]['thr'])  # Default `thr` value stored in model dict
            im_lst, target_lst = inference.segment_non_ivadomed(path_model, model_type, input_filenames, thr,
                                                                use_gpu=use_gpu, remove_temp_files=arguments.r)

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
                path_out = os.path.dirname(fname_seg)
            else:
                # NB: we use `arguments.i` here to preserve the original input directory, even if `input_filenames`
                #     is preprocessed in a tmpdir
                path_out = os.path.dirname(os.path.abspath(arguments.i[0]))
                basename = splitext(os.path.basename(arguments.i[0]))[0]
                fname_seg = os.path.join(path_out, f"{basename}{target}.nii.gz")

            # If output folder does not exist, create it
            if not (path_out == '' or os.path.exists(path_out)):
                os.makedirs(path_out)

            im_seg.save(fname_seg)
            output_filenames.append(fname_seg)

        # Use the result of the current model as additional input of the next model
        fname_prior = fname_seg

    for output_filename in output_filenames:
        img_kind = check_image_kind(Image(output_filename))
        display_viewer_syntax([arguments.i[0], output_filename],
                              im_types=['anat', img_kind],
                              opacities=['', '0.7'], verbose=verbose)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])