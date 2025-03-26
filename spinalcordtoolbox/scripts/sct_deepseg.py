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

import json
import os
import sys
import logging
from typing import Sequence
import textwrap
import functools

from spinalcordtoolbox.reports import qc2
from spinalcordtoolbox.image import splitext, Image, check_image_kind
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, display_viewer_syntax, ActionCreateFolder
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel, __version__, _git_info
from spinalcordtoolbox.utils.sys import LazyLoader

cuda = LazyLoader("cuda", globals(), 'torch.cuda')

inference = LazyLoader("inference", globals(), 'spinalcordtoolbox.deepseg.inference')
models = LazyLoader("models", globals(), 'spinalcordtoolbox.deepseg.models')

logger = logging.getLogger(__name__)


def get_parser(subparser_to_return=None):
    # Initialize the top-level `sct_deepseg` argparser
    parser = SCTArgumentParser(
        description="Segment an anatomical structure or pathologies according to the specified deep learning model.",
        epilog=models.list_tasks_string()
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        "-list-tasks",
        action='store_true',
        help="Display a list of tasks, along with detailed descriptions (including information on how the model was "
             "trained, what data it was trained on, any performance evaluations, associated papers, etc.)")

    # Add some universal arguments and their associated functionality
    parser.add_common_args()

    # Initialize the `subparsers` "special action object" that can be used to create subparsers
    # See https://docs.python.org/3/library/argparse.html#sub-commands for more details.
    parser_dict = {}
    subparsers = parser.add_subparsers(help=textwrap.dedent("""
        Segmentation task to perform.
            - To install a task, run `sct_deepseg TASK_NAME -install`
            - To segment an image, run `sct_deepseg TASK_NAME -i input.nii.gz`
            - To view additional options for a given task, run `sct_deepseg TASK_NAME -h`
    """), metavar="TASK_NAME", dest="task")

    # Generate 1 subparser per task, and add the following arguments to all subparsers
    # Even if all subparsers share these arguments, it's better to duplicate them, since it allows for the usage:
    #    `sct_deepseg TASK_NAME -i input.nii.gz`
    # In other words, the arguments can come after the task name, which matches current usage. Otherwise, we would have
    # to use the following usage instead, which feels weird when we're using subcommands:
    #    `sct_deepseg -i input.nii.gz TASK_NAME`
    for task_name, task_dict in models.TASKS.items():
        optional_ref = (f"{task_dict['citation']}\n\n" if task_dict['citation'] else "")
        subparser = parser_dict[task_name] = subparsers.add_parser(task_name, description=(f"""
{task_dict["description"]}

{task_dict["long_description"]}

## Reference

{optional_ref}Project URL: [{task_dict["url"]}]({task_dict["url"]})

## Usage

"""))

        input_output = subparser.add_argument_group("\nINPUT/OUTPUT")
        input_output.add_argument(
            "-i",
            nargs="+",
            help=f"Image to segment. Can be multiple images (separated with space)."
                 f"\n\nNote: If choosing `lesion_ms_mp2rage`, then the input "
                 f"data must be cropped around the spinal cord. ({models.CROP_MESSAGE})",
            metavar=Metavar.file)
        input_output.add_argument(
            "-o",
            help="Output file name. In case of multi-class segmentation, class-specific suffixes will be added. By default,"
                 "the suffix specified in the packaged model will be added and output extension will be `.nii.gz`.",
            metavar=Metavar.str)

        seg = subparser.add_argument_group('\nTASKS')
        seg.add_argument(
            "-install",
            help="Install models that are required for specified task.",
            action="store_true")
        seg.add_argument(
            "-custom-url",
            nargs="+",  # NB: `nargs="+"` won't work for installing custom ensemble models, but we no longer have any
            help="URL(s) pointing to the `.zip` asset for a model release. This option can be used with `-install` to "
                 "install a specific version of a model. To use this option, navigate to the 'Releases' page of the model, "
                 "find release you wish to install, and right-click + copy the URL of the `.zip` listed under 'Assets'.\n"
                 "NB: For multi-model tasks, provide multiple URLs. For single models, just provide one URL.\n"
                 "Example:\n"
                 "`sct_deepseg -install rootlets_t2 -custom-url "
                 "https://github.com/ivadomed/model-spinal-rootlets/releases/download/r20240523/model-spinal-rootlets_ventral_D106_r20240523.zip`\n"
                 "`sct_deepseg rootlets_t2 -i sub-amu01_T2w.nii.gz`")

        misc = subparser.add_argument_group('\nPARAMETERS')
        misc.add_argument(
            "-thr",
            type=float,
            dest='binarize_prediction',
            help="Binarize segmentation with specified threshold. Set to 0 for no thresholding (i.e., soft segmentation). "
                 "Default value is model-specific and was set during optimization "
                 "(more info at https://github.com/sct-pipeline/deepseg-threshold).",
            metavar=Metavar.float)
        misc.add_argument(
            "-largest",
            dest='keep_largest',
            type=int,
            help="Keep the largest connected-objects from the output segmentation. Specify the number of objects to keep."
                 "To keep all objects, set to 0")
        misc.add_argument(
            "-fill-holes",
            type=int,
            help="Fill small holes in the segmentation.",
            choices=(0, 1))
        misc.add_argument(
            "-remove-small",
            type=str,
            nargs="+",
            help="Minimal object size to keep with unit (mm3 or vox). A single value can be provided or one value per "
                 "prediction class. Single value example: 1mm3, 5vox. Multiple values example: 10 20 10vox (remove objects "
                 "smaller than 10 voxels for class 1 and 3, and smaller than 20 voxels for class 2).")

        misc = subparser.misc_arggroup
        misc.add_argument(
            '-qc',
            metavar=Metavar.folder,
            action=ActionCreateFolder,
            help="The path where the quality control generated content will be saved."
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
            "-qc-plane",
            metavar=Metavar.str,
            choices=('Axial', 'Sagittal'),
            default='Axial',
            help="Plane of the output QC. If Sagittal, it is highly recommended to provide the `-qc-seg` option, "
                 "as it will ensure the output QC is cropped to a reasonable field of view. "
                 "(Note: Sagittal view is not currently supported for rootlets/totalspineseg QC.)")
        misc.add_argument(
            "-qc-seg",
            metavar=Metavar.file,
            help=textwrap.dedent("""
                    Segmentation file to use for cropping the QC. This option is useful when you want to QC a region that is different from the output segmentation. For example, for lesion segmentation, it might be useful to provide a cord segmentation to expand the QC field of view to include the full cord, while also still excluding irrelevant tissue.
                    If not provided, the default behavior will depend on the `-qc-plane`:
                       - 'Axial': A sensibly chosen crop radius between 15-40 vox, depending on the resolution and segmentation type.
                       - 'Sagittal': The full image. (For very large images, this may cause a crash, so using `-qc-seg` is highly recommended.)
                """)  # noqa: E501 (line too long)
        )

        # Add common arguments
        subparser.add_common_args()
        subparser.add_tempfile_args()

    # Add options that only apply to a specific task
    parser_dict['tumor_edema_cavity_t1_t2'].add_argument(
        "-c",
        nargs="+",
        help="Contrast of the input. Specifies the contrast order of input images (e.g. `-c t1 t2`)",
        choices=('t1', 't2', 't2star'),
        metavar=Metavar.str)

    if subparser_to_return:
        return parser_dict[subparser_to_return]
    else:
        return parser


# Define subparsers to be used in the "gallery of tasks" documentation for each task.
# `sphinx-argparse` requires a function of no arguments that returns the parser, so here they are
for task in models.TASKS.keys():
    globals()[task] = functools.partial(get_parser, task)


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    if not (
        arguments.list_tasks
        or (arguments.task and arguments.install)
        or (arguments.task and arguments.i)
    ):
        parser.error("You must specify either a task name + '-install', a task name + an image ('-i'), or "
                     "'-list-tasks'.")

    # Deal with task long description
    if arguments.list_tasks:
        models.display_list_tasks()
        exit(0)

    if arguments.install:
        models_to_install = models.TASKS[arguments.task]['models']
        if arguments.custom_url:
            if len(arguments.custom_url) != len(models_to_install):
                parser.error(f"Expected {len(models_to_install)} URL(s) for task {arguments.install}, "
                             f"but got {len(arguments.custom_url)} URL(s) instead.")
            for name_model, custom_url in zip(models_to_install, arguments.custom_url):
                models.install_model(name_model, custom_url)
        else:
            for name_model in models_to_install:
                models.install_model(name_model)
        exit(0)

    # Deal with input/output
    for file in arguments.i:
        if not os.path.isfile(file):
            parser.error("This file does not exist: {}".format(file))

    # Get pipeline model names
    name_models = models.TASKS[arguments.task]['models']

    # Check if all input images and contrasts have been specified (only relevant for 'tumor-edema-cavity_t1-t2')
    if arguments.task == 'tumor_edema_cavity_t1_t2':
        required_contrasts = models.get_required_contrasts(arguments.task)
        if len(arguments.i) != len(required_contrasts):
            parser.error(
                "{} input files found. Please provide all required input files for the task {}, i.e. contrasts: {}."
                .format(len(arguments.i), arguments.task, ', '.join(required_contrasts)))
        if len(arguments.c) != len(arguments.i):
            parser.error(f"{len(arguments.i)} input files provided, but {len(arguments.c)} contrasts passed. "
                         f"Number of contrasts should match the number of inputs.")

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
            # Check folder version file ('{path_model}/source.json')
            elif not models.is_up_to_date(path_model):
                printv("Model {} is out of date. Re-installing it now...".format(name_model))
                models.install_model(name_model)
                path_models = models.find_model_folder_paths(path_model)  # Re-parse to find newly downloaded folders
        # If it is not, check if this is a path to a valid model
        else:
            path_model = os.path.abspath(name_model)
            path_models = models.find_model_folder_paths(path_model)
            if not models.is_valid(path_models):
                parser.error("The input model is invalid: {}".format(path_models))

        # Order input images (only relevant for 'tumor-edema-cavity_t1-t2')
        if arguments.task == 'tumor_edema_cavity_t1_t2':
            input_filenames = []
            for required_contrast in models.MODELS[name_model]['contrasts']:
                for provided_contrast, input_filename in zip(arguments.c, arguments.i):
                    if required_contrast == provided_contrast:
                        input_filenames.append(input_filename)
        else:
            input_filenames = arguments.i.copy()

        if arguments.task == 'sc_epi':
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
                   f"Please redownload the model using `sct_deepseg {arguments.task} -install` before continuing.", type="error")

        # Control GPU usage based on SCT-specific environment variable
        # NB: We use 'SCT_USE_GPU' as a "hidden option" to turn on GPU inference internally.
        # NB: Controlling which GPU(s) are used should be done by the environment variable 'CUDA_VISIBLE_DEVICES'.
        use_gpu = cuda.is_available() and "SCT_USE_GPU" in os.environ

        if model_type == 'ivadomed':
            # NB: For single models, the averaging will have no effect.
            #     For model ensembles, this will average the output of the ensemble into a single set of outputs.
            im_lst, target_lst = inference.segment_and_average_volumes(path_models, input_filenames, use_gpu=use_gpu,
                                                                       options={**vars(arguments),
                                                                                "fname_prior": fname_prior})
        else:
            thr = (arguments.binarize_prediction if arguments.binarize_prediction is not None
                   else models.MODELS[name_model]['thr'])  # Default `thr` value stored in model dict
            im_lst, target_lst = inference.segment_non_ivadomed(
                path_model, model_type, input_filenames, thr,
                # NOTE: contrast-agnostic nnunet model sometimes predicts pixels outside the cord, we want to
                # set keep_largest object as the default behaviour when using this model
                keep_largest=1 if arguments.task == 'spinalcord' else arguments.keep_largest,
                fill_holes_in_pred=arguments.fill_holes,
                remove_small=arguments.remove_small,
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

        # write JSON sidecar file
        source_path = os.path.join(path_model, "source.json")
        if os.path.isfile(source_path):
            with open(source_path, "r") as fp:
                source_dict = json.load(fp)
        sidecar_json = {
            'GeneratedBy': [
                {
                    "Name": f"spinalcordtoolbox: sct_deepseg {' '.join(os.path.basename(arg) for arg in argv)}",
                    "Version": __version__,
                    "CodeURL": f"https://github.com/spinalcordtoolbox/spinalcordtoolbox/"
                               f"blob/{_git_info()[1].strip('*')}/spinalcordtoolbox/scripts/sct_deepseg.py",
                    "ModelURL": source_dict["model_urls"],
                }
            ]
        }
        with open(splitext(fname_seg)[0] + ".json", "w") as fp:
            json.dump(sidecar_json, fp, indent=4)

    if arguments.qc is not None:
        # If `arguments.qc_seg is None`, each entry will be treated as an
        # empty file with the same size as the corresponding input image
        qc_seg = [arguments.qc_seg] * len(input_filenames)
        # Models can have multiple input images -- create 1 QC report per input image.
        if len(output_filenames) == len(input_filenames):
            iterator = zip(input_filenames, output_filenames, [None] * len(input_filenames), qc_seg)
        # Special case: totalspineseg which outputs 5 files per 1 input file
        # Just use the 5th image ([4]) which represents the step2 output
        elif arguments.task == 'totalspineseg':
            assert len(output_filenames) == 5 * len(input_filenames)
            iterator = zip(input_filenames, output_filenames[4::5], [None] * len(input_filenames), qc_seg)
        # Other models typically have 2 outputs per input (e.g. SC + lesion), so use both segs
        else:
            assert len(output_filenames) == 2 * len(input_filenames)
            iterator = zip(input_filenames, output_filenames[0::2], output_filenames[1::2], qc_seg)

        # Create one QC report per input image, with one or two segs per image
        species = 'mouse' if 'mouse' in arguments.task else 'human'  # used for resampling
        for fname_in, fname_seg1, fname_seg2, fname_qc_seg in iterator:
            qc2.sct_deepseg(
                fname_input=fname_in,
                fname_seg=fname_seg1,
                fname_seg2=fname_seg2,
                species=species,
                argv=argv,
                path_qc=os.path.abspath(arguments.qc),
                dataset=arguments.qc_dataset,
                subject=arguments.qc_subject,
                plane=arguments.qc_plane,
                fname_qc_seg=fname_qc_seg
            )

    images = [arguments.i[0]]
    im_types = ['anat']
    opacities = ['']
    for output_filename in output_filenames:
        images.append(output_filename)
        im_types.append(check_image_kind(Image(output_filename)))
        opacities.append('0.7')
    display_viewer_syntax(images, im_types=im_types, opacities=opacities, verbose=verbose)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
