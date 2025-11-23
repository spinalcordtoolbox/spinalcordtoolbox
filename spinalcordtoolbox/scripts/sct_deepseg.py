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

from argparse import SUPPRESS, Action
import json
import os
import sys
import logging
from typing import Sequence
from textwrap import dedent
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


class _TaskDeprecationAction(Action):
    """
    ArgParse action which, similar to help, terminates the program early if the user tries to use the old `-task`
    syntax and prompts them to update their command with the new `deepseg task_name` syntax instead
    """
    def __init__(self,
                 option_strings,
                 dest=SUPPRESS,
                 default=SUPPRESS,
                 help=SUPPRESS):
        # Slight modification of `_HelpAction` __init__, as this functions more-or-less identically
        super(_TaskDeprecationAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs='?',
            metavar='TASK',
            help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        # If no task name was provided for some reason, use the metavar instead
        if values is None:
            values = self.metavar

        # Print out an informative message for the user to help them transition to the new task
        printv(dedent(f"""
        The '-task' flag has been deprecated as of SCT version 7.0.
        To resolve this, change your command from this:

        > sct_deepseg -task {values} ...

        To this:

        > sct_deepseg {values} ...
        """[1:]))  # noqa: E501 (line too long)
        #   ^ This removes the implicit newline that `dedent` likes to add

        # Stop parsing immediately
        parser.exit()


class _ListTaskDetailsAction(Action):
    """
    ArgParse action which, similar to help, terminates the program early if the user wants the tasks listed out for
    them. Making this an action prevents some nasty side effects which might be caused by parsing other arguments.
    """
    def __init__(self,
                 option_strings,
                 dest=SUPPRESS,
                 default=SUPPRESS,
                 help=SUPPRESS):
        # Slight modification of `_HelpAction` __init__, as this functions more-or-less identically
        super(_ListTaskDetailsAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        # Display the detailed task list
        models.display_list_tasks()

        # Stop parsing immediately
        parser.exit()


def get_parser(subparser_to_return=None):
    # Initialize the top-level `sct_deepseg` argparser
    parser = SCTArgumentParser(
        description="Segment an anatomical structure or pathologies according to the specified deep learning model.",
        usage=dedent("""
        sct_deepseg TASK ...

        Examples:
            sct_deepseg spinalcord -h
            sct_deepseg gm_mouse_t1 -install
            sct_deepseg lesion_ms -i cMRI3712.nii.gz

        View available tasks:
            sct_deepseg -h
            sct_deepseg -task-details
        """[1:]),
        epilog=models.list_tasks_string()
    )

    # Hidden `-task` argument to help users transition to the command format
    parser.add_argument(
        "-task",
        action=_TaskDeprecationAction
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        "-task-details",
        action=_ListTaskDetailsAction,
        help="Display a list of tasks, along with detailed descriptions (including information on how the model was "
             "trained, what data it was trained on, and any performance evaluations, associated papers, etc. it may have)")
    optional.add_argument(
        "-h", "--help",
        action="help",
        help="Show this help message and exit."
    )

    # Initialize the `subparsers` "special action object" that can be used to create subparsers
    # See https://docs.python.org/3/library/argparse.html#sub-commands for more details.
    parser_dict = {}
    subparsers = parser.add_subparsers(help=dedent("""
        Segmentation task to perform.
            - To install a task, run `sct_deepseg TASK -install`
            - To segment an image, run `sct_deepseg TASK -i input.nii.gz`
            - To view additional options for a given task, run `sct_deepseg TASK -h`
    """), metavar="TASK", dest="task")

    # Generate 1 subparser per task, and add the following arguments to all subparsers
    # Even if all subparsers share these arguments, it's better to duplicate them, since it allows for the usage:
    #    `sct_deepseg TASK_NAME -i input.nii.gz`
    # In other words, the arguments can come after the task name, which matches current usage. Otherwise, we would have
    # to use the following usage instead, which feels weird when we're using subcommands:
    #    `sct_deepseg -i input.nii.gz TASK_NAME`
    for task_name, task_dict in models.TASKS.items():
        # Store certain argument objects for later use (as we may want to suppress them for specific tasks)
        task_args = {}

        # Build up the description text in parts so dedent doesn't have a stroke
        description_text = dedent(f"""
            {task_dict["description"]}

            {task_dict["long_description"]}

            ## Reference\n
        """)
        if task_dict.get('citation', False):
            description_text += f"{task_dict['citation']}\n\n"
        description_text += f"Project URL: [{task_dict['url']}]({task_dict['url']})"

        subparser = parser_dict[task_name] = subparsers.add_parser(
            task_name,
            description=description_text
        )

        input_output = subparser.add_argument_group("\nINPUT/OUTPUT")
        task_args['-i'] = input_output.add_argument(
            "-i",
            nargs="+",
            help="Image filename(s) to segment. If segmenting multiple files, separate filenames with a space.",
            metavar=Metavar.file)
        input_output.add_argument(
            "-o",
            help="Output file name. The chosen filename will be used as a base name, and model-specific suffixes will "
                 "be added to the end depending on the type of output (e.g. '_cord.nii.gz', '_gm.nii.gz', etc.).",
            metavar=Metavar.str)

        seg = subparser.add_argument_group('\nTASKS')
        seg.add_argument(
            "-install",
            help="Install models that are required for specified task.",
            action="store_true")
        seg.add_argument(
            "-custom-url",
            nargs="+",  # NB: `nargs="+"` won't work for installing custom ensemble models, but we no longer have any
            # NB: For multi-model tasks, provide multiple URLs. For single models, just provide one URL.
            #     We don't mention it in the help because we no longer have any multi-model tasks.
            #     But, if we were to re-add a multi-model task one day, we could selectively amend this message.
            help=f"URL(s) pointing to the `.zip` asset for a model release. This option can be used with `-install` to "
                 f"install a specific version of a model. To use this option, navigate to the 'Releases' page of the model, "
                 f"find release you wish to install, and right-click + copy the URL of the `.zip` listed under 'Assets'.\n"
                 f"Example:\n"
                 f"`sct_deepseg {task_name} -install -custom-url CUSTOM_URL`\n"
                 f"`sct_deepseg {task_name} -i t2.nii.gz`")

        params = subparser.add_argument_group('\nPARAMETERS')
        thr_values = [models.MODELS[model_name]['thr'] for model_name in task_dict['models']]
        task_args['-thr'] = params.add_argument(
            "-thr",
            type=float,
            dest='binarize_prediction',
            help=(f"Binarize segmentation with specified threshold. Set to 0 for no thresholding (i.e., soft segmentation). "
                  f"Default value is '{thr_values}', and was chosen by experimentation "
                  f"(more info at https://github.com/sct-pipeline/deepseg-threshold)."),
            metavar=Metavar.float)
        params.add_argument(
            "-largest",
            dest='keep_largest',
            type=int,
            default=0,
            choices=(0, 1),
            help="Keep the largest connected object from each output segmentation; if not set, all objects are kept.")
        params.add_argument(
            "-fill-holes",
            type=int,
            choices=(0, 1),
            default=0,
            help="If set, small holes in the segmentation will be filled in automatically."
        )
        params.add_argument(
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
        task_args['-qc-plane'] = misc.add_argument(
            "-qc-plane",
            metavar=Metavar.str,
            choices=('Axial', 'Sagittal'),
            default='Axial',
            help="Plane of the output QC. If Sagittal, it is highly recommended to provide the `-qc-seg` option, "
                 "as it will ensure the output QC is cropped to a reasonable field of view.")
        note_qc_seg = dedent("""

            If `-qc-seg` is not provided, the default behavior will depend on the value of `-qc-plane`:
              - 'Axial': Without '-qc-seg', a sensible crop radius between 15-40 vox will be automatically used, depending on the resolution and segmentation type.
              - 'Sagittal': Without '-qc-seg', the full image will be displayed by default. (For very large images, this may cause a crash, so using `-qc-seg` is highly recommended.)
        """)   # noqa: E501 (line too long)
        task_args['-qc-seg'] = misc.add_argument(
            "-qc-seg",
            metavar=Metavar.file,
            help="Segmentation file to use for cropping the QC. This option is useful when you want to QC a region "
                 "that is different from the output segmentation. For example, it might be useful to provide a "
                 "dilated cord segmentation to expand the QC field of view." + note_qc_seg
        )

        # Add common arguments
        subparser.add_common_args()
        subparser.add_tempfile_args()

        # Add options that only apply to specific tasks
        is_nnunet = all(models.MODELS[model_name]['framework'] == "nnunetv2" for model_name in task_dict['models'])
        if is_nnunet and task_name != 'totalspineseg':
            # Test time augmentation is an nnUNet-specific feature (`use_mirroring=True` internally)
            # But, the totalspineseg package doesn't support this argument (yet), so skip it
            params.add_argument(
                "-test-time-aug",
                action='store_true',
                help="Perform test-time augmentation (TTA) by flipping the input image along all axes and averaging the "
                     "resulting predictions.\n"
                     "Note: The time it takes to run the model will increase due to the additional predictions."
            )
        if task_name == 'tumor_edema_cavity_t1_t2':
            input_output.add_argument(
                "-c",
                nargs="+",
                help="Contrast of the input. Specifies the contrast order of input images (e.g. `-c t1 t2`)",
                choices=('t1', 't2', 't2star'),
                metavar=Metavar.str)
        if task_name == 'totalspineseg':
            params.add_argument(
                "-step1-only",
                type=int,
                help="If set to '1', only Step 1 will be performed. If not provided, both steps will be run.\n"
                     "- Step 1: Segments the spinal cord, spinal canal, vertebrae, and intervertebral discs (IVDs). Labels the IVDs, but vertebrae are left unlabeled.\n"
                     "- Step 2: Fine-tunes the segmentation, applies labels to vertebrae, and segments the sacrum if present.\n"
                     "More details on TotalSpineSeg's two models can be found here: https://github.com/neuropoly/totalspineseg/?tab=readme-ov-file#model-description",
                choices=(0, 1),
                default=0)
        if task_name == 'lesion_ms':
            # Add possibility of having soft segmentation for the lesion_ms task
            params.add_argument(
                "-soft-ms-lesion",
                action="store_true",
                help="If set, the model will output a soft segmentation (i.e. probability map) instead of a binary "
                     "segmentation."
            )
            # Add possibility of segmenting on only 1 fold for quicker inference
            params.add_argument(
                "-single-fold",
                action="store_true",
                help="If set, only 1 fold will be used for inference instead of the full 5-fold ensemble. This will speed up inference, but may reduce segmentation quality."
            )

        # Add input cropping note specific to the `lesion_ms_mp2rage` task
        if task_name == 'lesion_ms_mp2rage':
            task_args['-i'].help += dedent(f"""

            Note: For `lesion_ms_mp2rage`, the input
            data must be cropped around the spinal cord.
            ({models.CROP_MESSAGE})
        """)

        # Suppress arguments that are irrelevant for certain tasks
        # - Sagittal view is not currently supported for rootlets/totalspineseg QC
        #   This means that the `-qc-plane` argument (and the `-qc-seg` note) should be hidden for these tasks
        tasks_without_sagittal_qc = ('rootlets', 'totalspineseg')
        if task_name in tasks_without_sagittal_qc:
            task_args['-qc-plane'].help = SUPPRESS
            task_args['-qc-seg'].help = task_args['-qc-seg'].help.replace(note_qc_seg, "")

        # - If none of the models for this task have a default threshold, then thresholding is not applicable.
        if all(models.MODELS[model_name]['thr'] is None for model_name in task_dict['models']):
            task_args['-thr'].help = SUPPRESS

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

    # If the user called the command without arguments, display our help w/ an additional instructive message
    if not (
        hasattr(arguments, 'task_details')
        or (arguments.task and arguments.install)
        or (arguments.task and arguments.i)
    ):
        parser.error("You must specify either a task name + '-install', a task name + an image ('-i'), or "
                     "'-task-details'.")

    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

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
            # Pass any "extra" kwargs defined only for specific models/tasks
            extra_network_kwargs = {
                arg_name: getattr(arguments, arg_name)
                # "single_fold" -> used only by lesion_ms
                # "test_time_aug" -> used only by nnunetv2 models
                for arg_name in ["single_fold", "test_time_aug"]
                if hasattr(arguments, arg_name)
            }
            extra_inference_kwargs = {
                arg_name: getattr(arguments, arg_name)
                # "step1_only" -> used only by totalspineseg
                # "soft_ms_lesion" -> used only by lesion_ms
                for arg_name in ["step1_only", "soft_ms_lesion"]
                if hasattr(arguments, arg_name)
            }
            # The MS lesion model is multifold, which requires turning on the "ensemble averaging" behavior
            if arguments.task == 'lesion_ms':
                extra_inference_kwargs['ensemble'] = True
            # Run inference
            im_lst, target_lst = inference.segment_non_ivadomed(
                path_model, model_type, input_filenames, thr,
                # NOTE: contrast-agnostic nnunet model sometimes predicts pixels outside the cord, we want to
                # set keep_largest object as the default behaviour when using this model
                keep_largest=1 if arguments.task == 'spinalcord' else arguments.keep_largest,
                fill_holes_in_pred=arguments.fill_holes,
                remove_small=arguments.remove_small,
                use_gpu=use_gpu, remove_temp_files=arguments.r,
                # Pass any "extra" kwargs defined in task-specific subparsers
                extra_network_kwargs=extra_network_kwargs,
                extra_inference_kwargs=extra_inference_kwargs,
            )

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
        # Special case: totalspineseg which outputs 4-5 files per 1 input file
        elif arguments.task == 'totalspineseg':
            # `-step1-only: 1`: Use the 4th image ([3]) which represents the step1 output
            if getattr(arguments, "step1_only") == 1:
                assert len(output_filenames) == 4 * len(input_filenames)
                output_filenames_qc = output_filenames[3::4]
            # `-step1-only: 0`: Use the 5th image ([4]) which represents the step2 output
            else:
                assert len(output_filenames) == 5 * len(input_filenames)
                output_filenames_qc = output_filenames[4::5]
            iterator = zip(input_filenames, output_filenames_qc, [None] * len(input_filenames), qc_seg)
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
