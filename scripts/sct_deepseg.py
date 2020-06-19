#!/usr/bin/env python
# -*- coding: utf-8
"""
This command-line tool is the interface for the deepseg API that performs segmentation using deep learning from the
ivadomed package.
"""

# TODO: Add link to example image so users can decide wether their images look "close enough" to some of the proposed
#  models (e.g., mice, etc.).
# TODO: add test to make sure that all postprocessing flags match the core.DEFAULT dictionary items
# TODO: Fetch default value (and display) depending on the model that is used.
# TODO: accommodate multiclass segmentation


from __future__ import absolute_import

import sys
import os
import argparse
import colored

import spinalcordtoolbox as sct
import spinalcordtoolbox.deepseg.core
import spinalcordtoolbox.deepseg.models
import spinalcordtoolbox.utils

from sct_utils import init_sct, printv, display_viewer_syntax


def get_parser():

    parser = argparse.ArgumentParser(
        description="Segment an anatomical structure or pathologies according to the specified deep learning model.",
        add_help=None,
        formatter_class=sct.utils.SmartFormatter,
        prog=os.path.basename(__file__).strip(".py"))

    input_output = parser.add_argument_group("\nINPUT/OUTPUT")
    input_output.add_argument(
        "-i",
        help="Image to segment.",
        metavar=sct.utils.Metavar.file)
    input_output.add_argument(
        "-o",
        help="Output file name. In case of multi-class segmentation, class-specific suffixes will be added. By default,"
             "suffix '_seg' will be added and output extension will be .nii.gz.",
        metavar=sct.utils.Metavar.str)

    seg = parser.add_argument_group('\nTASKS')
    seg.add_argument(
        "-task",
        help="Task to perform. It could either be an official SCT task (in that case, simply enter the name of the "
             "task, example: -task segment_t2star_sc). To list official tasks, run: sct_deepseg -list-tasks.",
        metavar=sct.utils.Metavar.str)
    seg.add_argument(
        "-list-tasks",
        action='store_true',
        help="Display a list of tasks that can be achieved.")
    seg.add_argument(
        "-install-task",
        help="Install models that are required for specified task.",
        choices=list(sct.deepseg.models.TASKS.keys()))

    seg = parser.add_argument_group('\nMODELS')
    seg.add_argument(
        "-model",
        # TODO: add instructions at: https://github.com/neuropoly/ivado-medical-imaging
        help="Model to use. It could either be an official SCT model (in that case, simply enter the name of the "
             "model, example: -model t2_sc), or a path to the directory that contains a model, example: "
             "-model my_models/model. To list official models, run: sct_deepseg -list-models."
             "To build your own model, follow instructions at: https://github.com/neuropoly/ivado-medical-imaging",
        nargs='+',
        metavar=sct.utils.Metavar.str)
    seg.add_argument(
        "-list-models",
        action='store_true',
        help="Display a list of available models.")
    seg.add_argument(
        "-install-model",
        help="Install specified model.",
        choices=list(sct.deepseg.models.MODELS.keys()))

    misc = parser.add_argument_group('\nPARAMETERS')
    misc.add_argument(
        "-thr",
        type=float,
        help="Binarize segmentation with specified threshold. Set to 0 for no thresholding (i.e., soft segmentation). "
             "Default value is model-specific and was set during optimization "
             "(more info at https://github.com/sct-pipeline/deepseg-threshold).",
        metavar=sct.utils.Metavar.float,
        default=sct.deepseg.core.DEFAULTS['thr'])
    misc.add_argument(
        "-largest",
        type=int,
        help="Keep the largest connected-objects from the output segmentation. Specify the number of objects to keep."
             "To keep all objects, set to 0",
        default=sct.deepseg.core.DEFAULTS['largest'])
    misc.add_argument(
        "-fill-holes",
        type=int,
        help="Fill small holes in the segmentation.",
        choices=(0, 1),
        default=sct.deepseg.core.DEFAULTS['fill_holes'])

    misc = parser.add_argument_group('\nMISC')
    misc.add_argument(
        "-v",
        type=int,
        help="Verbose: 0 = no verbosity, 1 = verbose.",
        choices=(0, 1),
        default=1)
    misc.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    args = {k: v for k, v in vars(args).items() if v is not None}

    # Deal with model
    if args['list_models']:
        models = sct.deepseg.models.list_models()
        # Display beautiful output
        color = {True: 'green', False: 'red'}
        default = {True: '[*]', False: ''}
        print("{:<25s}DESCRIPTION".format("MODEL"))
        print("-" * 80)
        for name_model, value in models.items():
            path_model = sct.deepseg.models.folder(name_model)
            print("{}{}".format(
                colored.stylize(''.join((name_model, default[value['default']])).ljust(25),
                                colored.fg(color[sct.deepseg.models.is_valid(path_model)])),
                colored.stylize(value['description'],
                                colored.fg(color[sct.deepseg.models.is_valid(path_model)]))
                ))
        print(
            '\nLegend: {} | {} | default: {}\n'.format(
                colored.stylize("installed", colored.fg(color[True])),
                colored.stylize("not installed", colored.fg(color[False])),
                default[True]))
        exit(0)

    # Deal with task
    if args['list_tasks']:
        tasks = sct.deepseg.models.list_tasks()
        # Display beautiful output
        color = {True: 'green', False: 'red'}
        default = {True: '[*]', False: ''}
        print("{:<20s}{:<50s}MODELS".format("TASK", "DESCRIPTION"))
        print("-" * 80)
        for name_task, value in tasks.items():
            path_models = [sct.deepseg.models.folder(name_model) for name_model in value['models']]
            are_models_valid = [sct.deepseg.models.is_valid(path_model) for path_model in path_models]
            task_status = colored.stylize(name_task.ljust(20),
                                          colored.fg(color[all(are_models_valid)]))
            description_status = colored.stylize(value['description'].ljust(50),
                                                 colored.fg(color[all(are_models_valid)]))
            models_status = ', '.join([colored.stylize(model_name,
                                                       colored.fg(color[is_valid]))
                                       for model_name, is_valid in zip(value['models'], are_models_valid)])
            print("{}{}{}".format(task_status, description_status, models_status))

        print(
            '\nLegend: {} | {} | default: {}\n'.format(
                colored.stylize("installed", colored.fg(color[True])),
                colored.stylize("not installed", colored.fg(color[False])),
                default[True]))
        exit(0)

    if 'install_model' in args:
        sct.deepseg.models.install_model(args['install_model'])
        exit(0)

    if 'install_task' in args:
        for name_model in sct.deepseg.models.TASKS[args['install_task']]['models']:
            sct.deepseg.models.install_model(name_model)
        exit(0)

    # Deal with input/output
    if 'i' not in args:
        parser.error("The following arguments is required: -i")
    if not os.path.isfile(args['i']):
        parser.error("This file does not exist: {}".format(args['i']))

    # Check if at least a model or task has been specified
    if 'model' not in args and 'task' not in args:
        parser.error("You need to specify a model or a task.")

    # Get pipeline model names
    if 'task' in args:
        name_models = sct.deepseg.models.TASKS[args['task']]['models']

    if 'model' in args:
        name_models = args['model']

    # Run pipeline by iterating through the models
    fname_prior = None
    for name_model in name_models:
        # Check if this is an official model
        if name_model in list(sct.deepseg.models.MODELS.keys()):
            # If it is, check if it is installed
            path_model = spinalcordtoolbox.deepseg.models.folder(name_model)
            if not spinalcordtoolbox.deepseg.models.is_valid(path_model):
                printv("Model {} is not installed. Installing it now...".format(name_model))
                spinalcordtoolbox.deepseg.models.install_model(name_model)
        # If it is not, check if this is a path to a valid model
        else:
            path_model = os.path.abspath(name_model)
            if not sct.deepseg.models.is_valid(path_model):
                parser.error("The input model is invalid: {}".format(path_model))

        # Call segment_nifti
        fname_seg = sct.deepseg.core.segment_nifti(args['i'], path_model, fname_prior, args)
        # Use the result of the current model as additional input of the next model
        fname_prior = fname_seg

    display_viewer_syntax([args['i'], fname_seg], colormaps=['gray', 'red'], opacities=['', '0.7'])


if __name__ == '__main__':
    init_sct()
    main()
