#!/usr/bin/env python
#########################################################################################
#
# Detect vertebral levels using cord centerline (or segmentation).
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Eugenie Ullmann, Karun Raju, Tanguy Duval, Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sys
import os
import argparse

import numpy as np

from spinalcordtoolbox.image import Image, generate_output_file
from spinalcordtoolbox.vertebrae.core import create_label_z, get_z_and_disc_values_from_label, vertebral_detection, \
    clean_labeled_segmentation, label_vert
from spinalcordtoolbox.vertebrae.detect_c2c3 import detect_c2c3
from spinalcordtoolbox.reports.qc import generate_qc
from spinalcordtoolbox.math import dilate
from spinalcordtoolbox.labels import create_labels_along_segmentation
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, ActionCreateFolder, list_type, display_viewer_syntax
from spinalcordtoolbox.utils.sys import init_sct, run_proc, printv, __data_dir__, set_loglevel
from spinalcordtoolbox.utils.fs import tmp_create, cache_signature, cache_valid, cache_save, \
    copy, extract_fname, rmtree
from spinalcordtoolbox.math import threshold, laplacian

from spinalcordtoolbox.scripts import sct_straighten_spinalcord, sct_apply_transfo


# for vertebral_detection
param_default = {
    'shift_AP': 32,  # shift the centerline towards the spine (in voxel).
    'size_AP': 11,  # window size in AP direction (=y) (in voxel)
    'size_RL': 1,  # window size in RL direction (=x) (in voxel)
    'size_IS': 19,  # window size in IS direction (=z) (in voxel)
    'shift_AP_visu': 15,  # shift AP for displaying disc values
}


def vertebral_detection_param(string):
    """Custom parser for vertebral_detection advanced parameters."""
    param = param_default.copy()
    for key_value in string.split(','):
        try:
            key, value = key_value.split('=', maxsplit=1)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f'advanced parameters should be of the form "parameter=value", got "{key_value}" instead')
        if key in param:
            try:
                param[key] = int(value)
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f'advanced parameter "{key}" needs an integer value, got "{value}" instead')
        elif key == 'gaussian_std':
            # TODO(issue#3706): remove 'gaussian_std' completely for v5.7
            printv('WARNING: gaussian_std parameter is currently ignored, '
                   'and will be removed in a later version.', 1, type='warning')
        else:
            raise argparse.ArgumentTypeError(f'Unknown advanced parameter: {key}')
    return param


def get_parser():
    parser = SCTArgumentParser(
        description=(
            "This function takes an anatomical image and its cord segmentation (binary file), and outputs the "
            "cord segmentation labeled with vertebral level. The algorithm requires an initialization (first disc) and "
            "then performs a disc search in the superior, then inferior direction, using template disc matching based "
            "on mutual information score. The automatic method uses the module implemented in "
            "'spinalcordtoolbox/vertebrae/detect_c2c3.py' to detect the C2-C3 disc."
        )
    )

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        '-i',
        metavar=Metavar.file,
        required=True,
        help="Input image. Example: t2.nii.gz"
    )
    mandatory.add_argument(
        '-s',
        metavar=Metavar.file,
        required=True,
        help="Segmentation of the spinal cord. Example: t2_seg.nii.gz"
    )
    mandatory.add_argument(
        '-c',
        choices=['t1', 't2'],
        required=True,
        help="Type of image contrast. 't2': cord dark / CSF bright. 't1': cord bright / CSF dark"
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit."
    )
    optional.add_argument(
        '-t',
        metavar=Metavar.folder,
        default=os.path.join(__data_dir__, "PAM50"),
        help="Path to template."
    )
    optional.add_argument(
        '-initz',
        metavar=Metavar.list,
        type=list_type(',', int),
        help="Initialize using slice number and disc value. Example: 68,4 (slice 68 corresponds to disc C3/C4). "
             "Example: 125,3\n"
             "WARNING: Slice number should correspond to superior-inferior direction (e.g. Z in RPI orientation, but "
             "Y in LIP orientation)."
    )
    optional.add_argument(
        '-initcenter',
        metavar=Metavar.int,
        type=int,
        help="Initialize using disc value centered in the rostro-caudal direction. If the spine is curved, then "
             "consider the disc that projects onto the cord at the center of the z-FOV."
    )
    optional.add_argument(
        '-initfile',
        metavar=Metavar.file,
        help="Initialize labeling by providing a text file which includes either -initz or -initcenter flag."
    )
    optional.add_argument(
        '-initlabel',
        metavar=Metavar.file,
        help="Initialize vertebral labeling by providing a nifti file that has a single disc label. An example of "
             "such file is a single voxel with value '3', which would be located at the posterior tip of C2-C3 disc. "
             "Such label file can be created using: sct_label_utils -i IMAGE_REF -create-viewer 3 ; or by using the "
             "Python module 'detect_c2c3' implemented in 'spinalcordtoolbox/vertebrae/detect_c2c3.py'."
    )
    optional.add_argument(
        '-discfile',
        metavar=Metavar.file,
        help="File with disc labels, which will be used to transform the input segmentation into a vertebral level "
             "file. In that case, there will be no disc detection. The convention for disc labels is the following: "
             "value=3 -> disc C2/C3, value=4 -> disc C3/C4, etc."
    )
    optional.add_argument(
        '-ofolder',
        metavar=Metavar.file,
        action=ActionCreateFolder,
        default='',
        help="Output folder."
    )
    optional.add_argument(
        '-laplacian',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=0,
        help="Apply Laplacian filtering. More accurate but could mistake disc depending on anatomy."
    )
    optional.add_argument(
        '-clean-labels',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=0,
        help="Clean output labeled segmentation to resemble original segmentation."
    )
    optional.add_argument(
        '-scale-dist',
        metavar=Metavar.float,
        type=float,
        default=1.,
        help="Scaling factor to adjust the average distance between two adjacent intervertebral discs. For example, "
             "if you are dealing with images from pediatric population, the distance should be reduced, so you can "
             "try a scaling factor of about 0.7."
    )
    optional.add_argument(
        '-param',
        metavar=Metavar.list,
        type=vertebral_detection_param,
        default=','.join(f'{key}={value}' for key, value in param_default.items()),
        help='Advanced parameters. Assign value with "="; Separate arguments with ","\n'
             '  - shift_AP [mm]: AP shift of centerline for disc search\n'
             '  - size_AP [mm]: AP window size for disc search\n'
             '  - size_RL [mm]: RL window size for disc search\n'
             '  - size_IS [mm]: IS window size for disc search\n',
    )
    optional.add_argument(
        '-r',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=1,
        help="Remove temporary files."
    )
    optional.add_argument(
        '-v',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode"
    )
    optional.add_argument(
        '-qc',
        metavar=Metavar.folder,
        action=ActionCreateFolder,
        help="The path where the quality control generated content will be saved."
    )
    optional.add_argument(
        '-qc-dataset',
        metavar=Metavar.str,
        help="If provided, this string will be mentioned in the QC report as the dataset the process was run on."
    )
    optional.add_argument(
        '-qc-subject',
        metavar=Metavar.str,
        help="If provided, this string will be mentioned in the QC report as the subject the process was run on."
    )

    return parser


def main(argv=None):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)

    fname_in = os.path.abspath(arguments.i)
    fname_seg = os.path.abspath(arguments.s)
    contrast = arguments.c
    path_template = os.path.abspath(arguments.t)
    scale_dist = arguments.scale_dist
    path_output = os.path.abspath(arguments.ofolder)
    fname_disc = arguments.discfile
    if fname_disc is not None:
        fname_disc = os.path.abspath(fname_disc)

    initz = arguments.initz
    initcenter = arguments.initcenter
    # if user provided text file, parse and overwrite arguments
    if arguments.initfile is not None:
        args = open(arguments.initfile).read().split()
        iterator = iter(args)
        for arg in iterator:
            if arg == '-initz':
                try:
                    initz = [int(x) for x in next(iterator).split(',')]
                except (StopIteration, ValueError):
                    parser.error('-initz takes two arguments: position in superior-inferior direction, label value')
            elif arg == '-initcenter':
                try:
                    initcenter = int(next(iterator))
                except (StopIteration, ValueError):
                    parser.error('-initcenter takes an integer argument')
    if initz is not None and len(initz) != 2:
        parser.error('-initz takes two arguments: position in superior-inferior direction, label value')

    fname_initlabel = arguments.initlabel
    if fname_initlabel is not None:
        fname_initlabel = os.path.abspath(fname_initlabel)
    remove_temp_files = arguments.r
    clean_labels = arguments.clean_labels

    path_tmp = tmp_create(basename="label_vertebrae")

    # Copying input data to tmp folder
    printv('\nCopying input data to tmp folder...', verbose)
    Image(fname_in).save(os.path.join(path_tmp, "data.nii"))
    Image(fname_seg).save(os.path.join(path_tmp, "segmentation.nii"))

    # Go go temp folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    # Straighten spinal cord
    printv('\nStraighten spinal cord...', verbose)
    # check if warp_curve2straight and warp_straight2curve already exist (i.e. no need to do it another time)
    cache_sig = cache_signature(
        input_files=[fname_in, fname_seg],
    )
    fname_cache = "straightening.cache"
    if (cache_valid(os.path.join(curdir, fname_cache), cache_sig)
            and os.path.isfile(os.path.join(curdir, "warp_curve2straight.nii.gz"))
            and os.path.isfile(os.path.join(curdir, "warp_straight2curve.nii.gz"))
            and os.path.isfile(os.path.join(curdir, "straight_ref.nii.gz"))):
        # if they exist, copy them into current folder
        printv('Reusing existing warping field which seems to be valid', verbose, 'warning')
        copy(os.path.join(curdir, "warp_curve2straight.nii.gz"), 'warp_curve2straight.nii.gz')
        copy(os.path.join(curdir, "warp_straight2curve.nii.gz"), 'warp_straight2curve.nii.gz')
        copy(os.path.join(curdir, "straight_ref.nii.gz"), 'straight_ref.nii.gz')
        # apply straightening
        s, o = run_proc(['sct_apply_transfo', '-i', 'data.nii', '-w', 'warp_curve2straight.nii.gz', '-d', 'straight_ref.nii.gz', '-o', 'data_straight.nii'])
    else:
        sct_straighten_spinalcord.main(argv=[
            '-i', 'data.nii',
            '-s', 'segmentation.nii',
            '-r', str(remove_temp_files),
            '-v', '0',
        ])
        cache_save(os.path.join(path_output, fname_cache), cache_sig)

    # resample to 0.5mm isotropic to match template resolution
    printv('\nResample to 0.5mm isotropic...', verbose)
    s, o = run_proc(['sct_resample', '-i', 'data_straight.nii', '-mm', '0.5x0.5x0.5', '-x', 'linear', '-o', 'data_straightr.nii'], verbose=verbose)

    # Apply straightening to segmentation
    # N.B. Output is RPI
    printv('\nApply straightening to segmentation...', verbose)
    sct_apply_transfo.main(['-i', 'segmentation.nii',
                            '-d', 'data_straightr.nii',
                            '-w', 'warp_curve2straight.nii.gz',
                            '-o', 'segmentation_straight.nii',
                            '-x', 'linear',
                            '-v', '0'])

    # Threshold segmentation at 0.5
    img = Image('segmentation_straight.nii')
    img.data = threshold(img.data, 0.5)
    img.save()


    # If disc label file is provided, label vertebrae using that file instead of automatically
    if fname_disc:
        # Apply straightening to disc-label
        printv('\nApply straightening to disc labels...', verbose)
        run_proc('sct_apply_transfo -i %s -d %s -w %s -o %s -x %s' %
                 (fname_disc,
                  'data_straightr.nii',
                  'warp_curve2straight.nii.gz',
                  'labeldisc_straight.nii.gz',
                  'label'),
                 verbose=verbose
                 )
        label_vert('segmentation_straight.nii', 'labeldisc_straight.nii.gz', verbose=1)

    else:
        printv('\nCreate label to identify disc...', verbose)
        fname_labelz = os.path.join(path_tmp, 'labelz.nii.gz')
        if initcenter is not None:
            # find z centered in FOV
            nii = Image('segmentation.nii').change_orientation("RPI")
            nx, ny, nz, nt, px, py, pz, pt = nii.dim
            z_center = round(nz / 2)
            initz = [z_center, initcenter]
        if initz is not None:
            im_label = create_labels_along_segmentation(Image('segmentation.nii'), [tuple(initz)])
            im_label.save(fname_labelz)
        elif fname_initlabel is not None:
            Image(fname_initlabel).save(fname_labelz)
        else:
            # automatically finds C2-C3 disc
            im_data = Image('data.nii')
            im_seg = Image('segmentation.nii')
            # because verbose is also used for keeping temp files
            verbose_detect_c2c3 = 0 if remove_temp_files else 2
            im_label_c2c3 = detect_c2c3(im_data, im_seg, contrast, verbose=verbose_detect_c2c3)
            ind_label = np.where(im_label_c2c3.data)
            if np.size(ind_label) == 0:
                printv('Automatic C2-C3 detection failed. Please provide manual label with sct_label_utils', 1, 'error')
                sys.exit(1)
            im_label_c2c3.data[ind_label] = 3
            im_label_c2c3.save(fname_labelz)

        # dilate label so it is not lost when applying warping
        dilate(Image(fname_labelz), 3, 'ball').save(fname_labelz)

        # Apply straightening to z-label
        printv('\nAnd apply straightening to label...', verbose)
        sct_apply_transfo.main(['-i', 'labelz.nii.gz',
                                '-d', 'data_straightr.nii',
                                '-w', 'warp_curve2straight.nii.gz',
                                '-o', 'labelz_straight.nii.gz',
                                '-x', 'nn',
                                '-v', '0'])
        # get z value and disk value to initialize labeling
        printv('\nGet z and disc values from straight label...', verbose)
        init_disc = get_z_and_disc_values_from_label('labelz_straight.nii.gz')
        printv('.. ' + str(init_disc), verbose)

        # apply laplacian filtering
        if arguments.laplacian:
            printv('\nApply Laplacian filter...', verbose)
            img = Image("data_straightr.nii")

            # apply std dev to each axis of the image
            sigmas = [1 for i in range(len(img.data.shape))]

            # adjust sigma based on voxel size
            sigmas = [sigmas[i] / img.dim[i + 4] for i in range(3)]

            # smooth data
            img.data = laplacian(img.data, sigmas)
            img.save()

        # detect vertebral levels on straight spinal cord
        init_disc[1] = init_disc[1] - 1
        vertebral_detection('data_straightr.nii', 'segmentation_straight.nii', contrast, arguments.param, init_disc=init_disc,
                            verbose=verbose, path_template=path_template, path_output=path_output, scale_dist=scale_dist)

    # un-straighten labeled spinal cord
    printv('\nUn-straighten labeling...', verbose)
    sct_apply_transfo.main(['-i', 'segmentation_straight_labeled.nii',
                            '-d', 'segmentation.nii',
                            '-w', 'warp_straight2curve.nii.gz',
                            '-o', 'segmentation_labeled.nii',
                            '-x', 'nn',
                            '-v', '0'])

    if clean_labels:
        # Clean labeled segmentation
        printv('\nClean labeled segmentation (correct interpolation errors)...', verbose)
        clean_labeled_segmentation('segmentation_labeled.nii', 'segmentation.nii', 'segmentation_labeled.nii')

    # label discs
    printv('\nLabel discs...', verbose)
    printv('\nUn-straighten labeled discs...', verbose)
    run_proc('sct_apply_transfo -i %s -d %s -w %s -o %s -x %s' %
             ('segmentation_straight_labeled_disc.nii',
              'segmentation.nii',
              'warp_straight2curve.nii.gz',
              'segmentation_labeled_disc.nii',
              'label'),
             verbose=verbose,
             is_sct_binary=True,
             )


    # come back
    os.chdir(curdir)

    # Generate output files
    path_seg, file_seg, ext_seg = extract_fname(fname_seg)
    fname_seg_labeled = os.path.join(path_output, file_seg + '_labeled' + ext_seg)
    printv('\nGenerate output files...', verbose)
    generate_output_file(os.path.join(path_tmp, "segmentation_labeled.nii"), fname_seg_labeled)
    generate_output_file(os.path.join(path_tmp, "segmentation_labeled_disc.nii"), os.path.join(path_output, file_seg + '_labeled_discs' + ext_seg))
    # copy straightening files in case subsequent SCT functions need them
    generate_output_file(os.path.join(path_tmp, "warp_curve2straight.nii.gz"), os.path.join(path_output, "warp_curve2straight.nii.gz"), verbose=verbose)
    generate_output_file(os.path.join(path_tmp, "warp_straight2curve.nii.gz"), os.path.join(path_output, "warp_straight2curve.nii.gz"), verbose=verbose)
    generate_output_file(os.path.join(path_tmp, "straight_ref.nii.gz"), os.path.join(path_output, "straight_ref.nii.gz"), verbose=verbose)

    # Remove temporary files
    if remove_temp_files == 1:
        printv('\nRemove temporary files...', verbose)
        rmtree(path_tmp)

    # Generate QC report
    if arguments.qc is not None:
        path_qc = os.path.abspath(arguments.qc)
        qc_dataset = arguments.qc_dataset
        qc_subject = arguments.qc_subject
        labeled_seg_file = os.path.join(path_output, file_seg + '_labeled' + ext_seg)
        generate_qc(fname_in, fname_seg=labeled_seg_file, args=argv, path_qc=os.path.abspath(path_qc),
                    dataset=qc_dataset, subject=qc_subject, process='sct_label_vertebrae')

    display_viewer_syntax([fname_in, fname_seg_labeled], colormaps=['', 'subcortical'], opacities=['1', '0.5'])


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])

