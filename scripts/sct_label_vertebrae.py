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


from __future__ import division, absolute_import

import sys, os
import argparse
import numpy as np

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.vertebrae.core import create_label_z, get_z_and_disc_values_from_label, vertebral_detection, \
    clean_labeled_segmentation, label_discs, label_vert
from spinalcordtoolbox.vertebrae.detect_c2c3 import detect_c2c3
from spinalcordtoolbox.reports.qc import generate_qc
from spinalcordtoolbox.math import dilate

from sct_label_utils import ProcessLabels
# TODO: Properly test when first PR (that includes list_type) gets merged
from spinalcordtoolbox.utils import Metavar, SmartFormatter, ActionCreateFolder, list_type
import sct_utils as sct
import sct_straighten_spinalcord


# PARAMETERS
class Param:
    # The constructor
    def __init__(self):
        self.shift_AP = 32  # 0#32  # shift the centerline towards the spine (in voxel).
        self.size_AP = 11  # 41#11  # window size in AP direction (=y) (in voxel)
        self.size_RL = 1  # 1 # window size in RL direction (=x) (in voxel)
        self.size_IS = 19  # window size in IS direction (=z) (in voxel)
        self.shift_AP_visu = 15  # 0#15  # shift AP for displaying disc values
        self.smooth_factor = [3, 1, 1]  # [3, 1, 1]
        self.gaussian_std = 1.0  # STD of the Gaussian function, centered at the most rostral point of the image, and used to weight C2-C3 disk location finding towards the rostral portion of the FOV. Values to set between 0.1 (strong weighting) and 999 (no weighting).
        self.path_qc = None

    # update constructor with user's parameters
    def update(self, param_user):
        list_objects = param_user.split(',')
        for object in list_objects:
            if len(object) < 2:
                sct.printv('ERROR: Wrong usage.', 1, type='error')
            obj = object.split('=')
            if obj[0] == 'gaussian_std':
                setattr(self, obj[0], float(obj[1]))
            else:
                setattr(self, obj[0], int(obj[1]))


def get_parser():
    # initialize default param
    param_default = Param()
    # parser initialisation
    parser = argparse.ArgumentParser(
        description=(
            "This function takes an anatomical image and its cord segmentation (binary file), and outputs the "
            "cord segmentation labeled with vertebral level. The algorithm requires an initialization (first disc) and "
            "then performs a disc search in the superior, then inferior direction, using template disc matching based "
            "on mutual information score. The automatic method uses the module implemented in "
            "'spinalcordtoolbox/vertebrae/detect_c2c3.py' to detect the C2-C3 disc.\n"
            "Tips: To run the function with init txt file that includes flags -initz/-initcenter:\n"
            "  sct_label_vertebrae -i t2.nii.gz -s t2_seg-manual.nii.gz  '$(< init_label_vertebrae.txt)'"
        ),
        formatter_class=SmartFormatter,
        add_help=None,
        prog=os.path.basename(__file__).strip(".py")
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
        default=os.path.join(sct.__data_dir__, "PAM50"),
        help="Path to template."
    )
    optional.add_argument(
        '-initz',
        metavar=Metavar.list,
        type=list_type(',', int),
        help=("R|Initialize using slice number and disc value. Example: 68,4 (slice 68 corresponds to disc C3/C4). "
              "Example: 125,3\n"
              "WARNING: Slice number should correspond to superior-inferior direction (e.g. Z in RPI orientation, but "
              "Y in LIP orientation).")
    )
    optional.add_argument(
        '-initcenter',
        metavar=Metavar.int,
        type=int,
        help=("Initialize using disc value centered in the rostro-caudal direction. If the spine is curved, then "
              "consider the disc that projects onto the cord at the center of the z-FOV.")
    )
    optional.add_argument(
        '-initfile',
        metavar=Metavar.file,
        help="Initialize labeling by providing a text file which includes either -initz or -initcenter flag."
    )
    optional.add_argument(
        '-initlabel',
        metavar=Metavar.file,
        help=("Initialize vertebral labeling by providing a nifti file that has a single disc label. An example of "
              "such file is a single voxel with value '3', which would be located at the posterior tip of C2-C3 disc. "
              "Such label file can be created using: sct_label_utils -i IMAGE_REF -create-viewer 3 ; or by using the "
              "Python module 'detect_c2c3' implemented in 'spinalcordtoolbox/vertebrae/detect_c2c3.py'.")
    )
    optional.add_argument(
        '-discfile',
        metavar=Metavar.file,
        help=("File with disc labels, which will be used to transform the input segmentation into a vertebral level "
              "file. In that case, there will be no disc detection. The convention for disc labels is the following: "
              "value=3 -> disc C2/C3, value=4 -> disc C3/C4, etc.")
    )
    optional.add_argument(
        '-ofolder',
        metavar=Metavar.file,
        action=ActionCreateFolder,
        default='',
        help=("Output folder.")
    )
    optional.add_argument(
        '-denoise',
        choices=['0', '1'],
        default='0',
        help="Apply denoising filter to the data. Sometimes denoising is too aggressive, so use with care."
    )
    optional.add_argument(
        '-laplacian',
        choices=['0', '1'],
        default='0',
        help="Apply Laplacian filtering. More accurate but could mistake disc depending on anatomy."
    )
    optional.add_argument(
        '-scale-dist',
        metavar=Metavar.float,
        type=float,
        default=1.,
        help=("Scaling factor to adjust the average distance between two adjacent intervertebral discs. For example, "
              "if you are dealing with images from pediatric population, the distance should be reduced, so you can "
              "try a scaling factor of about 0.7.")
    )
    optional.add_argument(
        '-param',
        metavar=Metavar.list,
        type=list_type(',', str),
        help=(f"R|Advanced parameters. Assign value with \"=\"; Separate arguments with \",\"\n"
              f"  - shift_AP [mm]: AP shift of centerline for disc search. Default={param_default.shift_AP}.\n"
              f"  - size_AP [mm]: AP window size for disc search. Default={param_default.size_AP}.\n"
              f"  - size_RL [mm]: RL window size for disc search. Default={param_default.size_RL}.\n"
              f"  - size_IS [mm]: IS window size for disc search. Default={param_default.size_IS}.\n"
              f"  - gaussian_std [mm]: STD of the Gaussian function, centered at the most rostral point of the "
              f"image, and used to weight C2-C3 disk location finding towards the rostral portion of the FOV. Values "
              f"to set between 0.1 (strong weighting) and 999 (no weighting). "
              f"Default={param_default.gaussian_std}.\n")
    )
    optional.add_argument(
        '-r',
        choices=['0', '1'],
        default='1',
        help="Remove temporary files."
    )
    optional.add_argument(
        '-v',
        choices=['0', '1', '2'],
        default='1',
        help="Verbose. 0: nothing. 1: basic. 2: extended."
    )
    optional.add_argument(
        '-qc',
        metavar=Metavar.folder,
        action=ActionCreateFolder,
        default=param_default.path_qc,
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


def main(args=None):

    # initializations
    initz = ''
    initcenter = ''
    fname_initlabel = ''
    file_labelz = 'labelz.nii.gz'
    param = Param()

    # check user arguments
    parser = get_parser()
    if args:
        arguments = parser.parse_args(args)
    else:
        arguments = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    fname_in = os.path.abspath(arguments.i)
    fname_seg = os.path.abspath(arguments.s)
    contrast = arguments.c
    path_template = os.path.abspath(arguments.t)
    scale_dist = arguments.scale_dist
    path_output = arguments.ofolder
    param.path_qc = arguments.qc
    if arguments.discfile is not None:
        fname_disc = os.path.abspath(arguments.discfile)
    else:
        fname_disc = None
    if arguments.initz is not None:
        initz = arguments.initz
    if arguments.initcenter is not None:
        initcenter = arguments.initcenter
    # if user provided text file, parse and overwrite arguments
    if arguments.initfile is not None:
        file = open(arguments.initfile, 'r')
        initfile = ' ' + file.read().replace('\n', '')
        arg_initfile = initfile.split(' ')
        for idx_arg, arg in enumerate(arg_initfile):
            if arg == '-initz':
                initz = [int(x) for x in arg_initfile[idx_arg + 1].split(',')]
            if arg == '-initcenter':
                initcenter = int(arg_initfile[idx_arg + 1])
    if arguments.initlabel is not None:
        # get absolute path of label
        fname_initlabel = os.path.abspath(arguments.initlabel)
    if arguments.param is not None:
        param.update(arguments.param[0])
    verbose = int(arguments.v)
    sct.init_sct(log_level=verbose, update=True)  # Update log level
    remove_temp_files = int(arguments.r)
    denoise = int(arguments.denoise)
    laplacian = int(arguments.laplacian)

    path_tmp = sct.tmp_create(basename="label_vertebrae", verbose=verbose)

    # Copying input data to tmp folder
    sct.printv('\nCopying input data to tmp folder...', verbose)
    Image(fname_in).save(os.path.join(path_tmp, "data.nii"))
    Image(fname_seg).save(os.path.join(path_tmp, "segmentation.nii"))

    # Go go temp folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    # Straighten spinal cord
    sct.printv('\nStraighten spinal cord...', verbose)
    # check if warp_curve2straight and warp_straight2curve already exist (i.e. no need to do it another time)
    cache_sig = sct.cache_signature(
     input_files=[fname_in, fname_seg],
    )
    cachefile = os.path.join(curdir, "straightening.cache")
    if sct.cache_valid(cachefile, cache_sig) and os.path.isfile(os.path.join(curdir, "warp_curve2straight.nii.gz")) and os.path.isfile(os.path.join(curdir, "warp_straight2curve.nii.gz")) and os.path.isfile(os.path.join(curdir, "straight_ref.nii.gz")):
        # if they exist, copy them into current folder
        sct.printv('Reusing existing warping field which seems to be valid', verbose, 'warning')
        sct.copy(os.path.join(curdir, "warp_curve2straight.nii.gz"), 'warp_curve2straight.nii.gz')
        sct.copy(os.path.join(curdir, "warp_straight2curve.nii.gz"), 'warp_straight2curve.nii.gz')
        sct.copy(os.path.join(curdir, "straight_ref.nii.gz"), 'straight_ref.nii.gz')
        # apply straightening
        s, o = sct.run(['sct_apply_transfo', '-i', 'data.nii', '-w', 'warp_curve2straight.nii.gz', '-d', 'straight_ref.nii.gz', '-o', 'data_straight.nii'])
    else:
        sct_straighten_spinalcord.main(args=[
            '-i', 'data.nii',
            '-s', 'segmentation.nii',
            '-r', str(remove_temp_files),
            '-v', str(verbose),
        ])
        sct.cache_save(cachefile, cache_sig)

    # resample to 0.5mm isotropic to match template resolution
    sct.printv('\nResample to 0.5mm isotropic...', verbose)
    s, o = sct.run(['sct_resample', '-i', 'data_straight.nii', '-mm', '0.5x0.5x0.5', '-x', 'linear', '-o', 'data_straightr.nii'], verbose=verbose)

    # Apply straightening to segmentation
    # N.B. Output is RPI
    sct.printv('\nApply straightening to segmentation...', verbose)
    sct.run('isct_antsApplyTransforms -d 3 -i %s -r %s -t %s -o %s -n %s' %
            ('segmentation.nii',
             'data_straightr.nii',
             'warp_curve2straight.nii.gz',
             'segmentation_straight.nii',
             'Linear'),
            verbose=verbose,
            is_sct_binary=True,
           )
    # Threshold segmentation at 0.5
    sct.run(['sct_maths', '-i', 'segmentation_straight.nii', '-thr', '0.5', '-o', 'segmentation_straight.nii'], verbose)

    # If disc label file is provided, label vertebrae using that file instead of automatically
    if fname_disc:
        # Apply straightening to disc-label
        sct.printv('\nApply straightening to disc labels...', verbose)
        sct.run('isct_antsApplyTransforms -d 3 -i %s -r %s -t %s -o %s -n %s' %
                (fname_disc,
                 'data_straightr.nii',
                 'warp_curve2straight.nii.gz',
                 'labeldisc_straight.nii.gz',
                 'NearestNeighbor'),
                 verbose=verbose,
                 is_sct_binary=True,
                )
        label_vert('segmentation_straight.nii', 'labeldisc_straight.nii.gz', verbose=1)

    else:
        # create label to identify disc
        sct.printv('\nCreate label to identify disc...', verbose)
        fname_labelz = os.path.join(path_tmp, file_labelz)
        if initz or initcenter:
            if initcenter:
                # find z centered in FOV
                nii = Image('segmentation.nii').change_orientation("RPI")
                nx, ny, nz, nt, px, py, pz, pt = nii.dim  # Get dimensions
                z_center = int(np.round(nz / 2))  # get z_center
                initz = [z_center, initcenter]
            # create single label and output as labels.nii.gz
            label = ProcessLabels('segmentation.nii', fname_output='tmp.labelz.nii.gz',
                                      coordinates=['{},{}'.format(initz[0], initz[1])])
            im_label = label.process('create-seg')
            im_label.data = dilate(im_label.data, 3, 'ball')  # TODO: create a dilation method specific to labels,
            # which does not apply a convolution across all voxels (highly inneficient)
            im_label.save(fname_labelz)
        elif fname_initlabel:
            Image(fname_initlabel).save(fname_labelz)
        else:
            # automatically finds C2-C3 disc
            im_data = Image('data.nii')
            im_seg = Image('segmentation.nii')
            if not remove_temp_files:  # because verbose is here also used for keeping temp files
                verbose_detect_c2c3 = 2
            else:
                verbose_detect_c2c3 = 0
            im_label_c2c3 = detect_c2c3(im_data, im_seg, contrast, verbose=verbose_detect_c2c3)
            ind_label = np.where(im_label_c2c3.data)
            if not np.size(ind_label) == 0:
                im_label_c2c3.data[ind_label] = 3
            else:
                sct.printv('Automatic C2-C3 detection failed. Please provide manual label with sct_label_utils', 1, 'error')
                sys.exit()
            im_label_c2c3.save(fname_labelz)

        # dilate label so it is not lost when applying warping
        dilate(Image(fname_labelz), 3, 'ball').save(fname_labelz)

        # Apply straightening to z-label
        sct.printv('\nAnd apply straightening to label...', verbose)
        sct.run('isct_antsApplyTransforms -d 3 -i %s -r %s -t %s -o %s -n %s' %
                (file_labelz,
                 'data_straightr.nii',
                 'warp_curve2straight.nii.gz',
                 'labelz_straight.nii.gz',
                 'NearestNeighbor'),
                verbose=verbose,
                is_sct_binary=True,
               )
        # get z value and disk value to initialize labeling
        sct.printv('\nGet z and disc values from straight label...', verbose)
        init_disc = get_z_and_disc_values_from_label('labelz_straight.nii.gz')
        sct.printv('.. ' + str(init_disc), verbose)

        # denoise data
        if denoise:
            sct.printv('\nDenoise data...', verbose)
            sct.run(['sct_maths', '-i', 'data_straightr.nii', '-denoise', 'h=0.05', '-o', 'data_straightr.nii'], verbose)

        # apply laplacian filtering
        if laplacian:
            sct.printv('\nApply Laplacian filter...', verbose)
            sct.run(['sct_maths', '-i', 'data_straightr.nii', '-laplacian', '1', '-o', 'data_straightr.nii'], verbose)

        # detect vertebral levels on straight spinal cord
        init_disc[1]=init_disc[1]-1
        vertebral_detection('data_straightr.nii', 'segmentation_straight.nii', contrast, param, init_disc=init_disc,
                            verbose=verbose, path_template=path_template, path_output=path_output, scale_dist=scale_dist)

    # un-straighten labeled spinal cord
    sct.printv('\nUn-straighten labeling...', verbose)
    sct.run('isct_antsApplyTransforms -d 3 -i %s -r %s -t %s -o %s -n %s' %
            ('segmentation_straight_labeled.nii',
             'segmentation.nii',
             'warp_straight2curve.nii.gz',
             'segmentation_labeled.nii',
             'NearestNeighbor'),
            verbose=verbose,
            is_sct_binary=True,
           )
    # Clean labeled segmentation
    sct.printv('\nClean labeled segmentation (correct interpolation errors)...', verbose)
    clean_labeled_segmentation('segmentation_labeled.nii', 'segmentation.nii', 'segmentation_labeled.nii')

    # label discs
    sct.printv('\nLabel discs...', verbose)
    label_discs('segmentation_labeled.nii', verbose=verbose)

    # come back
    os.chdir(curdir)

    # Generate output files
    path_seg, file_seg, ext_seg = sct.extract_fname(fname_seg)
    fname_seg_labeled = os.path.join(path_output, file_seg + '_labeled' + ext_seg)
    sct.printv('\nGenerate output files...', verbose)
    sct.generate_output_file(os.path.join(path_tmp, "segmentation_labeled.nii"), fname_seg_labeled)
    sct.generate_output_file(os.path.join(path_tmp, "segmentation_labeled_disc.nii"), os.path.join(path_output, file_seg + '_labeled_discs' + ext_seg))
    # copy straightening files in case subsequent SCT functions need them
    sct.generate_output_file(os.path.join(path_tmp, "warp_curve2straight.nii.gz"), os.path.join(path_output, "warp_curve2straight.nii.gz"), verbose)
    sct.generate_output_file(os.path.join(path_tmp, "warp_straight2curve.nii.gz"), os.path.join(path_output, "warp_straight2curve.nii.gz"), verbose)
    sct.generate_output_file(os.path.join(path_tmp, "straight_ref.nii.gz"), os.path.join(path_output, "straight_ref.nii.gz"), verbose)

    # Remove temporary files
    if remove_temp_files == 1:
        sct.printv('\nRemove temporary files...', verbose)
        sct.rmtree(path_tmp)

    # Generate QC report
    if param.path_qc is not None:
        path_qc = os.path.abspath(arguments.qc)
        qc_dataset = arguments.qc_dataset
        qc_subject = arguments.qc_subject
        labeled_seg_file = os.path.join(path_output, file_seg + '_labeled' + ext_seg)
        generate_qc(fname_in, fname_seg=labeled_seg_file, args=args, path_qc=os.path.abspath(path_qc),
                    dataset=qc_dataset, subject=qc_subject, process='sct_label_vertebrae')

    sct.display_viewer_syntax([fname_in, fname_seg_labeled], colormaps=['', 'subcortical'], opacities=['1', '0.5'])


if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
