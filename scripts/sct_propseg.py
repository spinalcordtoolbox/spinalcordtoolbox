#!/usr/bin/env python
#########################################################################################
#
# Parser for PropSeg binary.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener
# Modified: 2015-03-03
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: remove temp files in case rescaled is not "1"

from __future__ import division, absolute_import

import os, sys

import numpy as np
from scipy import ndimage as ndi

import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.image import Image
import sct_image
import sct_utils as sct
from msct_parser import Parser
from spinalcordtoolbox.centerline import optic
from spinalcordtoolbox.reports.qc import generate_qc


def check_and_correct_segmentation(fname_segmentation, fname_centerline, folder_output='', threshold_distance=5.0,
                                   remove_temp_files=1, verbose=0):
    """
    This function takes the outputs of isct_propseg (centerline and segmentation) and check if the centerline of the
    segmentation is coherent with the centerline provided by the isct_propseg, especially on the edges (related
    to issue #1074).
    Args:
        fname_segmentation: filename of binary segmentation
        fname_centerline: filename of binary centerline
        threshold_distance: threshold, in mm, beyond which centerlines are not coherent
        verbose:

    Returns: None
    """
    sct.printv('\nCheck consistency of segmentation...', verbose)
    # creating a temporary folder in which all temporary files will be placed and deleted afterwards
    path_tmp = sct.tmp_create(basename="propseg", verbose=verbose)
    from sct_convert import convert
    convert(fname_segmentation, os.path.join(path_tmp, "tmp.segmentation.nii.gz"), verbose=0)
    convert(fname_centerline, os.path.join(path_tmp, "tmp.centerline.nii.gz"), verbose=0)
    fname_seg_absolute = os.path.abspath(fname_segmentation)

    # go to tmp folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    # convert segmentation image to RPI
    im_input = Image('tmp.segmentation.nii.gz')
    image_input_orientation = im_input.orientation

    sct_image.main("-i tmp.segmentation.nii.gz -setorient RPI -o tmp.segmentation_RPI.nii.gz -v 0".split())
    sct_image.main("-i tmp.centerline.nii.gz -setorient RPI -o tmp.centerline_RPI.nii.gz -v 0".split())

    # go through segmentation image, and compare with centerline from propseg
    im_seg = Image('tmp.segmentation_RPI.nii.gz')
    im_centerline = Image('tmp.centerline_RPI.nii.gz')

    # Get size of data
    sct.printv('\nGet data dimensions...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = im_seg.dim

    # extraction of centerline provided by isct_propseg and computation of center of mass for each slice
    # the centerline is defined as the center of the tubular mesh outputed by propseg.
    centerline, key_centerline = {}, []
    for i in range(nz):
        slice = im_centerline.data[:, :, i]
        if np.any(slice):
            x_centerline, y_centerline = ndi.measurements.center_of_mass(slice)
            centerline[str(i)] = [x_centerline, y_centerline]
            key_centerline.append(i)

    minz_centerline = np.min(key_centerline)
    maxz_centerline = np.max(key_centerline)
    mid_slice = int((maxz_centerline - minz_centerline) / 2)

    # for each slice of the segmentation, check if only one object is present. If not, remove the slice from segmentation.
    # If only one object (the spinal cord) is present in the slice, check if its center of mass is close to the centerline of isct_propseg.
    slices_to_remove = [False] * nz  # flag that decides if the slice must be removed
    for i in range(minz_centerline, maxz_centerline + 1):
        # extraction of slice
        slice = im_seg.data[:, :, i]
        distance = -1
        label_objects, nb_labels = ndi.label(slice)  # count binary objects in the slice
        if nb_labels > 1:  # if there is more that one object in the slice, the slice is removed from the segmentation
            slices_to_remove[i] = True
        elif nb_labels == 1:  # check if the centerline is coherent with the one from isct_propseg
            x_centerline, y_centerline = ndi.measurements.center_of_mass(slice)
            slice_nearest_coord = min(key_centerline, key=lambda x: abs(x - i))
            coord_nearest_coord = centerline[str(slice_nearest_coord)]
            distance = np.sqrt(((x_centerline - coord_nearest_coord[0]) * px) ** 2 +
                               ((y_centerline - coord_nearest_coord[1]) * py) ** 2 +
                               ((i - slice_nearest_coord) * pz) ** 2)

            if distance >= threshold_distance:  # threshold must be adjusted, default is 5 mm
                slices_to_remove[i] = True

    # Check list of removal and keep one continuous centerline (improve this comment)
    # Method:
    # starting from mid-centerline (in both directions), the first True encountered is applied to all following slices
    slice_to_change = False
    for i in range(mid_slice, nz):
        if slice_to_change:
            slices_to_remove[i] = True
        elif slices_to_remove[i]:
            slice_to_change = True

    slice_to_change = False
    for i in range(mid_slice, 0, -1):
        if slice_to_change:
            slices_to_remove[i] = True
        elif slices_to_remove[i]:
            slice_to_change = True

    for i in range(0, nz):
        # remove the slice
        if slices_to_remove[i]:
            im_seg.data[:, :, i] *= 0

    # saving the image
    im_seg.save('tmp.segmentation_RPI_c.nii.gz')

    # replacing old segmentation with the corrected one
    sct_image.main('-i tmp.segmentation_RPI_c.nii.gz -setorient {} -o {} -v 0'.
                   format(image_input_orientation, fname_seg_absolute).split())

    os.chdir(curdir)

    # display information about how much of the segmentation has been corrected

    # remove temporary files
    if remove_temp_files:
        # sct.printv("\nRemove temporary files...", verbose)
        sct.rmtree(path_tmp)


def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('''This program segments automatically the spinal cord on T1- and T2-weighted images, for any field of view. You must provide the type of contrast, the image as well as the output folder path.
The segmentation follows the spinal cord centerline, which is provided by an automatic tool: Optic. The initialization of the segmentation is made on the median slice of the centerline, and can be ajusted using the -init parameter. The initial radius of the tubular mesh that will be propagated should be adapted to size of the spinal cord on the initial propagation slice.
Primary output is the binary mask of the spinal cord segmentation. This method must provide VTK triangular mesh of the segmentation (option -mesh). Spinal cord centerline is available as a binary image (-centerline-binary) or a text file with coordinates in world referential (-centerline-coord).
Cross-sectional areas along the spinal cord can be available (-cross).
Several tips on segmentation correction can be found on the "Correction Tips" page of the documentation while advices on parameters adjustments can be found on the "Parameters" page.
If the segmentation fails at some location (e.g. due to poor contrast between spinal cord and CSF), edit your anatomical image (e.g. with fslview) and manually enhance the contrast by adding bright values around the spinal cord for T2-weighted images (dark values for T1-weighted). Then, launch the segmentation again.''')
    parser.add_option(name="-i",
                      type_value="image_nifti",
                      description="input image.",
                      mandatory=True,
                      example="t1.nii.gz")
    parser.add_option(name="-t",
                      type_value="multiple_choice",
                      description="type of image contrast, t2: cord dark / CSF bright ; t1: cord bright / CSF dark",
                      mandatory=False,
                      deprecated=1,
                      deprecated_by="-c",
                      example=['t1', 't2'])
    parser.add_option(name="-c",
                      type_value="multiple_choice",
                      description="type of image contrast, if your contrast is not in the available options (t1, t2, t2s, dwi), use t1 (cord bright / CSF dark) or t2 (cord dark / CSF bright)",
                      mandatory=True,
                      example=['t1', 't2', 't2s', 'dwi'])
    parser.usage.addSection("General options")
    parser.add_option(name="-ofolder",
                      type_value="folder_creation",
                      description="output folder.",
                      mandatory=False,
                      example="My_Output_Folder/",
                      default_value="")
    parser.add_option(name="-down",
                      type_value="int",
                      description="down limit of the propagation, default is 0",
                      mandatory=False)
    parser.add_option(name="-up",
                      type_value="int",
                      description="up limit of the propagation, default is the highest slice of the image",
                      mandatory=False)
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="remove temporary files.",
                      mandatory=False,
                      example=['0', '1'],
                      default_value='1')
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="1: display on, 0: display off (default)",
                      mandatory=False,
                      example=["0", "1"],
                      default_value="1")
    parser.add_option(name="-h",
                      type_value=None,
                      description="display this help",
                      mandatory=False)

    parser.usage.addSection("\nOutput options")
    parser.add_option(name="-mesh",
                      type_value=None,
                      description="output: mesh of the spinal cord segmentation",
                      mandatory=False)
    parser.add_option(name="-centerline-binary",
                      type_value=None,
                      description="output: centerline as a binary image ",
                      mandatory=False)
    parser.add_option(name="-CSF",
                      type_value=None,
                      description="output: CSF segmentation",
                      mandatory=False)
    parser.add_option(name="-centerline-coord",
                      type_value=None,
                      description="output: centerline in world coordinates",
                      mandatory=False)
    parser.add_option(name="-cross",
                      type_value=None,
                      description="output: cross-sectional areas",
                      mandatory=False)
    parser.add_option(name="-init-tube",
                      type_value=None,
                      description="output: initial tubular meshes ",
                      mandatory=False)
    parser.add_option(name="-low-resolution-mesh",
                      type_value=None,
                      description="output: low-resolution mesh",
                      mandatory=False)

    parser.usage.addSection("\nOptions helping the segmentation")
    parser.add_option(name="-init-centerline",
                      type_value="image_nifti",
                      description="filename of centerline to use for the propagation, "
                                  "format .txt or .nii, see file structure in documentation."
                                  "\nReplace filename by 'viewer' to use interactive viewer for providing centerline. "
                                  "Ex: -init-centerline viewer",
                      mandatory=False,
                      list_no_image=['viewer', 'hough', 'optic'])
    parser.add_option(name="-init",
                      type_value="float",
                      description="axial slice where the propagation starts, default is middle axial slice",
                      mandatory=False)
    parser.add_option(name="-init-mask",
                      type_value="image_nifti",
                      description="mask containing three center of the spinal cord, used to initiate the propagation.\nReplace filename by 'viewer' to use interactive viewer for providing mask. Ex: -init-mask viewer",
                      mandatory=False,
                      list_no_image=['viewer'])
    parser.add_option(name="-mask-correction",
                      type_value="image_nifti",
                      description="mask containing binary pixels at edges of the spinal cord on which the segmentation algorithm will be forced to register the surface. Can be used in case of poor/missing contrast between spinal cord and CSF or in the presence of artefacts/pathologies.",
                      mandatory=False)
    parser.add_option(name="-rescale",
                      type_value="float",
                      description="Rescale the image (only the header, not the data) in order to enable segmentation "
                                  "on spinal cords with dimensions different than that of humans (e.g., mice, rats, "
                                  "elephants, etc.). For example, if the spinal cord is 2x smaller than that of human,"
                                  "then use -rescale 2",
                      default_value=1,
                      mandatory=False)
    parser.add_option(name="-radius",
                      type_value="float",
                      description="approximate radius (in mm) of the spinal cord, default is 4",
                      mandatory=False)
    parser.add_option(name="-nbiter",
                      type_value="int",
                      description="stop condition (affects only the Z propogation): number of iteration for the propagation for both direction, default is 200",
                      mandatory=False)
    parser.add_option(name="-max-area",
                      type_value="float",
                      description="[mm^2], stop condition (affects only the Z propogation): maximum cross-sectional area, default is 120",
                      mandatory=False)
    parser.add_option(name="-max-deformation",
                      type_value="float",
                      description="[mm], stop condition (affects only the Z propogation): maximum deformation per iteration, default is 2.5",
                      mandatory=False)
    parser.add_option(name="-min-contrast",
                      type_value="float",
                      description="[intensity value], stop condition (affects only the Z propogation): minimum local SC/CSF contrast, default is 50",
                      mandatory=False)
    parser.add_option(name="-d",
                      type_value="float",
                      description="trade-off between distance of most promising point (d is high) and feature strength (d is low), default depend on the contrast. Range of values from 0 to 50. 15-25 values show good results, default is 10",
                      mandatory=False)
    parser.add_option(name="-distance-search",
                      type_value="float",
                      description="maximum distance of optimal points computation along the surface normals. Range of values from 0 to 30, default is 15",
                      mandatory=False)
    parser.add_option(name="-alpha",
                      type_value="float",
                      description="trade-off between internal (alpha is high) and external (alpha is low) forces. Range of values from 0 to 50, default is 25",
                      mandatory=False)
    parser.add_option(name='-qc',
                      type_value='folder_creation',
                      description='The path where the quality control generated content will be saved',
                      default_value=None)
    parser.add_option(name='-qc-dataset',
                      type_value='str',
                      description='If provided, this string will be mentioned in the QC report as the dataset the process was run on',
                      )
    parser.add_option(name='-qc-subject',
                      type_value='str',
                      description='If provided, this string will be mentioned in the QC report as the subject the process was run on',
                      )
    parser.add_option(name='-correct-seg',
                      type_value="multiple_choice",
                      description="Enable (1) or disable (0) the algorithm that checks and correct the output "
                                  "segmentation. More specifically, the algorithm checks if the segmentation is "
                                  "consistent with the centerline provided by isct_propseg.",
                      mandatory=False,
                      example=["0", "1"],
                      default_value="1")
    parser.add_option(name='-igt',
                      type_value='image_nifti',
                      description='File name of ground-truth segmentation.',
                      mandatory=False)

    # DEPRECATED OPTIONS
    parser.add_option(name="-detect-nii",
                      type_value=None,
                      description="output: spinal cord detection as a nifti image",
                      mandatory=False,
                      deprecated=True)
    parser.add_option(name="-detect-png",
                      type_value=None,
                      description="output: spinal cord detection as a PNG image",
                      mandatory=False,
                      deprecated=True)
    parser.add_option(name="-detect-n",
                      type_value="int",
                      description="number of axial slices computed in the detection process, default is 4",
                      mandatory=False,
                      deprecated=True)
    parser.add_option(name="-detect-gap",
                      type_value="int",
                      description="gap along Z direction (in mm) for the detection process, default is 4",
                      mandatory=False,
                      deprecated=True)
    parser.add_option(name="-init-validation",
                      type_value=None,
                      description="enable validation on spinal cord detection based on discriminant analysis",
                      mandatory=False,
                      deprecated=True)

    return parser


def func_rescale_header(fname_data, rescale_factor, verbose=0):
    """
    Rescale the voxel dimension by modifying the NIFTI header qform. Write the output file in a temp folder.
    :param fname_data:
    :param rescale_factor:
    :return: fname_data_rescaled
    """
    import nibabel as nib
    img = nib.load(fname_data)
    # get qform
    qform = img.header.get_qform()
    # multiply by scaling factor
    qform[0:3, 0:3] *= rescale_factor
    # generate a new nifti file
    header_rescaled = img.header.copy()
    header_rescaled.set_qform(qform)
    # the data are the same-- only the header changes
    img_rescaled = nib.nifti1.Nifti1Image(img.get_data(), None, header=header_rescaled)
    path_tmp = sct.tmp_create(basename="propseg", verbose=verbose)
    fname_data_rescaled = os.path.join(path_tmp, os.path.basename(sct.add_suffix(fname_data, "_rescaled")))
    nib.save(img_rescaled, fname_data_rescaled)
    return fname_data_rescaled



def propseg(img_input, options_dict):
    """
    :param img_input: source image, to be segmented
    :param options_dict: arguments as dictionary
    :return: segmented Image
    """
    arguments = options_dict
    fname_input_data = img_input.absolutepath
    fname_data = os.path.abspath(fname_input_data)
    contrast_type = arguments["-c"]
    contrast_type_conversion = {'t1': 't1', 't2': 't2', 't2s': 't2', 'dwi': 't1'}
    contrast_type_propseg = contrast_type_conversion[contrast_type]

    # Starting building the command
    cmd = ['isct_propseg', '-t', contrast_type_propseg]

    if "-ofolder" in arguments:
        folder_output = arguments["-ofolder"]
    else:
        folder_output = './'
    cmd += ['-o', folder_output]
    if not os.path.isdir(folder_output) and os.path.exists(folder_output):
        logger.error("output directory %s is not a valid directory" % folder_output)
    if not os.path.exists(folder_output):
        os.makedirs(folder_output)

    if "-down" in arguments:
        cmd += ["-down", str(arguments["-down"])]
    if "-up" in arguments:
        cmd += ["-up", str(arguments["-up"])]

    remove_temp_files = 1
    if "-r" in arguments:
        remove_temp_files = int(arguments["-r"])

    verbose = int(arguments.get('-v'))
    sct.init_sct(log_level=verbose, update=True)  # Update log level
    # Update for propseg binary
    if verbose > 0:
        cmd += ["-verbose"]

    # Output options
    if "-mesh" in arguments:
        cmd += ["-mesh"]
    if "-centerline-binary" in arguments:
        cmd += ["-centerline-binary"]
    if "-CSF" in arguments:
        cmd += ["-CSF"]
    if "-centerline-coord" in arguments:
        cmd += ["-centerline-coord"]
    if "-cross" in arguments:
        cmd += ["-cross"]
    if "-init-tube" in arguments:
        cmd += ["-init-tube"]
    if "-low-resolution-mesh" in arguments:
        cmd += ["-low-resolution-mesh"]
    if "-detect-nii" in arguments:
        cmd += ["-detect-nii"]
    if "-detect-png" in arguments:
        cmd += ["-detect-png"]

    # Helping options
    use_viewer = None
    use_optic = True  # enabled by default
    init_option = None
    rescale_header = arguments["-rescale"]
    if "-init" in arguments:
        init_option = float(arguments["-init"])
        if init_option < 0:
            sct.printv('Command-line usage error: ' + str(init_option) + " is not a valid value for '-init'", 1, 'error')
            sys.exit(1)
    if "-init-centerline" in arguments:
        if str(arguments["-init-centerline"]) == "viewer":
            use_viewer = "centerline"
        elif str(arguments["-init-centerline"]) == "hough":
            use_optic = False
        else:
            if rescale_header is not 1:
                fname_labels_viewer = func_rescale_header(str(arguments["-init-centerline"]), rescale_header, verbose=verbose)
            else:
                fname_labels_viewer = str(arguments["-init-centerline"])
            cmd += ["-init-centerline", fname_labels_viewer]
            use_optic = False
    if "-init-mask" in arguments:
        if str(arguments["-init-mask"]) == "viewer":
            use_viewer = "mask"
        else:
            if rescale_header is not 1:
                fname_labels_viewer = func_rescale_header(str(arguments["-init-mask"]), rescale_header)
            else:
                fname_labels_viewer = str(arguments["-init-mask"])
            cmd += ["-init-mask", fname_labels_viewer]
            use_optic = False
    if "-mask-correction" in arguments:
        cmd += ["-mask-correction", str(arguments["-mask-correction"])]
    if "-radius" in arguments:
        cmd += ["-radius", str(arguments["-radius"])]
    if "-detect-n" in arguments:
        cmd += ["-detect-n", str(arguments["-detect-n"])]
    if "-detect-gap" in arguments:
        cmd += ["-detect-gap", str(arguments["-detect-gap"])]
    if "-init-validation" in arguments:
        cmd += ["-init-validation"]
    if "-nbiter" in arguments:
        cmd += ["-nbiter", str(arguments["-nbiter"])]
    if "-max-area" in arguments:
        cmd += ["-max-area", str(arguments["-max-area"])]
    if "-max-deformation" in arguments:
        cmd += ["-max-deformation", str(arguments["-max-deformation"])]
    if "-min-contrast" in arguments:
        cmd += ["-min-contrast", str(arguments["-min-contrast"])]
    if "-d" in arguments:
        cmd += ["-d", str(arguments["-d"])]
    if "-distance-search" in arguments:
        cmd += ["-dsearch", str(arguments["-distance-search"])]
    if "-alpha" in arguments:
        cmd += ["-alpha", str(arguments["-alpha"])]

    # check if input image is in 3D. Otherwise itk image reader will cut the 4D image in 3D volumes and only take the first one.
    image_input = Image(fname_data)
    image_input_rpi = image_input.copy().change_orientation('RPI')
    nx, ny, nz, nt, px, py, pz, pt = image_input_rpi.dim
    if nt > 1:
        sct.printv('ERROR: your input image needs to be 3D in order to be segmented.', 1, 'error')

    path_data, file_data, ext_data = sct.extract_fname(fname_data)
    path_tmp = sct.tmp_create(basename="label_vertebrae", verbose=verbose)

    # rescale header (see issue #1406)
    if rescale_header is not 1:
        fname_data_propseg = func_rescale_header(fname_data, rescale_header)
    else:
        fname_data_propseg = fname_data

    # add to command
    cmd += ['-i', fname_data_propseg]

    # if centerline or mask is asked using viewer
    if use_viewer:
        from spinalcordtoolbox.gui.base import AnatomicalParams
        from spinalcordtoolbox.gui.centerline import launch_centerline_dialog

        params = AnatomicalParams()
        if use_viewer == 'mask':
            params.num_points = 3
            params.interval_in_mm = 15  # superior-inferior interval between two consecutive labels
            params.starting_slice = 'midfovminusinterval'
        if use_viewer == 'centerline':
            # setting maximum number of points to a reasonable value
            params.num_points = 20
            params.interval_in_mm = 30
            params.starting_slice = 'top'
        im_data = Image(fname_data_propseg)

        im_mask_viewer = msct_image.zeros_like(im_data)
        # im_mask_viewer.absolutepath = sct.add_suffix(fname_data_propseg, '_labels_viewer')
        controller = launch_centerline_dialog(im_data, im_mask_viewer, params)
        fname_labels_viewer = sct.add_suffix(fname_data_propseg, '_labels_viewer')

        if not controller.saved:
            sct.printv('The viewer has been closed before entering all manual points. Please try again.', 1, 'error')
            sys.exit(1)
        # save labels
        controller.as_niftii(fname_labels_viewer)

        # add mask filename to parameters string
        if use_viewer == "centerline":
            cmd += ["-init-centerline", fname_labels_viewer]
        elif use_viewer == "mask":
            cmd += ["-init-mask", fname_labels_viewer]

    # If using OptiC
    elif use_optic:
        image_centerline = optic.detect_centerline(image_input, contrast_type, verbose)
        fname_centerline_optic = os.path.join(path_tmp, 'centerline_optic.nii.gz')
        image_centerline.save(fname_centerline_optic)
        cmd += ["-init-centerline", fname_centerline_optic]

    if init_option is not None:
        if init_option > 1:
            init_option /= (nz - 1)
        cmd += ['-init', str(init_option)]

    # enabling centerline extraction by default (needed by check_and_correct_segmentation() )
    cmd += ['-centerline-binary']

    # run propseg
    status, output = sct.run(cmd, verbose, raise_exception=False, is_sct_binary=True)

    # check status is not 0
    if not status == 0:
        sct.printv('Automatic cord detection failed. Please initialize using -init-centerline or -init-mask (see help)',
                   1, 'error')
        sys.exit(1)

    # build output filename
    fname_seg = os.path.join(folder_output, os.path.basename(sct.add_suffix(fname_data, "_seg")))
    fname_centerline = os.path.join(folder_output, os.path.basename(sct.add_suffix(fname_data, "_centerline")))
    # in case header was rescaled, we need to update the output file names by removing the "_rescaled"
    if rescale_header is not 1:
        sct.mv(os.path.join(folder_output, sct.add_suffix(os.path.basename(fname_data_propseg), "_seg")),
                  fname_seg)
        sct.mv(os.path.join(folder_output, sct.add_suffix(os.path.basename(fname_data_propseg), "_centerline")),
                  fname_centerline)
        # if user was used, copy the labelled points to the output folder (they will then be scaled back)
        if use_viewer:
            fname_labels_viewer_new = os.path.join(folder_output, os.path.basename(sct.add_suffix(fname_data,
                                                                                                  "_labels_viewer")))
            sct.copy(fname_labels_viewer, fname_labels_viewer_new)
            # update variable (used later)
            fname_labels_viewer = fname_labels_viewer_new

    # check consistency of segmentation
    if arguments["-correct-seg"] == "1":
        check_and_correct_segmentation(fname_seg, fname_centerline, folder_output=folder_output, threshold_distance=3.0,
                                       remove_temp_files=remove_temp_files, verbose=verbose)

    # copy header from input to segmentation to make sure qform is the same
    sct.printv("Copy header input --> output(s) to make sure qform is the same.", verbose)
    list_fname = [fname_seg, fname_centerline]
    if use_viewer:
        list_fname.append(fname_labels_viewer)
    for fname in list_fname:
        im = Image(fname)
        im.header = image_input.header
        im.save(dtype='int8')  # they are all binary masks hence fine to save as int8

    return Image(fname_seg)


def main(arguments):
    fname_input_data = os.path.abspath(arguments["-i"])
    img_input = Image(fname_input_data)
    img_seg = propseg(img_input, arguments)
    fname_seg = img_seg.absolutepath
    path_qc = arguments.get("-qc", None)
    qc_dataset = arguments.get("-qc-dataset", None)
    qc_subject = arguments.get("-qc-subject", None)
    if path_qc is not None:
        generate_qc(fname_in1=fname_input_data, fname_seg=fname_seg, args=args, path_qc=os.path.abspath(path_qc),
                    dataset=qc_dataset, subject=qc_subject, process='sct_propseg')
    sct.display_viewer_syntax([fname_input_data, fname_seg], colormaps=['gray', 'red'], opacities=['', '1'])


if __name__ == "__main__":
    sct.init_sct()
    parser = get_parser()
    args = sys.argv[1:]
    arguments = parser.parse(args)
    res = main(arguments)
    raise SystemExit(res)
