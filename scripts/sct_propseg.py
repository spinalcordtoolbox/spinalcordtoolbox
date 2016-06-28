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

from msct_parser import Parser
import sys
import sct_utils as sct
import os


def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('''This program segments automatically the spinal cord on T1- and T2-weighted images, for any field of view. You must provide the type of contrast, the image as well as the output folder path.
Initialization is provided by a spinal cord detection module based on the elliptical Hough transform on multiple axial slices. The result of the detection is available as a PNG image using option -detection-display.
Parameters of the spinal cord detection are :
 - the position (in inferior-superior direction) of the initialization
 - the number of axial slices
 - the gap (in pixel) between two axial slices
 - the approximate radius of the spinal cord

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
                      example=['t1','t2'])
    parser.add_option(name="-c",
                      type_value="multiple_choice",
                      description="type of image contrast, t2: cord dark / CSF bright ; t1: cord bright / CSF dark",
                      mandatory=True,
                      example=['t1','t2'])
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
    parser.add_option(name="-detect-nii",
                      type_value=None,
                      description="output: spinal cord detection as a nifti image",
                      mandatory=False)
    parser.add_option(name="-detect-png",
                      type_value=None,
                      description="output: spinal cord detection as a PNG image",
                      mandatory=False)

    parser.usage.addSection("\nOptions helping the segmentation")
    parser.add_option(name="-init-centerline",
                      type_value="image_nifti",
                      description="filename of centerline to use for the propagation, format .txt or .nii, see file structure in documentation.\nReplace filename by 'viewer' to use interactive viewer for providing centerline. Ex: -init-centerline viewer",
                      mandatory=False)
    parser.add_option(name="-init",
                      type_value="float",
                      description="axial slice where the propagation starts, default is middle axial slice",
                      mandatory=False)
    parser.add_option(name="-init-mask",
                      type_value="image_nifti",
                      description="mask containing three center of the spinal cord, used to initiate the propagation.\nReplace filename by 'viewer' to use interactive viewer for providing mask. Ex: -init-mask viewer",
                      mandatory=False)
    parser.add_option(name="-mask-correction",
                      type_value="image_nifti",
                      description="mask containing binary pixels at edges of the spinal cord on which the segmentation algorithm will be forced to register the surface. Can be used in case of poor/missing contrast between spinal cord and CSF or in the presence of artefacts/pathologies.",
                      mandatory=False)
    parser.add_option(name="-radius",
                      type_value="float",
                      description="approximate radius of the spinal cord, default is 4 mm",
                      mandatory=False)
    parser.add_option(name="-detect-n",
                      type_value="int",
                      description="number of axial slices computed in the detection process, default is 4",
                      mandatory=False)
    parser.add_option(name="-detect-gap",
                      type_value="int",
                      description="gap in Z direction for the detection process, default is 4",
                      mandatory=False)
    parser.add_option(name="-init-validation",
                      type_value=None,
                      description="enable validation on spinal cord detection based on discriminant analysis",
                      mandatory=False)
    parser.add_option(name="-nbiter",
                      type_value="int",
                      description="stop condition: number of iteration for the propagation for both direction, default is 200",
                      mandatory=False)
    parser.add_option(name="-max-area",
                      type_value="float",
                      description="[mm^2], stop condition: maximum cross-sectional area, default is 120 mm^2",
                      mandatory=False)
    parser.add_option(name="-max-deformation",
                      type_value="float",
                      description="[mm], stop condition: maximum deformation per iteration, default is 2.5 mm",
                      mandatory=False)
    parser.add_option(name="-min-contrast",
                      type_value="float",
                      description="[intensity value], stop condition: minimum local SC/CSF contrast, default is 50",
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
    return parser

if __name__ == "__main__":
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])

    input_filename = arguments["-i"]
    contrast_type = arguments["-c"]

    # Building the command
    cmd = "isct_propseg" + " -i " + input_filename + " -t " + contrast_type

    if "-ofolder" in arguments:
        folder_output = sct.slash_at_the_end(arguments["-ofolder"], slash=1)
    else:
        folder_output = './'
    cmd += " -o " + folder_output
    if not os.path.exists(folder_output):
        os.makedirs(folder_output)

    if "-down" in arguments:
        cmd += " -down " + str(arguments["-down"])
    if "-up" in arguments:
        cmd += " -up " + str(arguments["-up"])

    verbose = 0
    if "-v" in arguments:
        if arguments["-v"] is "1":
            verbose = 2
            cmd += " -verbose"

    # Output options
    if "-mesh" in arguments:
        cmd += " -mesh"
    if "-centerline-binary" in arguments:
        cmd += " -centerline-binary"
    if "-CSF" in arguments:
        cmd += " -CSF"
    if "-centerline-coord" in arguments:
        cmd += " -centerline-coord"
    if "-cross" in arguments:
        cmd += " -cross"
    if "-init-tube" in arguments:
        cmd += " -init-tube"
    if "-low-resolution-mesh" in arguments:
        cmd += " -low-resolution-mesh"
    if "-detect-nii" in arguments:
        cmd += " -detect-nii"
    if "-detect-png" in arguments:
        cmd += " -detect-png"

    # Helping options
    use_viewer = None
    if "-init-centerline" in arguments:
        if str(arguments["-init-centerline"]) == "viewer":
            use_viewer = "centerline"
        else:
            cmd += " -init-centerline " + str(arguments["-init-centerline"])
    if "-init" in arguments:
        cmd += " -init " + str(arguments["-init"])
    if "-init-mask" in arguments:
        if str(arguments["-init-mask"]) == "viewer":
            use_viewer = "mask"
        else:
            cmd += " -init-mask " + str(arguments["-init-mask"])
    if "-mask-correction" in arguments:
        cmd += " -mask-correction " + str(arguments["-mask-correction"])
    if "-radius" in arguments:
        cmd += " -radius " + str(arguments["-radius"])
    if "-detect-n" in arguments:
        cmd += " -detect-n " + str(arguments["-detect-n"])
    if "-detect-gap" in arguments:
        cmd += " -detect-gap " + str(arguments["-detect-gap"])
    if "-init-validation" in arguments:
        cmd += " -init-validation"
    if "-nbiter" in arguments:
        cmd += " -nbiter " + str(arguments["-nbiter"])
    if "-max-area" in arguments:
        cmd += " -max-area " + str(arguments["-max-area"])
    if "-max-deformation" in arguments:
        cmd += " -max-deformation " + str(arguments["-max-deformation"])
    if "-min-contrast" in arguments:
        cmd += " -min-contrast " + str(arguments["-min-contrast"])
    if "-d" in arguments:
        cmd += " -d " + str(arguments["-d"])
    if "-distance-search" in arguments:
        cmd += " -dsearch " + str(arguments["-distance-search"])
    if "-alpha" in arguments:
        cmd += " -alpha " + str(arguments["-alpha"])

    # check if input image is in 3D. Otherwise itk image reader will cut the 4D image in 3D volumes and only take the first one.
    from msct_image import Image
    image_input = Image(input_filename)
    nx, ny, nz, nt, px, py, pz, pt = image_input.dim
    if nt > 1:
        sct.printv('ERROR: your input image needs to be 3D in order to be segmented.', 1, 'error')

    # if centerline or mask is asked using viewer
    if use_viewer:
        # make sure image is in SAL orientation, as it is the orientation used by PropSeg
        from sct_image import orientation
        image_input_orientation = orientation(image_input, get=True, verbose=False)
        path_fname, file_fname, ext_fname = sct.extract_fname(input_filename)
        reoriented_image_filename = 'tmp.' + sct.add_suffix(file_fname + ext_fname, "_SAL")
        sct.run('sct_image -i ' + input_filename + ' -o ' + folder_output + reoriented_image_filename + ' -setorient SAL -v 0', verbose=False)

        from sct_viewer import ClickViewer
        image_input_reoriented = Image(folder_output + reoriented_image_filename)
        viewer = ClickViewer(image_input_reoriented)
        if use_viewer == "mask":
            viewer.number_of_slices = 3
            viewer.gap_inter_slice = 10

        # start the viewer that ask the user to enter a few points along the spinal cord
        mask_points = viewer.start()
        if mask_points:
            # create the mask containing either the three-points or centerline mask for initialization
            mask_filename = sct.add_suffix(reoriented_image_filename, "_mask_viewer")
            sct.run("sct_label_utils -i " + folder_output + reoriented_image_filename + " -create " + mask_points + " -o " + folder_output + mask_filename, verbose=False)

            # reorient the initialization mask to correspond to input image orientation
            mask_reoriented_filename = sct.add_suffix(file_fname + ext_fname, "_mask_viewer")
            sct.run('sct_image -i ' + folder_output + mask_filename + ' -o ' + folder_output + mask_reoriented_filename + ' -setorient ' + image_input_orientation + ' -v 0', verbose=False)

            # remove temporary files
            sct.run('rm -rf ' + folder_output + 'tmp.*')

            # add mask filename to parameters string
            if use_viewer == "centerline":
                cmd += " -init-centerline " + folder_output + mask_reoriented_filename
            elif use_viewer == "mask":
                cmd += " -init-mask " + folder_output + mask_reoriented_filename
        else:
            sct.printv('\nERROR: the viewer has been closed before entering all manual points. Please try again.', verbose, type='error')

    sct.run(cmd, verbose)

    sct.printv('\nDone! To view results, type:', verbose)
    # extracting output filename
    path_fname, file_fname, ext_fname = sct.extract_fname(input_filename)
    output_filename = file_fname+"_seg"+ext_fname

    if folder_output == "./":
        output_name = output_filename
    else:
        output_name = folder_output + output_filename
    sct.printv("fslview "+input_filename+" "+output_name+" -l Red -b 0,1 -t 0.7 &\n", verbose, 'info')