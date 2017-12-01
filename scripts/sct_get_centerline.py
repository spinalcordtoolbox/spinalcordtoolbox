#!/usr/bin/env python

import os
import sys
import shutil

import sct_utils as sct
from msct_parser import Parser
from spinalcordtoolbox.centerline import optic
from msct_image import Image
from sct_image import orientation
from msct_types import Centerline
from sct_viewer import ClickViewerPropseg
from sct_straighten_spinalcord import smooth_centerline


def viewer_centerline(image_fname, interslice_gap, verbose):
    image_input_reoriented = Image(image_fname)
    nx, ny, nz, nt, px, py, pz, pt = image_input_reoriented.dim
    viewer = ClickViewerPropseg(image_input_reoriented)

    viewer.gap_inter_slice = int(interslice_gap / px)  # px because the image is supposed to be SAL orientation
    viewer.number_of_slices = 0
    viewer.calculate_list_slices()

    # start the viewer that ask the user to enter a few points along the spinal cord
    mask_points = viewer.start()

    if not mask_points and viewer.closed:
        mask_points = viewer.list_points_useful_notation

    if mask_points:
        # create the mask containing either the three-points or centerline mask for initialization
        mask_filename = sct.add_suffix(image_fname, "_mask_viewer")
        sct.run("sct_label_utils -i " + image_fname + " -create " + mask_points + " -o " + mask_filename, verbose=False)

        fname_output = mask_filename

    else:
        sct.printv('\nERROR: the viewer has been closed before entering all manual points. Please try again.', 1, type='error')
        fname_output = None

    return fname_output

def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description("""This function allows the extraction of the spinal cord centerline. Two methods are available: OptiC (automatic) and Viewer (manual).""")

    parser.add_option(name="-i",
                      type_value="image_nifti",
                      description="input image.",
                      mandatory=True,
                      example="t1.nii.gz")
    parser.add_option(name="-c",
                      type_value="multiple_choice",
                      description="type of image contrast.",
                      mandatory=False,
                      example=['t1', 't2', 't2s', 'dwi'])
    parser.add_option(name="-ofolder",
                      type_value="folder_creation",
                      description="output folder.",
                      mandatory=False,
                      example="My_Output_Folder/",
                      default_value="")
    parser.add_option(name="-roi",
                      type_value="multiple_choice",
                      description="outputs a ROI file, compatible with JIM software.",
                      mandatory=False,
                      example=['0', '1'],
                      default_value='0')
    parser.add_option(name="-method",
                      type_value="multiple_choice",
                      description="Method used for extracting the centerline.\n"
                                  "optic: automatic spinal cord detection method\n"
                                  "viewer: manually selected a few points, approximation with NURBS",
                      mandatory=False,
                      example=['optic', 'viewer'],
                      default_value='optic')
    parser.add_option(name="-gap",
                      type_value="float",
                      description="Gap in mm between manually selected points when using the Viewer method.",
                      mandatory=False,
                      default_value='10.0')
    parser.add_option(name="-igt",
                      type_value="image_nifti",
                      description="File name of ground-truth centerline or segmentation (binary nifti).",
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
    return parser

def run_main():
    sct.start_stream_logger()
    parser = get_parser()
    args = sys.argv[1:]
    arguments = parser.parse(args)

    # Input filename
    fname_input_data = arguments["-i"]
    fname_data = os.path.abspath(fname_input_data)

    # Method used
    method = 'optic'
    if "-method" in arguments:
        method = arguments["-method"]

    # Contrast type
    contrast_type = ''
    if "-c" in arguments:
        contrast_type = arguments["-c"]
    if method == 'optic' and not contrast_type:
        # Contrast must be
        error = 'ERROR: -c is a mandatory argument when using Optic method.'
        sct.printv(error, type='error')
        return

    # Ga between slices
    interslice_gap = 10.0
    if "-gap" in arguments:
        interslice_gap = float(arguments["-gap"])

    # Output folder
    if "-ofolder" in arguments:
        folder_output = arguments["-ofolder"]
    else:
        folder_output = '.'

    # Remove temporary files
    remove_temp_files = True
    if "-r" in arguments:
        remove_temp_files = bool(int(arguments["-r"]))

    # Outputs a ROI file
    output_roi = False
    if "-roi" in arguments:
        output_roi = bool(int(arguments["-roi"]))

    # Verbosity
    verbose = 0
    if "-v" in arguments:
        verbose = int(arguments["-v"])

    if method == 'viewer':
        path_data, file_data, ext_data = sct.extract_fname(fname_data)

        # create temporary folder
        temp_folder = sct.TempFolder()
        temp_folder.copy_from(fname_data)
        temp_folder.chdir()

        # make sure image is in SAL orientation, as it is the orientation used by the viewer
        image_input = Image(fname_data)
        image_input_orientation = orientation(image_input, get=True, verbose=False)
        reoriented_image_filename = sct.add_suffix(file_data + ext_data, "_SAL")
        cmd_image = 'sct_image -i "%s" -o "%s" -setorient SAL -v 0' % (fname_data, reoriented_image_filename)
        sct.run(cmd_image, verbose=False)

        # extract points manually using the viewer
        fname_points = viewer_centerline(image_fname=reoriented_image_filename, interslice_gap=interslice_gap, verbose=verbose)

        if fname_points is not None:
            image_points_RPI = sct.add_suffix(fname_points, "_RPI")
            cmd_image = 'sct_image -i "%s" -o "%s" -setorient RPI -v 0' % (fname_points, image_points_RPI)
            sct.run(cmd_image, verbose=False)

            image_input_reoriented = Image(image_points_RPI)

            # fit centerline, smooth it and return the first derivative (in physical space)
            x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline(image_points_RPI, algo_fitting='nurbs', nurbs_pts_number=3000, phys_coordinates=True, verbose=verbose, all_slices=False)
            centerline = Centerline(x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv)

            # average centerline coordinates over slices of the image
            x_centerline_fit_rescorr, y_centerline_fit_rescorr, z_centerline_rescorr, x_centerline_deriv_rescorr, y_centerline_deriv_rescorr, z_centerline_deriv_rescorr = centerline.average_coordinates_over_slices(image_input_reoriented)

            # compute z_centerline in image coordinates for usage in vertebrae mapping
            voxel_coordinates = image_input_reoriented.transfo_phys2pix([[x_centerline_fit_rescorr[i], y_centerline_fit_rescorr[i], z_centerline_rescorr[i]] for i in range(len(z_centerline_rescorr))])
            x_centerline_voxel = [coord[0] for coord in voxel_coordinates]
            y_centerline_voxel = [coord[1] for coord in voxel_coordinates]
            z_centerline_voxel = [coord[2] for coord in voxel_coordinates]

            # compute z_centerline in image coordinates with continuous precision
            voxel_coordinates = image_input_reoriented.transfo_phys2continuouspix([[x_centerline_fit_rescorr[i], y_centerline_fit_rescorr[i], z_centerline_rescorr[i]] for i in range(len(z_centerline_rescorr))])
            x_centerline_voxel_cont = [coord[0] for coord in voxel_coordinates]
            y_centerline_voxel_cont = [coord[1] for coord in voxel_coordinates]
            z_centerline_voxel_cont = [coord[2] for coord in voxel_coordinates]

            # Create an image with the centerline
            image_input_reoriented.data *= 0
            min_z_index, max_z_index = int(round(min(z_centerline_voxel))), int(round(max(z_centerline_voxel)))
            for iz in range(min_z_index, max_z_index + 1):
                image_input_reoriented.data[int(round(x_centerline_voxel[iz - min_z_index])), int(round(y_centerline_voxel[iz - min_z_index])), int(iz)] = 1  # if index is out of bounds here for hanning: either the segmentation has holes or labels have been added to the file

            # Write the centerline image
            sct.printv('\nWrite NIFTI volumes...', verbose)
            fname_centerline_oriented = file_data + '_centerline' + ext_data
            image_input_reoriented.setFileName(fname_centerline_oriented)
            image_input_reoriented.changeType('uint8')
            image_input_reoriented.save()

            sct.printv('\nSet to original orientation...', verbose)
            sct.run('sct_image -i ' + fname_centerline_oriented + ' -setorient ' + image_input_orientation + ' -o ' + fname_centerline_oriented)

            # create a txt file with the centerline
            fname_centerline_oriented_txt = file_data + '_centerline.txt'
            file_results = open(fname_centerline_oriented_txt, 'w')
            for i in range(min_z_index, max_z_index + 1):
                file_results.write(str(int(i)) + ' ' + str(round(x_centerline_voxel_cont[i - min_z_index], 2)) + ' ' + str(round(y_centerline_voxel_cont[i - min_z_index], 2)) + '\n')
            file_results.close()

            fname_centerline_oriented_roi = optic.centerline2roi(fname_image=fname_centerline_oriented,
                                                                     folder_output='./',
                                                                     verbose=verbose)

            # return to initial folder
            temp_folder.chdir_undo()

            # copy result to output folder
            sct.copy(os.path.join(temp_folder.get_path(), fname_centerline_oriented), folder_output)
            sct.copy(os.path.join(temp_folder.get_path(), fname_centerline_oriented_txt), folder_output)
            if output_roi:
                sct.copy(os.path.join(temp_folder.get_path(), fname_centerline_oriented_roi), folder_output)
            centerline_filename = os.path.join(folder_output, fname_centerline_oriented)



        else:
            centerline_filename = 'error'

        # delete temporary folder
        if remove_temp_files:
            temp_folder.cleanup()

    else:
        # condition on verbose when using OptiC
        if verbose == 1:
            verbose = 2

        # OptiC models
        path_script = os.path.dirname(__file__)
        path_sct = os.path.dirname(path_script)
        optic_models_path = os.path.join(path_sct, 'data/optic_models', '{}_model'.format(contrast_type))

        # Execute OptiC binary
        _, centerline_filename = optic.detect_centerline(image_fname=fname_data,
                                                    contrast_type=contrast_type,
                                                    optic_models_path=optic_models_path,
                                                    folder_output=folder_output,
                                                    remove_temp_files=remove_temp_files,
                                                    output_roi=output_roi,
                                                    verbose=verbose)

    sct.display_viewer_syntax([fname_input_data, centerline_filename], colormaps=['gray', 'red'], opacities=['', '1'])

if __name__ == '__main__':
    run_main()
