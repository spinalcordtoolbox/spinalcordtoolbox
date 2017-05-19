import os
import shutil

import nibabel as nib

import sct_utils as sct
from sct_image import orientation
from msct_image import Image


def detect_centerline(image_fname, init_option, contrast_type,
                        optic_models_path, folder_output,
                        remove_temp_files=False, verbose=0):
    """This method will use the OptiC to detect the centerline.

    :param image_fname: The input image filename.
    :param init_option: Axial slice where the propagation starts.
    :param contrast_type: The contrast type.
    :param optic_models_path: The path with the Optic model files.
    :param folder_output: The OptiC output folder.
    :param remove_temp_files: Remove the temporary created files.
    :param verbose: Adjusts the verbosity of the logging.

    :returns: The OptiC output filename.
    """

    image_input = Image(image_fname)
    path_data, file_data, ext_data = sct.extract_fname(image_fname)

    sct.printv('Detecting the spinal cord using OptiC', verbose=verbose)
    image_input_orientation = orientation(image_input, get=True,
                                          verbose=False)

    temp_folder = sct.TempFolder()
    temp_folder.copy_from(image_fname)
    temp_folder.chdir()

    # convert image data type to int16, as required by opencv (backend in OptiC)
    image_int_filename = sct.add_suffix(file_data + ext_data, "_int16")
    cmd_type = 'sct_image -i "%s" -o "%s" -type int16 -v 0' % \
               (file_data + ext_data, image_int_filename)
    sct.run(cmd_type, verbose=0)

    # reorient the input image to RPI + convert to .nii
    reoriented_image_filename = sct.add_suffix(image_int_filename, "_RPI")
    img_filename = ''.join(sct.extract_fname(reoriented_image_filename)[:2])
    reoriented_image_filename_nii = img_filename + '.nii'
    cmd_reorient = 'sct_image -i "%s" -o "%s" -setorient RPI -v 0' % \
                (image_int_filename, reoriented_image_filename_nii)
    sct.run(cmd_reorient, verbose=0)

    image_rpi_init = Image(reoriented_image_filename_nii)
    nxr, nyr, nzr, ntr, pxr, pyr, pzr, ptr = image_rpi_init.dim
    if init_option is not None:
        if init_option > 1:
            init_option /= (nzr - 1)
        cmd += " -init " + str(init_option)

    # call the OptiC method to generate the spinal cord centerline
    optic_input = img_filename
    optic_filename = img_filename + '_optic'

    os.environ["FSLOUTPUTTYPE"] = "NIFTI_PAIR"
    cmd_optic = 'isct_spine_detect -ctype=dpdt -lambda=1 "%s" "%s" "%s"' % \
                (optic_models_path, optic_input, optic_filename)
    sct.run(cmd_optic, verbose=0)

    # convert .img and .hdr files to .nii.gz
    optic_hdr_filename = img_filename + '_optic_ctr.hdr'
    centerline_optic_RPI_filename = sct.add_suffix(file_data + ext_data,
                                                   "_centerline_optic_RPI")
    img = nib.load(optic_hdr_filename)
    nib.save(img, centerline_optic_RPI_filename)

    # reorient the output image to initial orientation
    centerline_optic_filename = sct.add_suffix(file_data + ext_data,
                                               "_centerline_optic")
    cmd_reorient = 'sct_image -i "%s" -o "%s" -setorient "%s" -v 0' % \
                   (centerline_optic_RPI_filename,
                    centerline_optic_filename,
                    image_input_orientation)
    sct.run(cmd_reorient, verbose=0)

    # copy centerline to parent folder
    sct.printv('Copy output to ' + folder_output, verbose=0)
    if os.path.isabs(folder_output):
        shutil.copy(centerline_optic_filename, folder_output)
    else:
        dst_path = os.path.join("../", folder_output)
        shutil.copy(centerline_optic_filename,
                    '../' + folder_output)

    # return to initial folder
    os.chdir('..')

    # delete temporary folder
    if remove_temp_files:
        temp_folder.cleanup()

    return os.path.join(folder_output,
                        centerline_optic_filename)
