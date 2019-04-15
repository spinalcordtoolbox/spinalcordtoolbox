#!/usr/bin/env python
# -*- coding: utf-8
# Functions dealing with centerline detection and manipulation

from __future__ import absolute_import, division

import os, datetime, logging

import numpy as np

import sct_utils as sct
from ..image import Image

logger = logging.getLogger(__name__)


def centerline2roi(fname_image, folder_output='./', verbose=0):
    """
    Tis method converts a binary centerline image to a .roi centerline file

    :param fname_image: filename of the binary centerline image, in RPI orientation
    :param folder_output: path to output folder where to copy .roi centerline
    :param verbose: adjusts the verbosity of the logging.
    :returns: filename of the .roi centerline that has been created
    """
    # TODO: change folder_output to fname_out
    path_data, file_data, ext_data = sct.extract_fname(fname_image)
    fname_output = file_data + '.roi'

    date_now = datetime.datetime.now()
    ROI_TEMPLATE = 'Begin Marker ROI\n' \
                   '  Build version="7.0_33"\n' \
                   '  Annotation=""\n' \
                   '  Colour=0\n' \
                   '  Image source="{fname_segmentation}"\n' \
                   '  Created  "{creation_date}" by Operator ID="SCT"\n' \
                   '  Slice={slice_num}\n' \
                   '  Begin Shape\n' \
                   '    X={position_x}; Y={position_y}\n' \
                   '  End Shape\n' \
                   'End Marker ROI\n'

    im = Image(fname_image)
    nx, ny, nz, nt, px, py, pz, pt = im.dim
    coordinates_centerline = im.getNonZeroCoordinates(sorting='z')

    f = open(fname_output, "w")
    sct.printv('\nWriting ROI file...', verbose)

    for coord in coordinates_centerline:
        coord_phys_center = im.transfo_pix2phys([[(nx - 1) / 2.0, (ny - 1) / 2.0, coord.z]])[0]
        coord_phys = im.transfo_pix2phys([[coord.x, coord.y, coord.z]])[0]
        f.write(ROI_TEMPLATE.format(fname_segmentation=fname_image,
                                    creation_date=date_now.strftime("%d %B %Y %H:%M:%S.%f %Z"),
                                    slice_num=coord.z + 1,
                                    position_x=coord_phys_center[0] - coord_phys[0],
                                    position_y=coord_phys_center[1] - coord_phys[1]))

    f.close()

    if os.path.abspath(folder_output) != os.getcwd():
        sct.copy(fname_output, folder_output)

    return fname_output


def detect_centerline(img, contrast, verbose=1):
    """Detect spinal cord centerline using OptiC.
    :param img: input Image() object.
    :param contrast: str: The type of contrast. Will define the path to Optic model.
    :returns: Image(): Output centerline
    """

    # Fetch path to Optic model based on contrast
    optic_models_path = os.path.join(sct.__sct_dir__, 'data', 'optic_models', '{}_model'.format(contrast))

    logger.debug('Detecting the spinal cord using OptiC')
    img_orientation = img.orientation

    temp_folder = sct.TempFolder()
    temp_folder.chdir()

    # convert image data type to int16, as required by opencv (backend in OptiC)
    img_int16 = img.copy()
    # Replace non-numeric values by zero
    img_data = img.data
    img_data[np.where(np.isnan(img_data))] = 0
    img_data[np.where(np.isinf(img_data))] = 0
    img_int16.data[np.where(np.isnan(img_int16.data))] = 0
    img_int16.data[np.where(np.isinf(img_int16.data))] = 0
    # rescale intensity
    min_out = np.iinfo('uint16').min
    max_out = np.iinfo('uint16').max
    min_in = np.nanmin(img_data)
    max_in = np.nanmax(img_data)
    data_rescaled = img_data.astype('float') * (max_out - min_out) / (max_in - min_in)
    img_int16.data = data_rescaled - (data_rescaled.min() - min_out)
    # change data type
    img_int16.change_type(np.uint16)
    # reorient the input image to RPI + convert to .nii
    img_int16.change_orientation('RPI')
    file_img = 'img_rpi_uint16'
    img_int16.save(file_img+'.nii')

    # call the OptiC method to generate the spinal cord centerline
    optic_input = file_img
    optic_filename = file_img + '_optic'
    os.environ["FSLOUTPUTTYPE"] = "NIFTI_PAIR"
    cmd_optic = [
     'isct_spine_detect',
     '-ctype=dpdt',
     '-lambda=1',
     optic_models_path,
     optic_input,
     optic_filename,
    ]
    # TODO: output coordinates, for each slice, in continuous (not discrete) values.

    sct.run(cmd_optic, is_sct_binary=True, verbose=0)

    # convert .img and .hdr files to .nii.gz
    img_ctl = Image(file_img + '_optic_ctr.hdr')
    img_ctl.change_orientation(img_orientation)

    # return to initial folder
    temp_folder.chdir_undo()
    if verbose < 2:
        logger.info("Remove temporary files...")
        temp_folder.cleanup()

    return img_ctl
