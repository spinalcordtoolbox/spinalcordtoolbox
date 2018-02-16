#!/usr/bin/env python
#########################################################################################
#
# Function to segment the spinal cord using deep convolutional networks
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener & Charley Gros
# Modified: 2018-01-22
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import numpy as np
import shutil
from scipy.ndimage.measurements import center_of_mass, label
from scipy.ndimage.morphology import binary_fill_holes
from skimage.exposure import rescale_intensity

import os
import sys
from spinalcordtoolbox.centerline import optic
import sct_utils as sct
from msct_image import Image
from msct_parser import Parser
from sct_image import set_orientation

import spinalcordtoolbox.resample.nipy_resample
from spinalcordtoolbox.deepseg_sc.cnn_models import nn_architecture


def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description("""Spinal Cord Segmentation using deep convolutional networks.""")

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
    parser.add_option(name='-qc',
                      type_value='folder_creation',
                      description='The path where the quality control generated content will be saved',
                      default_value=os.path.expanduser('~/qc_data'))
    parser.add_option(name='-noqc',
                      type_value=None,
                      description='Prevent the generation of the QC report',
                      mandatory=False)
    parser.add_option(name='-igt',
                      type_value='image_nifti',
                      description='File name of ground-truth segmentation.',
                      mandatory=False)
    return parser


def apply_intensity_normalization(img_path, fname_out):
    img = Image(img_path)
    img_normalized = img.copy()
    p2, p98 = np.percentile(img.data, (2, 98))
    img_normalized.data = rescale_intensity(img.data, in_range=(p2, p98), out_range=(0, 255))
    img_normalized.setFileName(fname_out)
    img_normalized.save()

    return img_normalized


def _find_crop_start_end(coord_ctr, crop_size, im_dim):
    half_size = crop_size // 2
    coord_start, coord_end = int(coord_ctr) - half_size + 1, int(coord_ctr) + half_size + 1

    if coord_end > im_dim:
        coord_end = im_dim
        coord_start = im_dim - crop_size if im_dim >= crop_size else 0
    if coord_start < 0:
        coord_start = 0
        coord_end = crop_size if im_dim >= crop_size else im_dim

    return coord_start, coord_end


def crop_image_around_centerline(im_in, ctr_in, im_out, crop_size, x_dim_half, y_dim_half):
    im_in, im_ctr = Image(im_in), Image(ctr_in)

    im_new = im_in.copy()
    im_new.dim = tuple([crop_size, crop_size, im_in.dim[2]] + list(im_in.dim[3:]))

    data_im_new = np.zeros((crop_size, crop_size, im_in.dim[2]))

    x_lst, y_lst = [], []
    for zz in range(im_in.dim[2]):
        x_ctr, y_ctr = center_of_mass(im_ctr.data[:, :, zz])

        x_start, x_end = _find_crop_start_end(x_ctr, crop_size, im_in.dim[0])
        y_start, y_end = _find_crop_start_end(y_ctr, crop_size, im_in.dim[1])

        crop_im = np.zeros((crop_size, crop_size))
        x_shape, y_shape = im_in.data[x_start:x_end, y_start:y_end, zz].shape
        crop_im[:x_shape, :y_shape] = im_in.data[x_start:x_end, y_start:y_end, zz]

        data_im_new[:, :, zz] = crop_im

        x_lst.append(str(x_start))
        y_lst.append(str(y_start))

    im_new.data = data_im_new

    im_new.setFileName(im_out)

    im_new.save()

    del im_in, im_ctr
    del im_new

    return x_lst, y_lst


def _remove_extrem_holes(z_lst, end_z, start_z=0):
    if start_z in z_lst:
        while start_z in z_lst:
            z_lst = z_lst[1:]
            start_z += 1
        if len(z_lst):
            z_lst.pop(0)

    if end_z in z_lst:
        while end_z in z_lst:
            z_lst = z_lst[:-1]
            end_z -= 1

    return z_lst


def _list2range(lst):
    tmplst = lst[:]
    tmplst.sort()
    start = tmplst[0]

    currentrange = [start, start + 1]

    for item in tmplst[1:]:
        if currentrange[1] == item:  # contiguous
            currentrange[1] += 1
        else:  # new range start
            yield list(currentrange)
            currentrange = [item, item + 1]

    yield list(currentrange)  # last range


def _fill_z_holes(zz_lst, data, z_spaccing):
    data_interpol = np.copy(data)
    for z_hole_start, z_hole_end in list(_list2range(zz_lst)):
        z_ref_start, z_ref_end = z_hole_start - 1, z_hole_end
        slice_ref_start, slice_ref_end = data[:, :, z_ref_start], data[:, :, z_ref_end]

        hole_cur_lst = range(z_hole_start, z_hole_end)
        lenght_hole = len(hole_cur_lst) + 1
        phys_lenght_hole = lenght_hole * z_spaccing

        denom_interpolation = (lenght_hole + 1)

        if phys_lenght_hole < 30:
            sct.log.warning('Filling an hole in the segmentation around z_slice #:' + str(z_ref_start))

            for idx_z, z_hole_cur in enumerate(hole_cur_lst):
                num_interpolation = (lenght_hole - idx_z - 1) * slice_ref_start  # Contribution of the bottom ref slice
                num_interpolation += (idx_z + 1) * slice_ref_end  # Contribution of the top ref slice

                slice_interpolation = num_interpolation * 1. / denom_interpolation
                slice_interpolation = (slice_interpolation > 0).astype(np.int)

                data_interpol[:, :, z_hole_cur] = slice_interpolation

    return data_interpol


def fill_z_holes(fname_in):
    im_in = Image(fname_in)
    data_in = im_in.data.astype(np.int)

    zz_zeros = [zz for zz in range(im_in.dim[2]) if 1 not in list(np.unique(data_in[:, :, zz]))]
    zz_holes = _remove_extrem_holes(zz_zeros, im_in.dim[2] - 1, 0)

    # filling z_holes, i.e. interpolate for z_slice not segmented
    im_in.data = _fill_z_holes(zz_holes, data_in, im_in.dim[6]) if len(zz_holes) else data_in

    im_in.setFileName(fname_in)
    im_in.save()
    del im_in


def main():
    sct.start_stream_logger()
    parser = get_parser()
    args = sys.argv[1:]
    arguments = parser.parse(args)

    fname_image = arguments['-i']
    contrast_type = arguments['-c']

    if "-ofolder" in arguments:
        output_folder = arguments["-ofolder"]
    else:
        output_folder = os.getcwd()

    if '-r' in arguments:
        remove_temp_files = arguments['-r']

    if '-v' in arguments:
        verbose = arguments['-v']

    if '-qc' in arguments:
        qc_path = arguments['-qc']

    if '-noqc' in arguments:
        qc_path = None

    deep_segmentation_spinalcord(fname_image, contrast_type, output_folder, qc_path=qc_path,
                                 remove_temp_files=remove_temp_files, verbose=verbose, args=args)


def deep_segmentation_spinalcord(fname_image, contrast_type, output_folder, qc_path=None, remove_temp_files=1, verbose=1, args=None):
    # initalizing parameters
    crop_size = 64  # TODO: this parameter should actually be passed by the model, as it is one of its fixed parameter

    # loading models required to perform the segmentation
    # this step can be long, as some models (e.g., DL) are heavy
    sct.log.info("Loading models...")
    path_script = os.path.dirname(__file__)
    path_sct = os.path.dirname(path_script)
    optic_models_fname = os.path.join(path_sct, 'data/optic_models', '{}_model'.format(contrast_type))

    segmentation_model_fname = os.path.join(path_sct, 'data/deepseg_sc_models', '{}_seg_sc.h5'.format(contrast_type))
    seg_model = nn_architecture(height=crop_size, width=crop_size, depth=2 if contrast_type != 't2' else 3)
    seg_model.load_weights(segmentation_model_fname)

    # create temporary folder with intermediate results
    sct.log.info("Creating temporary folder...")
    file_fname = os.path.basename(fname_image)
    tmp_folder = sct.TempFolder()
    tmp_folder_path = tmp_folder.get_path()
    fname_image_tmp = tmp_folder.copy_from(fname_image)
    tmp_folder.chdir()
    print tmp_folder_path

    # orientation of the image, should be RPI
    sct.log.info("Reorient the image to RPI, if necessary...")
    fname_orient = sct.add_suffix(file_fname, '_RPI')
    im_2orient = Image(file_fname)
    original_orientation = im_2orient.orientation
    if original_orientation != 'RPI':
        im_orient = set_orientation(im_2orient, 'RPI')
        im_orient.setFileName(fname_orient)
        im_orient.save()
    else:
        im_orient = im_2orient
        shutil.copyfile(fname_image_tmp, fname_orient)

    # resampling RPI image
    sct.log.info("Resample the image to 0.5 mm isotropic resolution...")
    fname_res = sct.add_suffix(fname_orient, '_resampled')
    im_2res = im_orient
    input_resolution = im_2res.dim[4:7]
    new_resolution = 'x'.join(['0.5', '0.5', str(input_resolution[2])])
    spinalcordtoolbox.resample.nipy_resample.resample_file(fname_orient, fname_res, new_resolution,
                                                           'mm', 'linear', verbose=0)

    # find the spinal cord centerline - execute OptiC binary
    sct.log.info("Finding the spinal cord centerline...")
    _, centerline_filename = optic.detect_centerline(image_fname=fname_res,
                                                     contrast_type=contrast_type,
                                                     optic_models_path=optic_models_fname,
                                                     folder_output=tmp_folder_path,
                                                     remove_temp_files=remove_temp_files,
                                                     output_roi=False,
                                                     verbose=0)

    # crop image around the spinal cord centerline
    sct.log.info("Cropping the image around the spinal cord...")
    fname_crop = sct.add_suffix(fname_res, '_crop')
    X_CROP_LST, Y_CROP_LST = crop_image_around_centerline(im_in=fname_res, ctr_in=centerline_filename, im_out=fname_crop,
                                                          crop_size=crop_size,
                                                          x_dim_half=crop_size // 2, y_dim_half=crop_size // 2)

    # normalize the intensity of the images
    sct.log.info("Normalizing the intensity...")
    fname_norm = sct.add_suffix(fname_crop, '_norm')
    image_normalized = apply_intensity_normalization(img_path=fname_crop, fname_out=fname_norm)

    # segment the spinal cord
    sct.log.info("Segmenting the spinal cord using deep learning on 2D images...")
    fname_seg_crop = sct.add_suffix(fname_norm, '_seg')
    seg_crop = image_normalized.copy()
    seg_crop.data *= 0.0
    for zz in list(reversed(range(image_normalized.dim[2]))):
        pred_seg = seg_model.predict(np.expand_dims(np.expand_dims(image_normalized.data[:, :, zz], -1), 0))[0, :, :, 0]
        pred_seg_th = (pred_seg > 0.0001).astype(int)

        if contrast_type in ['t2', 't1']:
            labeled_obj, num_obj = label(pred_seg_th)
            if num_obj > 1:
                pred_seg_th = (labeled_obj == (np.bincount(labeled_obj.flat)[1:].argmax() + 1))

            pred_seg_th = binary_fill_holes(pred_seg_th, structure=np.ones((3, 3))).astype(np.int)

        seg_crop.data[:, :, zz] = pred_seg_th
    seg_crop.setFileName(fname_seg_crop)
    seg_crop.save()

    sct.log.info("Reassembling the image...")
    fname_seg_res_RPI = sct.add_suffix(file_fname, '_res_RPI_seg')
    im = Image(fname_res)
    seg_unCrop = im.copy()
    seg_unCrop.data *= 0

    for zz in range(seg_unCrop.dim[2]):
        pred_seg = seg_crop.data[:, :, zz]
        x_start, y_start = int(X_CROP_LST[zz]), int(Y_CROP_LST[zz])
        x_end = x_start + crop_size if x_start + crop_size < seg_unCrop.dim[0] else seg_unCrop.dim[0]
        y_end = y_start + crop_size if y_start + crop_size < seg_unCrop.dim[1] else seg_unCrop.dim[1]
        seg_unCrop.data[x_start:x_end, y_start:y_end, zz] = pred_seg[0:x_end - x_start, 0:y_end - y_start]

    seg_unCrop.setFileName(fname_seg_res_RPI)
    seg_unCrop.save()

    # resample to initial resolution
    sct.log.info("Resampling the segmentation to the original image resolution...")
    fname_seg_RPI = sct.add_suffix(file_fname, '_RPI_seg')
    initial_resolution = 'x'.join([str(input_resolution[0]), str(input_resolution[1]), str(input_resolution[2])])
    spinalcordtoolbox.resample.nipy_resample.resample_file(fname_seg_res_RPI, fname_seg_RPI, initial_resolution,
                                                           'mm', 'linear', verbose=0)

    # binarize the resampled image to remove interpolation effects
    sct.log.info("Binarizing the segmentation to avoid interpolation effects...")
    sct.run('sct_maths -i ' + fname_seg_RPI + ' -bin 0.75 -o ' + fname_seg_RPI, verbose=0)

    # post processing step to z_regularized
    fill_z_holes(fname_in=fname_seg_RPI)

    # reorient to initial orientation
    sct.log.info("Reorienting the segmentation to the original image orientation...")
    fname_seg = sct.add_suffix(file_fname, '_seg')
    if original_orientation != 'RPI':
        im_orient = set_orientation(Image(fname_seg_RPI), original_orientation)
        im_orient.setFileName(fname_seg)
        im_orient.save()
    else:
        shutil.copyfile(fname_seg_RPI, fname_seg)

    tmp_folder.chdir_undo()

    # copy image from temporary folder into output folder
    shutil.copyfile(tmp_folder_path + '/' + fname_seg, output_folder + '/' + fname_seg)

    # remove temporary files
    if remove_temp_files:
        sct.log.info("Remove temporary files...")
        tmp_folder.cleanup()

    if qc_path is not None:
        import spinalcordtoolbox.reports.qc as qc
        import spinalcordtoolbox.reports.slice as qcslice

        param = qc.Params(fname_image, 'sct_propseg', args, 'Axial', qc_path)
        report = qc.QcReport(param, '')

        @qc.QcImage(report, 'none', [qc.QcImage.listed_seg, ])
        def test(qslice):
            return qslice.mosaic()

        try:
            test(qcslice.Axial(Image(fname_image), Image(fname_seg)))
            sct.log.info('Sucessfully generated the QC results in %s' % param.qc_results)
            sct.log.info('Use the following command to see the results in a browser:')
            sct.log.info('sct_qc -folder %s' % qc_path)
        except:
            sct.log.warning('Issue when creating QC report.')

    sct.display_viewer_syntax([fname_image, output_folder + '/' + fname_seg], colormaps=['gray', 'red'], opacities=['', '0.7'])


if __name__ == "__main__":
    main()
