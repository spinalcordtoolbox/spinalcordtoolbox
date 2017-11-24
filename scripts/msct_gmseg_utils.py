#!/usr/bin/env python
########################################################################################################################
#
#
# Utility functions used for the segmentation of the gray matter
#
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Sara Dupont
# Created: 2016-06-14
#
# About the license: see the file LICENSE.TXT
########################################################################################################################
import sys, io, os, math, time, random, shutil

import numpy as np

from msct_image import Image
from sct_image import set_orientation
from sct_utils import extract_fname, printv, add_suffix
import sct_utils as sct
from sct_crop_image import ImageCropper
import sct_create_mask
import sct_register_multimodal, sct_apply_transfo

########################################################################################################################
#                                   CLASS SLICE
########################################################################################################################
class Slice:
    """
    Slice instance used in the model dictionary for the segmentation of the gray matter
    """

    def __init__(self, slice_id=None, im=None, gm_seg=None, wm_seg=None, im_m=None, gm_seg_m=None, wm_seg_m=None, level=0):
        """
        Slice constructor
        :param slice_id: slice ID number, type: int
        :param im: original image (a T2star 2D image croped around the spinal cord), type: numpy array
        :param gm_seg: manual gray matter segmentation of the original image, type: list of numpy array
        :param wm_seg: manual white matter segmentation of the original image, type: list of numpy array
        :param reg_to_m: name of the file containing the transformation for this slice to go from the image original space to the model space, type: string
        :param im_m: image in the model space, type: numpy array
        :param gm_seg_m: manual gray matter segmentation in the model space, type: numpy array
        :param wm_seg_m: manual white matter segmentation in the model space, type: numpy array
        :param level: vertebral level of the slice, type: int
        """
        self.id = slice_id
        self.im = im
        self.gm_seg = gm_seg
        self.wm_seg = wm_seg
        self.im_M = im_m
        self.gm_seg_M = gm_seg_m
        self.wm_seg_M = wm_seg_m
        self.level = level

    def set(self, slice_id=None, im=None, gm_seg=None, wm_seg=None, im_m=None, gm_seg_m=None, wm_seg_m=None, level=None):
        """
        Slice setter, only the specified parameters are set
        :param slice_id: slice ID number, type: int
        :param im: original image (a T2star 2D image croped around the spinal cord), type: numpy array
        :param gm_seg: manual gray matter segmentation of the original image, type: list of numpy array
        :param wm_seg: manual white matter segmentation of the original image, type: list of numpy array
        :param reg_to_m: name of the file containing the transformation for this slice to go from the image original space to the model space, type: string
        :param im_m: image in the model space, type: numpy array
        :param gm_seg_m: manual gray matter segmentation in the model space, type: numpy array
        :param wm_seg_m: manual white matter segmentation in the model space, type: numpy array
        :param level: vertebral level of the slice, type: int
        """
        if slice_id is not None:
            self.id = slice_id
        if im is not None:
            self.im = np.asarray(im)
        if gm_seg is not None:
            self.gm_seg = np.asarray(gm_seg)
        if wm_seg is not None:
            self.wm_seg = np.asarray(wm_seg)
        if im_m is not None:
            self.im_M = np.asarray(im_m)
        if gm_seg_m is not None:
            self.gm_seg_M = np.asarray(gm_seg_m)
        if wm_seg_m is not None:
            self.wm_seg_M = np.asarray(wm_seg_m)
        if level is not None:
            self.level = level

    def __repr__(self):
        s = '\nSlice #' + str(self.id)
        if self.level is not None:
            s += 'Level : ' + str(self.level)
        s += '\nImage : \n' + str(self.im)
        s += '\nGray matter segmentation : \n' + str(self.gm_seg)
        if self.im_M is not None:
            s += '\nImage in the common model space: \n' + str(self.im_M)
        if self.gm_seg_M is not None:
            s += '\nGray matter segmentation in the common model space: \n' + str(self.gm_seg_M)
        return s


########################################################################################################################
#                               FUNCTIONS USED FOR PRE-PROCESSING
########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
def pre_processing(fname_target, fname_sc_seg, fname_level=None, fname_manual_gmseg=None, new_res=0.3, square_size_size_mm=22.5, denoising=True, verbose=1, rm_tmp=True, for_model=False):
    printv('\nPre-process data...', verbose, 'normal')

    tmp_dir = sct.tmp_create()

    sct.copy(fname_target, tmp_dir)
    fname_target = ''.join(extract_fname(fname_target)[1:])
    sct.copy(fname_sc_seg, tmp_dir)
    fname_sc_seg = ''.join(extract_fname(fname_sc_seg)[1:])

    curdir = os.getcwd()
    os.chdir(tmp_dir)

    original_info = {'orientation': None, 'im_sc_seg_rpi': None, 'interpolated_images': []}

    im_target = Image(fname_target).copy()
    im_sc_seg = Image(fname_sc_seg).copy()

    # get original orientation
    printv('  Reorient...', verbose, 'normal')
    original_info['orientation'] = im_target.orientation

    # assert images are in the same orientation
    assert im_target.orientation == im_sc_seg.orientation, "ERROR: the image to segment and it's SC segmentation are not in the same orientation"

    im_target_rpi = set_orientation(im_target, 'RPI')
    im_sc_seg_rpi = set_orientation(im_sc_seg, 'RPI')
    original_info['im_sc_seg_rpi'] = im_sc_seg_rpi.copy()  # target image in RPI will be used to post-process segmentations

    # denoise using P. Coupe non local means algorithm (see [Manjon et al. JMRI 2010]) implemented in dipy
    if denoising:
        printv('  Denoise...', verbose, 'normal')
        # crop image before denoising to fasten denoising
        nx, ny, nz, nt, px, py, pz, pt = im_target_rpi.dim
        size_x, size_y = (square_size_size_mm + 1) / px, (square_size_size_mm + 1) / py
        size = int(math.ceil(max(size_x, size_y)))
        # create mask
        fname_mask = 'mask_pre_crop.nii.gz'
        sct_create_mask.main(['-i', im_target_rpi.absolutepath, '-p', 'centerline,' + im_sc_seg_rpi.absolutepath, '-f', 'box', '-size', str(size), '-o', fname_mask])
        # crop image
        fname_target_crop = add_suffix(im_target_rpi.absolutepath, '_pre_crop')
        crop_im = ImageCropper(input_file=im_target_rpi.absolutepath, output_file=fname_target_crop, mask=fname_mask)
        im_target_rpi_crop = crop_im.crop()
        # crop segmentation
        fname_sc_seg_crop = add_suffix(im_sc_seg_rpi.absolutepath, '_pre_crop')
        crop_sc_seg = ImageCropper(input_file=im_sc_seg_rpi.absolutepath, output_file=fname_sc_seg_crop, mask=fname_mask)
        im_sc_seg_rpi_crop = crop_sc_seg.crop()
        # denoising
        from sct_maths import denoise_nlmeans
        block_radius = 3
        block_radius = int(im_target_rpi_crop.data.shape[2] / 2) if im_target_rpi_crop.data.shape[2] < (block_radius*2) else block_radius
        patch_radius = block_radius -1
        data_denoised = denoise_nlmeans(im_target_rpi_crop.data, block_radius=block_radius, patch_radius=patch_radius)
        im_target_rpi_crop.data = data_denoised

        im_target_rpi = im_target_rpi_crop
        im_sc_seg_rpi = im_sc_seg_rpi_crop
    else:
        fname_mask = None

    # interpolate image to reference square image (resample and square crop centered on SC)
    printv('  Interpolate data to the model space...', verbose, 'normal')
    list_im_slices = interpolate_im_to_ref(im_target_rpi, im_sc_seg_rpi, new_res=new_res, sq_size_size_mm=square_size_size_mm)
    original_info['interpolated_images'] = list_im_slices # list of images (not Slice() objects)

    printv('  Mask data using the spinal cord segmentation...', verbose, 'normal')
    list_sc_seg_slices = interpolate_im_to_ref(im_sc_seg_rpi, im_sc_seg_rpi, new_res=new_res, sq_size_size_mm=square_size_size_mm, interpolation_mode=1)
    for i in range(len(list_im_slices)):
        # list_im_slices[i].data[list_sc_seg_slices[i].data == 0] = 0
        list_sc_seg_slices[i] = binarize(list_sc_seg_slices[i], thr_min=0.5, thr_max=1)
        list_im_slices[i].data = list_im_slices[i].data * list_sc_seg_slices[i].data

    printv('  Split along rostro-caudal direction...', verbose, 'normal')
    list_slices_target = [Slice(slice_id=i, im=im_slice.data, gm_seg=[], wm_seg=[]) for i, im_slice in enumerate(list_im_slices)]

    # load vertebral levels
    if fname_level is not None:
        printv('  Load vertebral levels...', verbose, 'normal')
        # copy level file to tmp dir
        os.chdir(curdir)
        sct.copy(fname_level, tmp_dir)
        os.chdir(tmp_dir)
        # change fname level to only file name (path = tmp dir now)
        fname_level = ''.join(extract_fname(fname_level)[1:])
        # load levels
        list_slices_target = load_level(list_slices_target, fname_level)

    os.chdir(curdir)

    # load manual gmseg if there is one (model data)
    if fname_manual_gmseg is not None:
        printv('\n\tLoad manual GM segmentation(s) ...', verbose, 'normal')
        list_slices_target = load_manual_gmseg(list_slices_target, fname_manual_gmseg, tmp_dir, im_sc_seg_rpi, new_res, square_size_size_mm, for_model=for_model, fname_mask=fname_mask)

    if rm_tmp:
        # remove tmp folder
        shutil.rmtree(tmp_dir)
    return list_slices_target, original_info


# ----------------------------------------------------------------------------------------------------------------------
def interpolate_im_to_ref(im_input, im_input_sc, new_res=0.3, sq_size_size_mm=22.5, interpolation_mode=3):
    nx, ny, nz, nt, px, py, pz, pt = im_input.dim

    im_input_sc = im_input_sc.copy()
    im_input = im_input.copy()

    # keep only spacing and origin in qform to avoid rotation issues
    input_qform = im_input.hdr.get_qform()
    for i in range(4):
        for j in range(4):
            if i != j and j != 3:
                input_qform[i, j] = 0

    im_input.hdr.set_qform(input_qform)
    im_input.hdr.set_sform(input_qform)
    im_input_sc.hdr = im_input.hdr

    sq_size = int(sq_size_size_mm / new_res)
    # create a reference image : square of ones
    im_ref = Image(np.ones((sq_size, sq_size, 1), dtype=np.int), dim=(sq_size, sq_size, 1, 0, new_res, new_res, pz, 0), orientation='RPI')

    # copy input qform matrix to reference image
    im_ref.hdr.set_qform(im_input.hdr.get_qform())
    im_ref.hdr.set_sform(im_input.hdr.get_sform())

    # set correct header to reference image
    im_ref.hdr.set_data_shape((sq_size, sq_size, 1))
    im_ref.hdr.set_zooms((new_res, new_res, pz))

    # save image to set orientation to RPI (not properly done at the creation of the image)
    fname_ref = 'im_ref.nii.gz'
    im_ref.setFileName(fname_ref)
    im_ref.save()
    im_ref = set_orientation(im_ref, 'RPI', fname_out=fname_ref)

    # set header origin to zero to get physical coordinates of the center of the square
    im_ref.hdr.as_analyze_map()['qoffset_x'] = 0
    im_ref.hdr.as_analyze_map()['qoffset_y'] = 0
    im_ref.hdr.as_analyze_map()['qoffset_z'] = 0
    im_ref.hdr.set_sform(im_ref.hdr.get_qform())
    im_ref.hdr.set_qform(im_ref.hdr.get_qform())
    [[x_square_center_phys, y_square_center_phys, z_square_center_phys]] = im_ref.transfo_pix2phys(coordi=[[int(sq_size / 2), int(sq_size / 2), 0]])

    list_interpolate_images = []
    # iterate on z dimension of input image
    for iz in range(nz):
        # copy reference image: one reference image per slice
        im_ref_slice_iz = im_ref.copy()

        # get center of mass of SC for slice iz
        x_seg, y_seg = (im_input_sc.data[:, :, iz] > 0).nonzero()
        x_center, y_center = np.mean(x_seg), np.mean(y_seg)
        [[x_center_phys, y_center_phys, z_center_phys]] = im_input_sc.transfo_pix2phys(coordi=[[x_center, y_center, iz]])

        # center reference image on SC for slice iz
        im_ref_slice_iz.hdr.as_analyze_map()['qoffset_x'] = x_center_phys - x_square_center_phys
        im_ref_slice_iz.hdr.as_analyze_map()['qoffset_y'] = y_center_phys - y_square_center_phys
        im_ref_slice_iz.hdr.as_analyze_map()['qoffset_z'] = z_center_phys
        im_ref_slice_iz.hdr.set_sform(im_ref_slice_iz.hdr.get_qform())
        im_ref_slice_iz.hdr.set_qform(im_ref_slice_iz.hdr.get_qform())

        # interpolate input image to reference image
        im_input_interpolate_iz = im_input.interpolate_from_image(im_ref_slice_iz, interpolation_mode=interpolation_mode, border='nearest')
        # reshape data to 2D if needed
        if len(im_input_interpolate_iz.data.shape) == 3:
            im_input_interpolate_iz.data = im_input_interpolate_iz.data.reshape(im_input_interpolate_iz.data.shape[:-1])
        # add slice to list
        list_interpolate_images.append(im_input_interpolate_iz)

    return list_interpolate_images


# ----------------------------------------------------------------------------------------------------------------------
def load_level(list_slices_target, fname_level):
    verbose = 1
    path_level, file_level, ext_level = extract_fname(fname_level)

    #  ####### Check if the level file is an image or a text file
    # Level file is an image
    if ext_level in ['.nii', '.nii.gz']:
        im_level = Image(fname_level)
        im_level.change_orientation('IRP')

        list_level = []
        list_med_level = []
        for slice_level in im_level.data:
            try:
                # vertebral level of the slice
                l = np.mean(slice_level[slice_level > 0])
                # median of the vertebral level of the slice: if all voxels are int, med will be an int.
                med = np.median(slice_level[slice_level > 0])
                # change med in int if it is an int
                med = int(med) if int(med) == med else med
            except Exception, e:
                printv('WARNING: ' + str(e) + '\nNo level label found. Level will be set to 0 for this slice', verbose, 'warning')
                l = 0
                med = 0
            list_level.append(l)
            list_med_level.append(med)

        # if all median of level are int for all slices : consider level as int
        if all([isinstance(med, int) for med in list_med_level]):
            # level as int are placed in the middle of each vertebra (that's why there is a "+0.5")
            list_level = [int(round(l)) + 0.5 for l in list_level]

    # Level file is a text file
    elif ext_level == '.txt':
        file_level = open(fname_level, 'r')
        lines_level = file_level.readlines()
        file_level.close()

        list_level_by_slice = []
        list_type_level = []  # True or int, False for float
        for line in lines_level[1:]:
            i_slice, level = line.split(',')

            # correct level value
            for c in [' ', '\n', '\r', '\t']:
                level = level.replace(c, '')

            try:
                level = float(level)
            except Exception, e:
                # adapt if level value is not unique
                if len(level) > 2:
                    l1 = l2 = 0
                    if '-' in level:
                        l1, l2 = level.split('-')
                    elif '/' in level:
                        l1, l2 = level.split('/')
                    # convention = the vertebral disk between two levels belong to the lower level (C2-C3 = C3)
                    level = max(float(l1), float(l2))
                else:
                    # level unrecognized
                    level = 0

            i_slice = int(i_slice)

            list_type_level.append(int(level) == level)
            list_level_by_slice.append((i_slice, level))

        # sort list by number of slice
        list_level_by_slice.sort()

        if all(list_type_level):  # levels are int
            # add 0.5 to the int level to place in the middle of the vertebra
            to_add = 0.5
        else:
            # levels are float: keep them untouched
            to_add = 0

        list_level = [l[1] + to_add for l in list_level_by_slice]

    # Level file is not recognized
    else:
        list_level = None
        printv('ERROR: the level file is nor an image nor a text file ...', verbose, 'error')

    #  ####### Set level number for each slice of list_slices_target:
    for target_slice, level in zip(list_slices_target, list_level):
        target_slice.set(level=level)

    return list_slices_target


# ----------------------------------------------------------------------------------------------------------------------
def load_manual_gmseg(list_slices_target, list_fname_manual_gmseg, tmp_dir, im_sc_seg_rpi, new_res, square_size_size_mm, for_model=False, fname_mask=None):
    if isinstance(list_fname_manual_gmseg, str):
        # consider fname_manual_gmseg as a list of file names to allow multiple manual GM segmentation
        list_fname_manual_gmseg = [list_fname_manual_gmseg]

    curdir = os.getcwd()

    for fname_manual_gmseg in list_fname_manual_gmseg:
        sct.copy(fname_manual_gmseg, tmp_dir)
        # change fname level to only file name (path = tmp dir now)
        path_gm, file_gm, ext_gm = extract_fname(fname_manual_gmseg)
        fname_manual_gmseg = file_gm + ext_gm
        os.chdir(tmp_dir)

        im_manual_gmseg = Image(fname_manual_gmseg)

        # reorient to RPI
        im_manual_gmseg = set_orientation(im_manual_gmseg, 'RPI')

        if fname_mask is not None:
            fname_gmseg_crop = add_suffix(im_manual_gmseg.absolutepath, '_pre_crop')
            crop_im = ImageCropper(input_file=im_manual_gmseg.absolutepath, output_file=fname_gmseg_crop,
                                   mask=fname_mask)
            im_manual_gmseg_crop = crop_im.crop()
            im_manual_gmseg = im_manual_gmseg_crop

        # assert gmseg has the right number of slices
        assert im_manual_gmseg.data.shape[2] == len(list_slices_target), 'ERROR: the manual GM segmentation has not the same number of slices than the image.'

        # interpolate gm to reference image
        nz_gmseg, nx_gmseg, ny_gmseg, nt_gmseg, pz_gmseg, px_gmseg, py_gmseg, pt_gmseg = im_manual_gmseg.dim

        list_im_gm = interpolate_im_to_ref(im_manual_gmseg, im_sc_seg_rpi, new_res=new_res, sq_size_size_mm=square_size_size_mm, interpolation_mode=0)

        # load gm seg in list of slices
        n_poped = 0
        for im_gm, slice_im in zip(list_im_gm, list_slices_target):
            if im_gm.data.max() == 0 and for_model:
                list_slices_target.pop(slice_im.id - n_poped)
                n_poped += 1
            else:
                slice_im.gm_seg.append(im_gm.data)
                wm_slice = (slice_im.im > 0) - im_gm.data
                slice_im.wm_seg.append(wm_slice)

        os.chdir(curdir)

    return list_slices_target

########################################### End of pre-processing function #############################################


########################################################################################################################
#                               FUNCTIONS USED FOR PROCESSING DATA (data model and data to segment)
########################################################################################################################
def register_data(im_src, im_dest, param_reg, path_copy_warp=None, rm_tmp=True):
    '''

    Parameters
    ----------
    im_src: class Image: source image
    im_dest: class Image: destination image
    param_reg: str: registration parameter
    path_copy_warp: path: path to copy the warping fields

    Returns: im_src_reg: class Image: source image registered on destination image
    -------

    '''
    # im_src and im_dest are already preprocessed (in theory: im_dest = mean_image)
    # binarize images to get seg
    im_src_seg = binarize(im_src, thr_min=1, thr_max=1)
    im_dest_seg = binarize(im_dest)
    # create tmp dir and go in it
    tmp_dir = sct.tmp_create()
    curdir = os.getcwd()
    os.chdir(tmp_dir)
    # save image and seg
    fname_src = 'src.nii.gz'
    im_src.setFileName(fname_src)
    im_src.save()
    fname_src_seg = 'src_seg.nii.gz'
    im_src_seg.setFileName(fname_src_seg)
    im_src_seg.save()
    fname_dest = 'dest.nii.gz'
    im_dest.setFileName(fname_dest)
    im_dest.save()
    fname_dest_seg = 'dest_seg.nii.gz'
    im_dest_seg.setFileName(fname_dest_seg)
    im_dest_seg.save()
    # do registration using param_reg
    sct_register_multimodal.main(args=['-i', fname_src,
                                       '-d', fname_dest,
                                       '-iseg', fname_src_seg,
                                       '-dseg', fname_dest_seg,
                                       '-param', param_reg])

    # get registration result
    fname_src_reg = add_suffix(fname_src, '_reg')
    im_src_reg = Image(fname_src_reg)
    # get out of tmp dir
    os.chdir(curdir)

    # copy warping fields
    if path_copy_warp is not None and os.path.isdir(os.path.abspath(path_copy_warp)):
        path_copy_warp = os.path.abspath(path_copy_warp)
        file_src = extract_fname(fname_src)[1]
        file_dest = extract_fname(fname_dest)[1]
        fname_src2dest = 'warp_' + file_src + '2' + file_dest + '.nii.gz'
        fname_dest2src = 'warp_' + file_dest + '2' + file_src + '.nii.gz'
        sct.copy(os.path.join(tmp_dir, fname_src2dest), path_copy_warp)
        sct.copy(os.path.join(tmp_dir, fname_dest2src), path_copy_warp)

    if rm_tmp:
        # remove tmp dir
        shutil.rmtree(tmp_dir)
    # return res image
    return im_src_reg, fname_src2dest, fname_dest2src


def apply_transfo(im_src, im_dest, warp, interp='spline', rm_tmp=True):
    # create tmp dir and go in it
    tmp_dir = sct.tmp_create()
    # copy warping field to tmp dir
    sct.copy(warp, tmp_dir)
    warp = ''.join(extract_fname(warp)[1:])
    # go to tmp dir
    curdir = os.getcwd()
    os.chdir(tmp_dir)
    # save image and seg
    fname_src = 'src.nii.gz'
    im_src.setFileName(fname_src)
    im_src.save()
    fname_dest = 'dest.nii.gz'
    im_dest.setFileName(fname_dest)
    im_dest.save()
    # apply warping field
    fname_src_reg = add_suffix(fname_src, '_reg')
    sct_apply_transfo.main(args=['-i', fname_src,
                                  '-d', fname_dest,
                                  '-w', warp,
                                  '-x', interp])

    im_src_reg = Image(fname_src_reg)
    # get out of tmp dir
    os.chdir(curdir)
    if rm_tmp:
        # remove tmp dir
        shutil.rmtree(tmp_dir)
    # return res image
    return im_src_reg


# ------------------------------------------------------------------------------------------------------------------
def average_gm_wm(list_of_slices, model_space=True, bin=False):
    # compute mean GM and WM image
    list_gm = []
    list_wm = []
    for dic_slice in list_of_slices:
        if model_space:
            for wm in dic_slice.wm_seg_M:
                list_wm.append(wm)
            for gm in dic_slice.gm_seg_M:
                list_gm.append(gm)
        else:
            for wm in dic_slice.wm_seg:
                list_wm.append(wm)
            for gm in dic_slice.gm_seg:
                list_gm.append(gm)

    data_mean_gm = np.mean(list_gm, axis=0)
    data_mean_wm = np.mean(list_wm, axis=0)

    if bin:
        data_mean_gm[data_mean_gm < 0.5] = 0
        data_mean_gm[data_mean_gm >= 0.5] = 1
        data_mean_wm[data_mean_wm < 0.5] = 0
        data_mean_wm[data_mean_wm >= 0.5] = 1

    return data_mean_gm, data_mean_wm


def normalize_slice(data, data_gm, data_wm, val_gm, val_wm, val_min=None, val_max=None):
    '''
    Function to normalize the intensity of data to the GM and WM values given by val_gm and val_wm.
    All parameters are arrays
    Parameters
    ----------
    data : ndarray: data to normalized
    data_gm : ndarray: data to get slice GM value from
    data_wm : ndarray: data to get slice WM value from
    val_gm : GM value to normalize data on
    val_wm : WM value to normalize data on

    Returns
    -------
    '''
    assert data.size == data_gm.size, "Data to normalized and GM data do not have the same shape."
    assert data.size == data_wm.size, "Data to normalized and WM data do not have the same shape."
    # avoid shape error because of 3D-like shape for 2D (x, x, 1) instead of (x,x)
    data_gm = data_gm.reshape(data.shape)
    data_wm = data_wm.reshape(data.shape)

    # put almost zero background to zero
    data[data < 0.01] = 0

    # binarize GM and WM data if needed
    if np.min(data_gm) != 0 or np.max(data_gm) != 1:
        data_gm[data_gm < 0.5] = 0
        data_gm[data_gm >= 0.5] = 1
    if np.min(data_wm) != 0 or np.max(data_wm) != 1:
        data_wm[data_wm < 0.5] = 0
        data_wm[data_wm >= 0.5] = 1

    # get GM and WM values in slice
    data_in_gm = data[data_gm == 1]
    data_in_wm = data[data_wm == 1]
    med_data_gm = np.median(data_in_gm)
    med_data_wm = np.median(data_in_wm)
    std_data = np.mean([np.std(data_in_gm), np.std(data_in_wm)])

    # compute normalized data
    # if median values are too close: use min and max to normalize data
    if abs(med_data_gm - med_data_wm) < std_data and val_min is not None and val_max is not None:
        try:
            min_data = min(np.min(data_in_gm[data_in_gm.nonzero()]), np.min(data_in_wm[data_in_wm.nonzero()]))
            max_data = max(np.max(data_in_gm[data_in_gm.nonzero()]), np.max(data_in_wm[data_in_wm.nonzero()]))
            new_data = ((data - min_data) * (val_max - val_min) / (max_data - min_data)) + val_min
        except ValueError:
            printv('WARNING: an incomplete slice will not be normalized', 1, 'warning')
            return data
    # else (=normal data): use median values to normalize data
    else:
        new_data = ((data - med_data_wm) * (val_gm - val_wm) / (med_data_gm - med_data_wm)) + val_wm

    # put almost zero background to zero
    new_data[data < 0.01] = 0  # put at 0 the background
    new_data[new_data < 0.01] = 0  # put at 0 the background

    # return normalized data
    return new_data

########################################### End of processing function #############################################


########################################################################################################################
#                                                   UTILS FUNCTIONS
########################################################################################################################
def binarize(im, thr_min=None, thr_max=None):
    if thr_min is None and thr_max is not None:
        thr_min = thr_max
    if thr_max is None and thr_min is not None:
        thr_max = thr_min
    if thr_min is None and thr_max is None:
        thr_min = thr_max = np.max(im.data) / 2
    im_bin = im.copy()
    im_bin.data[im.data >= thr_max] = 1
    im_bin.data[im.data < thr_min] = 0

    return im_bin
