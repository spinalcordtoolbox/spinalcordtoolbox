#!/usr/bin/env python
########################################################################################################################
#
#
# Utility functions used for the segmentation of the gray matter
#
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Sara Dupont
# Modified: 2016-06-14
#
# About the license: see the file LICENSE.TXT
########################################################################################################################
from msct_image import Image
from sct_image import set_orientation, pad_image
from sct_resample import main as sct_resample_main
from sct_create_mask import main as sct_create_mask_main
from sct_utils import add_suffix, extract_fname, printv
from sct_maths import denoise_ornlm
import numpy as np

########################################################################################################################
#                                   CLASS SLICE
########################################################################################################################
class Slice:
    """
    Slice instance used in the model dictionary for the segmentation of the gray matter
    """
    def __init__(self, slice_id=None, im=None, sc_seg=None, gm_seg=None, wm_seg=None, reg_to_m=None, im_m=None, gm_seg_m=None, wm_seg_m=None, level=None):
        """
        Slice constructor

        :param slice_id: slice ID number, type: int

        :param im: original image (a T2star 2D image croped around the spinal cord), type: numpy array

        :param gm_seg: manual gray matter segmentation of the original image, type: numpy array

        :param wm_seg: manual white matter segmentation of the original image, type: numpy array

        :param reg_to_m: name of the file containing the transformation for this slice to go from the image original space to the model space, type: string

        :param im_m: image in the model space, type: numpy array

        :param gm_seg_m: manual gray matter segmentation in the model space, type: numpy array

        :param wm_seg_m: manual white matter segmentation in the model space, type: numpy array

        :param level: vertebral level of the slice, type: int
        """
        self.id = slice_id
        self.im = im
        self.sc_seg = sc_seg
        self.gm_seg = gm_seg
        self.wm_seg = wm_seg
        self.reg_to_M = reg_to_m
        self.im_M = im_m
        self.gm_seg_M = gm_seg_m
        self.wm_seg_M = wm_seg_m
        self.level = level

    def set(self, slice_id=None, im=None, sc_seg=None, gm_seg=None, wm_seg=None, reg_to_m=None, im_m=None, gm_seg_m=None, wm_seg_m=None, level=None):
        """
        Slice setter, only the specified parameters are set

        :param slice_id: slice ID number, type: int

        :param im: original image (a T2star 2D image croped around the spinal cord), type: numpy array

        :param gm_seg: manual gray matter segmentation of the original image, type: numpy array

        :param wm_seg: manual white matter segmentation of the original image, type: numpy array

        :param reg_to_m: name of the file containing the transformation for this slice to go from the image original space to the model space, type: string

        :param im_m: image in the model space, type: numpy array

        :param gm_seg_m: manual gray matter segmentation in the model space, type: numpy array

        :param wm_seg_m: manual white matter segmentation in the model space, type: numpy array

        :param level: vertebral level of the slice, type: int
        """
        if slice_id is not None:
            self.id = slice_id
        if im is not None:
            self.im = im
        if sc_seg is not None:
            self.sc_seg = sc_seg
        if gm_seg is not None:
            self.gm_seg = gm_seg
        if wm_seg is not None:
            self.wm_seg = wm_seg
        if reg_to_m is not None:
            self.reg_to_M = reg_to_m
        if im_m is not None:
            self.im_M = im_m
        if gm_seg_m is not None:
            self.gm_seg_M = gm_seg_m
        if wm_seg_m is not None:
            self.wm_seg_M = wm_seg_m
        if level is not None:
            self.level = level

    def __repr__(self):
        s = '\nSlice #' + str(self.id)
        if self.level is not None:
            s += 'Level : ' + str(self.level)
        s += '\nImage : \n' + str(self.im)
        s += '\nGray matter segmentation : \n' + str(self.gm_seg)
        s += '\nTransformation to model space : ' + self.reg_to_M
        if self.im_M is not None:
            s += '\nImage in the common model space: \n' + str(self.im_M)
        if self.gm_seg_M is not None:
            s += '\nGray matter segmentation in the common model space: \n' + str(self.gm_seg_M)
        return s


########################################################################################################################
#                               FUNCTIONS USED FOR PRE-PROCESSING
########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
def pre_processing(fname_target, fname_sc_seg, fname_level=None, new_res=0.3, mask_size_mm=22.5, denoising=True, verbose=1):
    printv('\nPre-processing data ...', verbose, 'normal')
    original_info = {}

    im_target = Image(fname_target)
    im_sc_seg = Image(fname_sc_seg)


    # get original orientation
    printv('\n\tReorient ...', verbose, 'normal')
    original_info['orientation'] = im_target.orientation
    # assert images are in the same orientation
    assert im_target.orientation == im_sc_seg.orientation, "ERROR: the image to segment and it's SC segmentation are not in the same orientation"

    # reorient data (IRP is convenient to split along rostro-caudal direction)
    im_target = set_orientation(im_target, 'IRP')
    im_sc_seg = set_orientation(im_sc_seg, 'IRP')

    fname_target = im_target.absolutepath
    fname_sc_seg = im_sc_seg.absolutepath

    # assert size of images are identical
    assert im_target.data.shape == im_sc_seg.data.shape, "ERROR: the image to segment and it's SC segmentation does not have the same size"

    # get original pixel size: x = right-left, y = antero-posterior, z = inferior-superior (IRP = zxy)
    nz_target, nx_target, ny_target, nt_target, pz_target, px_target, py_target, pt_target = im_target.dim
    original_info['px'] = px_target
    original_info['py'] = py_target

    # resample to an axial resolution of -new_res-
    printv('\n\tResample to an axial resolution of '+str(new_res)+'x'+str(new_res)+'mm2 ...', verbose, 'normal')
    im_target = axial_resample(im_target, new_res)
    im_sc_seg = axial_resample(im_sc_seg, new_res)

    # denoise using P. Coupe algorithm (see [Manjon et al. JMRI 2010])
    if denoising:
        printv('\n\tDenoise ...', verbose, 'normal')
        im_target.data = denoise_ornlm(im_target.data)

    # pad in case SC is too close to the edges
    printv('\n\tPad in case the spinal cord is to close to the edges ...', verbose, 'normal')
    mask_size_pix = int(mask_size_mm/new_res)
    pad = mask_size_pix/2 + 2

    im_target = pad_image(im_target, pad_x_i=0, pad_x_f=0, pad_y_i=pad, pad_y_f=pad, pad_z_i=pad, pad_z_f=pad) # x=IS, y=RL, z=PA
    im_sc_seg = pad_image(im_sc_seg, pad_x_i=0, pad_x_f=0, pad_y_i=pad, pad_y_f=pad, pad_z_i=pad, pad_z_f=pad) # x=IS, y=RL, z=PA

    # mask using SC and crop around a square mask
    printv('\n\tMask and crop data ...', verbose, 'normal')
    im_target, im_square_mask = mask_and_crop_target(im_target, im_sc_seg, mask_size_pix, verbose=verbose)
    original_info['square_mask'] = im_square_mask

    # split along rostro-caudal direction create a list of axial slices
    printv('\n\tSplit along rostro-caudal direction...', verbose, 'normal')
    list_slices_target = [Slice(slice_id=i, im=data_slice) for i, data_slice in enumerate(im_target.data)]

    # load vertebral levels
    if fname_level is not None:
        printv('\n\tLoad vertebral levels ...', verbose, 'normal')
        list_slices_target = load_level(list_slices_target, fname_level)

    printv('\nPre-processing done!', verbose, 'normal')
    return list_slices_target, original_info


# ----------------------------------------------------------------------------------------------------------------------
def axial_resample(im, npx):
    fname = im.absolutepath
    fname_resample = add_suffix(fname, '_r')

    # image must be in IRP
    nz, nx, ny, nt, pz, px, py, pt = im.dim
    sct_resample_main(['-i', fname, '-mm', str(pz)+'x'+str(npx)+'x'+str(npx), '-o', fname_resample, '-v', '0'])
    im_resample = Image(fname_resample)

    return im_resample


# ----------------------------------------------------------------------------------------------------------------------
def mask_and_crop_target(im_target, im_sc_seg, mask_size_pix, verbose=1):
    # mask target using SC mask (= set background to zero)
    printv('\n\t\tMask data using the spinal cord mask ...', verbose, 'normal')
    im_target.data[im_sc_seg.data == 0] = 0
    fname_target = add_suffix(im_target.absolutepath, '_masked_sc')
    im_target.setFileName(fname_target)
    im_target.save()

    # save im_sc_seg so that it can be called by sct_create_mask
    fname_sc_seg = im_sc_seg.absolutepath
    im_sc_seg.save()

    # create a square mask and use it to crop
    printv('\n\t\tCreate a square mask centered in the spinal cord ...', verbose, 'normal')
    fname_mask = "square_mask.nii.gz"
    sct_create_mask_main(['-i', fname_target, '-p', 'centerline,'+fname_sc_seg, '-size', str(mask_size_pix), '-f', 'box', '-o', fname_mask, '-v', '0'])
    im_mask = Image(fname_mask)

    # crop along the square mask and stack slices
    printv('\n\t\tCrop using square mask and stack slices ...', verbose, 'normal')
    im_target.crop_and_stack(im_mask, suffix='', save=False)

    return im_target, im_mask


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
                med = int(med) if int(med)==med else med
            except Exception, e:
                printv('WARNING: ' + str(e) + '\nNo level label found. Level will be set to 0 for this slice', verbose, 'warning')
                l = 0
                med = 0
            list_level.append(l)
            list_med_level.append(med)

        # if all median of level are int for all slices : consider level as int
        if all([type(med) == int for med in list_med_level]):
            # level as int are placed in the middle of each vertebra (that's why there is a "+0.5")
            list_level = [int(round(l))+0.5 for l in list_level]

    # Level file is a text file
    elif ext_level == '.txt':
        file_level = open(fname_level, 'r')
        lines_level = file_level.readlines()
        file_level.close()

        list_level_by_slice = []
        list_type_level = []  # True or int, False for float
        for line in lines_level[1:]:
            i_slice, level = line.split(',')
            i_slice, level = int(i_slice), float(level)
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

        list_level = [l[1]+to_add for l in list_level_by_slice]

    # Level file is not recognized
    else:
        list_level = None
        printv('ERROR: the level file is nor an image nor a text file ...', verbose, 'error')

    #  ####### Set level number for each slice of list_slices_target:
    for target_slice, level in zip(list_slices_target, list_level):
        target_slice.set(level=level)

    return list_slices_target

########################################### End of pre-processing function #############################################


