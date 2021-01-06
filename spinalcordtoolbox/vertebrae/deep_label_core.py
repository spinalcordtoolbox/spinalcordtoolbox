#!/usr/bin/env python
# -*- coding: utf-8
# Core functions dealing with vertebral labeling

# TODO: remove i/o as much as possible

import os
import logging

import numpy as np
import scipy.ndimage.measurements
from scipy.ndimage.filters import gaussian_filter
from ivadomed import preprocessing as imed_preprocessing
import nibabel as nib
from spinalcordtoolbox.vertebrae.core import label_discs, label_segmentation, center_of_mass, create_label_z
import logging
import spinalcordtoolbox.scripts.sct_deepseg as sct_deepseg
from scipy.signal import gaussian

logging.getLogger('matplotlib.font_manager').disabled = True

from spinalcordtoolbox.metadata import get_file_label
from spinalcordtoolbox.math import dilate, mutual_information
from spinalcordtoolbox.utils.sys import run_proc, printv
from spinalcordtoolbox.image import Image, add_suffix

logger = logging.getLogger(__name__)


def label_vert(fname_seg, fname_label, verbose=1):
    """
    Label segmentation using vertebral labeling information. No orientation expected.

    :param fname_seg: file name of segmentation.
    :param fname_label: file name for a labelled segmentation that will be used to label the input segmentation
    :param fname_out: file name of the output labeled segmentation. If empty, will add suffix "_labeled" to fname_seg
    :param verbose:
    :return:
    """
    # Open labels
    im_disc = Image(fname_label).change_orientation("RPI")
    # retrieve all labels
    coord_label = im_disc.getNonZeroCoordinates()
    # compute list_disc_z and list_disc_value
    list_disc_z = []
    list_disc_value = []
    for i in range(len(coord_label)):
        list_disc_z.insert(0, coord_label[i].z)
        # '-1' to use the convention "disc labelvalue=3 ==> disc C2/C3"
        list_disc_value.insert(0, coord_label[i].value - 1)

    list_disc_value = [x for (y, x) in sorted(zip(list_disc_z, list_disc_value), reverse=True)]
    list_disc_z = [y for (y, x) in sorted(zip(list_disc_z, list_disc_value), reverse=True)]
    # label segmentation
    label_segmentation(fname_seg, list_disc_z, list_disc_value, verbose=verbose)
    label_discs(fname_seg, list_disc_z, list_disc_value, verbose=verbose)


def vertebral_detection(fname, fname_seg, contrast, param, init_disc, verbose=1, path_template='', path_output='../',
                        scale_dist=1.):
    """
    Find intervertebral discs in straightened image using template matching

    :param fname: file name of straigthened spinal cord
    :param fname_seg: file name of straigthened spinal cord segmentation
    :param contrast: t1 or t2
    :param param:  advanced parameters
    :param init_disc: reference coordinates and value for a disc. c2/c3 is often used and automatically detected with sct_label_vertebrae
    :param verbose:
    :param path_template: path to the used template. Template should be a straighten image with sing-voxel labels on the posterior tip of each disc
    :param path_output: output path for verbose=2 pictures
    :param scale_dist: float: Scaling factor to adjust average distance between two adjacent intervertebral discs
    :return: None

    """
    logger.info('Look for template...')
    logger.info('Path template: %s', path_template)

    # adjust file names if MNI-Poly-AMU template is used (by default: PAM50)
    fname_template = get_file_label(os.path.join(path_template, 'template'), id_label=11,
                                    output='filewithpath')  # label = intevertebral dic label template (PAM50)

    # Open template and vertebral levels
    logger.info('Open template and vertebral levels...')
    data_template = Image(fname_template).data

    # open anatomical volume
    im_input = Image(fname)
    data = im_input.data

    # smooth data
    data = gaussian_filter(data, param.smooth_factor, output=None, mode="reflect")

    # get dimension of src
    nx, ny, nz = data.shape
    # define xc and yc (centered in the field of view)
    xc = int(np.round(nx / 2))  # direction RL
    yc = int(np.round(ny / 2))  # direction AP
    # get dimension of template
    nxt, nyt, nzt = data_template.shape
    # define xc and yc (centered in the field of view)
    xct = int(np.round(nxt / 2))  # direction RL
    yct = int(np.round(nyt / 2))  # direction AP

    # define mean distance (in voxel) between adjacent discs: [C1/C2 -> C2/C3], [C2/C3 -> C4/C5], ..., [L1/L2 -> L2/L3]
    # attribute value to each disc. Starts from max level, then decrease.
    min_level = data_template[data_template.nonzero()].min()
    max_level = data_template[data_template.nonzero()].max()
    list_disc_value_template = list(range(min_level, max_level))
    # add disc above top one
    list_disc_value_template.insert(int(0), min_level - 1)
    printv('\nDisc values from template: ' + str(list_disc_value_template), verbose)
    # get disc z-values
    list_disc_z_template = data_template.nonzero()[2].tolist()
    list_disc_z_template.sort()
    list_disc_z_template.reverse()
    logger.info('Z-values for each disc: %s', list_disc_z_template)
    list_distance_template = (
            np.diff(list_disc_z_template) * (-1)).tolist()  # multiplies by -1 to get positive distances
    # Update distance with scaling factor
    list_distance_template = [i * scale_dist for i in list_distance_template]
    logger.info('Distances between discs (in voxel): %s', list_distance_template)

    # display init disc
    if verbose == 2:
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        fig_disc = Figure()
        FigureCanvas(fig_disc)
        ax_disc = fig_disc.add_subplot(111)
        # ax_disc = fig_disc.add_axes((0, 0, 1, 1))
        # get percentile for automatic contrast adjustment
        data_display = np.mean(data[xc - param.size_RL:xc + param.size_RL, :, :], axis=0).transpose()
        percmin = np.percentile(data_display, 10)
        percmax = np.percentile(data_display, 90)
        # display image
        ax_disc.matshow(data_display, cmap='gray', clim=[percmin, percmax], origin='lower')
        ax_disc.set_title('Anatomical image')
        # ax.autoscale(enable=False)  # to prevent autoscale of axis when displaying plot
        ax_disc.scatter(yc + param.shift_AP_visu, init_disc[0], c='yellow', s=10)
        ax_disc.text(yc + param.shift_AP_visu + 4, init_disc[0], str(init_disc[1]) + '/' + str(init_disc[1] + 1),
                     verticalalignment='center', horizontalalignment='left', color='pink', fontsize=7)

    # FIND DISCS
    # ===========================================================================
    logger.info('Detect intervertebral discs...')
    # assign initial z and disc
    current_z = init_disc[0]
    current_disc = init_disc[1]
    # create list for z and disc
    list_disc_z = []
    list_disc_value = []
    zrange = list(range(-10, 10))
    direction = 'superior'
    search_next_disc = True
    # image is straighten and oriented according to RPI convention before.
    mid_index = int(np.round(nib.load(fname).header.get_data_shape()[0]/2.0))
    image_mid = imed_preprocessing.get_midslice_average(fname, mid_index)
    nib.save(image_mid, "input_image.nii.gz")
    if contrast == "t2":
        sct_deepseg.main(['-i', 'input_image.nii.gz', '-task', 'find_disc_t2', '-thr', '-1', '-o', 'hm_tmp.nii.gz'])
    elif contrast == "t1":
        sct_deepseg.main(['-i', 'input_image.nii.gz', '-task', 'find_disc_t1', '-thr', '-1', '-o', 'hm_tmp.nii.gz'])

    run_proc(['sct_resample', '-i', 'hm_tmp.nii.gz', '-mm', '0.5x0.5x0.5', '-x', 'linear', '-o', 'hm_tmp_r.nii.gz'])
    run_proc(['sct_resample', '-i', fname_seg, '-mm', '0.5x0.5x0.5', '-x', 'nn', '-o', fname_seg])
    im_hm = Image('hm_tmp_r.nii.gz')
    data_hm = im_hm.data
    while search_next_disc:
        logger.info('Current disc: %s (z=%s). Direction: %s', current_disc, current_z, direction)

        try:
            # get z corresponding to current disc on template
            current_z_template = list_disc_z_template[current_disc]
        except:
            # in case reached the bottom (see issue #849)
            logger.warning('Reached the bottom of the template. Stop searching.')
            break
        # find next disc
        # N.B. Do not search for C1/C2 disc (because poorly visible), use template distance instead
        if current_disc != 1:
            current_z = compute_corr_3d(data_hm, data_template, x=xc, xshift=0, xsize=param.size_RL,
                                        y=yc, yshift=param.shift_AP, ysize=param.size_AP,
                                        z=current_z, zshift=0, zsize=param.size_IS,
                                        xtarget=xct, ytarget=yct, ztarget=current_z_template,
                                        zrange=zrange, verbose=verbose, save_suffix='_disc' + str(current_disc),
                                        gaussian_std=999, path_output=path_output)

        # display new disc
        if verbose == 2:
            ax_disc.scatter(yc + param.shift_AP_visu, current_z, c='yellow', s=10)
            ax_disc.text(yc + param.shift_AP_visu + 4, current_z, str(current_disc) + '/' + str(current_disc + 1),
                         verticalalignment='center', horizontalalignment='left', color='yellow', fontsize=7)

        # append to main list
        if direction == 'superior':
            # append at the beginning
            list_disc_z.insert(0, current_z)
            list_disc_value.insert(0, current_disc)
        elif direction == 'inferior':
            # append at the end
            list_disc_z.append(current_z)
            list_disc_value.append(current_disc)

        # adjust correcting factor based on already-identified discs
        if len(list_disc_z) > 1:
            # compute distance between already-identified discs
            list_distance_current = (np.diff(list_disc_z) * (-1)).tolist()
            # retrieve the template distance corresponding to the already-identified discs
            index_disc_identified = [i for i, j in enumerate(list_disc_value_template) if j in list_disc_value[:-1]]
            list_distance_template_identified = [list_distance_template[i] for i in index_disc_identified]
            # divide subject and template distances for the identified discs
            list_subject_to_template_distance = [float(list_distance_current[i]) / list_distance_template_identified[i]
                                                 for i in range(len(list_distance_current))]
            # average across identified discs to obtain an average correcting factor
            correcting_factor = np.mean(list_subject_to_template_distance)
            logger.info('.. correcting factor: %s', correcting_factor)
        else:
            correcting_factor = 1
        # update list_distance specific for the subject
        list_distance = [int(np.round(list_distance_template[i] * correcting_factor)) for i in
                         range(len(list_distance_template))]

        # assign new current_z and disc value
        if direction == 'superior':
            try:
                approx_distance_to_next_disc = list_distance[list_disc_value_template.index(current_disc - 1)]
            except (IndexError, ValueError):
                logger.warning('Disc value not included in template. Using previously-calculated distance: %s', approx_distance_to_next_disc)
            # assign new current_z and disc value
            current_z = current_z + approx_distance_to_next_disc
            current_disc = current_disc - 1
        elif direction == 'inferior':
            try:
                approx_distance_to_next_disc = list_distance[list_disc_value_template.index(current_disc)]
            except (IndexError, ValueError):
                logger.warning('Disc value not included in template. Using previously-calculated distance: %s', approx_distance_to_next_disc)
            # assign new current_z and disc value
            current_z = current_z - approx_distance_to_next_disc
            current_disc = current_disc + 1

        # if current_z is larger than searching zone, switch direction (and start from initial z minus approximate
        # distance from updated template distance)
        if current_z >= nz or current_disc == 0:
            logger.info('.. Switching to inferior direction.')
            direction = 'inferior'
            current_disc = init_disc[1] + 1
            current_z = init_disc[0] - list_distance[list_disc_value_template.index(current_disc)]
        # if current_z is lower than searching zone, stop searching
        if current_z <= 0:
            search_next_disc = False

    if verbose == 2:
        fig_disc.savefig('fig_label_discs.png')

    # if upper disc is not 1, add disc above top disc based on mean_distance_adjusted
    upper_disc = min(list_disc_value)
    # if not upper_disc == 1:
    logger.info('Adding top disc based on adjusted template distance: #%s', upper_disc - 1)
    approx_distance_to_next_disc = list_distance[list_disc_value_template.index(upper_disc - 1)]
    next_z = max(list_disc_z) + approx_distance_to_next_disc
    logger.info('.. approximate distance: %s', approx_distance_to_next_disc)
    # make sure next disc does not go beyond FOV in superior direction
    if next_z > nz:
        list_disc_z.insert(0, nz)
    else:
        list_disc_z.insert(0, next_z)
    # assign disc value
    list_disc_value.insert(0, upper_disc - 1)

    # Label segmentation
    label_segmentation(fname_seg, list_disc_z, list_disc_value, verbose=verbose)
    label_disc_posterior(list_disc_z, list_disc_value, 'hm_tmp_r.nii.gz', fname_seg)
    label_discs(fname_seg, list_disc_z, list_disc_value, verbose=verbose)


def get_z_and_disc_values_from_label(fname_label):
    """
    Find z-value and label-value based on labeled image in RPI orientation

    :param fname_label: image in RPI orientation that contains label
    :return: [z_label, value_label] int list
    """
    nii = Image(fname_label)
    # get center of mass of label
    x_label, y_label, z_label = center_of_mass(nii.data)
    x_label, y_label, z_label = int(np.round(x_label)), int(np.round(y_label)), int(np.round(z_label))
    # get label value
    value_label = int(nii.data[x_label, y_label, z_label])
    return [z_label, value_label]


def compute_corr_3d(src, target, x, xshift, xsize, y, yshift, ysize, z, zshift, zsize, xtarget, ytarget, ztarget,
                    zrange, verbose, save_suffix, gaussian_std, path_output):
    """
    FIXME doc
    Find z that maximizes correlation between src and target 3d data.

    :param src: 3d source data
    :param target: 3d target data
    :param x:
    :param xshift:
    :param xsize:
    :param y:
    :param yshift:
    :param ysize:
    :param z:
    :param zshift:
    :param zsize:
    :param xtarget:
    :param ytarget:
    :param ztarget:
    :param zrange:
    :param verbose:
    :param save_suffix:
    :param gaussian_std:
    :return:
    """
    # parameters
    thr_corr = 0.2  # disc correlation threshold. Below this value, use template distance.
    src = src[:, :, :]
    # get dimensions from src
    nx, ny, nz = src.shape
    # Get pattern from template
    pattern = target[xtarget - xsize:xtarget + xsize,
                     ytarget + yshift - ysize: ytarget + yshift + ysize + 1,
                     ztarget + zshift - zsize: ztarget + zshift + zsize + 1]
    pattern1d = np.sum(pattern, axis=(0, 1))
    # convolve pattern1d with gaussian to get similar curve as input
    a = gaussian(30, std=5)
    pattern1d = np.convolve(pattern1d, a, 'same')
    # initializations
    I_corr = np.zeros(len(zrange))
    allzeros = 0
    # current_z = 0
    ind_I = 0
    # loop across range of z defined by src
    for iz in zrange:
        # if pattern extends towards the top part of the image, then crop and pad with zeros
        if z + iz + zsize + 1 > nz:
            padding_size = z + iz + zsize + 1 - nz
            data_chunk3d = src[:,
                               y + yshift: y + yshift + ysize + 1,
                               z + iz - zsize: z + iz + zsize + 1 - padding_size]
            data_chunk3d = np.pad(data_chunk3d, ((0, 0), (0, 0), (0, padding_size)), 'constant',
                                  constant_values=0)
        # if pattern extends towards bottom part of the image, then crop and pad with zeros
        elif z + iz - zsize < 0:
            padding_size = abs(iz - zsize)
            data_chunk3d = src[:,
                               y + yshift - ysize: y + yshift + ysize + 1,
                               z + iz - zsize + padding_size: z + iz + zsize + 1]
            data_chunk3d = np.pad(data_chunk3d, ((0, 0), (0, 0), (padding_size, 0)), 'constant',
                                  constant_values=0)
        else:
            data_chunk3d = src[:,
                               :ytarget + ysize,
                               z + iz - zsize: z + iz + zsize + 1]

        # convert subject pattern to 1d profile
        data_chunk1d = np.sum(data_chunk3d, axis=(0, 1))
        # check if data_chunk1d contains at least one non-zero value
        if (data_chunk1d.size == pattern1d.size) and np.any(data_chunk1d) and np.any(pattern1d):
            # Normalize value before correlation (correlation tends to diminish otherwise since label value increases)
            pattern1d_norm = pattern1d / (max(pattern1d))
            data_chunk1d_norm = data_chunk1d / (max(data_chunk1d))
            a = np.correlate(data_chunk1d_norm, pattern1d_norm)
            I_corr[ind_I] = a
        else:
            allzeros = 1
        ind_I = ind_I + 1
    # ind_y = ind_y + 1
    if allzeros:
        logger.warning('Data contained zero. We probably hit the edge of the image.')

    # adjust correlation with Gaussian function centered at the right edge of the curve (most rostral point of FOV)
    gaussian_window = gaussian(len(I_corr) * 2, std=len(I_corr) * gaussian_std)
    I_corr_gauss = np.multiply(I_corr, gaussian_window[0:len(I_corr)])

    # Find global maximum
    if np.any(I_corr_gauss):
        # if I_corr contains at least a non-zero value
        ind_peak = np.argmax(I_corr_gauss)  # index of max along z
        ind_dl = np.argmax(data_chunk1d)
        logger.info('.. Peak found: z=%s (correlation = %s)', zrange[ind_peak], I_corr_gauss[ind_peak])
        # check if correlation is high enough
        if I_corr_gauss[ind_peak] < thr_corr:
            logger.warning('Correlation is too low. Using adjusted template distance.')
            ind_peak = zrange.index(0)  # approx_distance_to_next_disc
            ind_dl = ind_peak
    else:
        # if I_corr contains only zeros
        logger.warning('Correlation vector only contains zeros. Using adjusted template distance.')
        ind_peak = zrange.index(0)  # approx_distance_to_next_disc
        ind_dl = ind_peak

    # display patterns and correlation
    if verbose == 2:
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        fig = Figure(figsize=(15, 7))
        FigureCanvas(fig)
        # display template pattern
        ax = fig.add_subplot(131)
        ax.plot(pattern1d)
        ax.set_title('Template pattern')
        # display subject pattern at best z
        ax = fig.add_subplot(132)
        iz = zrange[ind_peak]
        data_chunk3d = src[:,
                           y + yshift - ysize: y + yshift + ysize + 1,
                           z - iz - zsize: z + iz + zsize + 1]
        ax.plot(np.sum(data_chunk3d, axis=(0, 1)))
        ax.set_title('Subject accross all iz')
        # display correlation curve
        ax = fig.add_subplot(133)
        ax.plot(range(len(I_corr)), I_corr)
        ax.plot(range(len(I_corr)), I_corr_gauss, 'black', linestyle='dashed')
        ax.legend(['I_corr', 'I_corr_gauss'])
        ax.set_title('Mutual Info, gaussian_std=' + str(gaussian_std))
        ax.plot(ind_peak, I_corr_gauss[ind_peak], 'ro')
        ax.axvline(x=zrange.index(0), linewidth=1, color='black', linestyle='dashed')
        ax.axhline(y=thr_corr, linewidth=1, color='r', linestyle='dashed')
        ax.grid()
        # save figure
        fig.savefig('fig_pattern' + save_suffix + '.png')
        np.save('pattern' + save_suffix + '.npy', pattern1d)
        # show figure for each iz
        i = 1
        j = 1
        ind = 1
        fig = Figure(figsize=(15, 15))
        FigureCanvas(fig)
        for indic in zrange:
            ax = fig.add_subplot(4, 5, ind)
            data_chunk3d = src[:,
                               y + yshift - ysize: y + yshift + ysize + 1,
                               z + indic - zsize: z + indic + zsize + 1]
            ax.plot(np.sum(data_chunk3d, axis=(0, 1)))
            ax.set_title('Subject at iz' + str(indic))
            ind = ind + 1
        fig.savefig('alliz' + save_suffix + 'png')

    # return z-origin (z) + z-displacement minus zshift (to account for non-centered disc)
    return z + zrange[ind_peak] - zshift


def label_disc_posterior(list_disc_z, list_disc_value, fname_hm, fname_data):
    """
    Function used to put label on the posterior tip for each disc using the previously found cooordinates.
    This is done using the maximum value in the RL direction of the network prediction (heatmap)
    for the found IS coordinates.


    :param list_disc_z: list of position alongside the IS axis
    :param list_disc_value: list of label value (e.g., [1,2,3,4]). The index of the value correspond to the index of the position in list_disc_z
    :param fname_hm: path to the heatmap output by the network
    :param fname_data: Path to sgementation data with same resolution as the heatmap (fname_hm)
    :return: None

    """
    im_hm = Image(fname_hm)
    image_out = Image(fname_data)
    nx, ny, nz = image_out.dim[0], image_out.dim[1], image_out.dim[2]
    data_disc = np.zeros([nx, ny, nz])
    AP_profile = np.sum(im_hm.data[:, :, :], axis=(0, 2))
    default = np.argmax(AP_profile)
    for iz in range(len(list_disc_z)):
        # some point are added to the list based on the template distance (might be out of bounds)
        if list_disc_z[iz] < nz:
            slice = im_hm.data[:, :, list_disc_z[iz] - 1]
            if np.any(slice) and list_disc_value[iz] > 2:
                pos = np.where(slice == np.max(slice))
                if len(pos[1]) > 1:
                    ap_pos = pos[1][0]
                else:
                    ap_pos = pos[1]
                if abs(ap_pos - default) < 20:
                    data_disc[int(np.round(nx / 2.0)), ap_pos, list_disc_z[iz] - 1] = list_disc_value[iz] + 1
                else:
                    data_disc[int(np.round(nx / 2)), default, list_disc_z[iz] - 1] = list_disc_value[iz] + 1
            else:
                # Since the image is supposedly straighten, we can assume that most of the disc are aligned
                # therefore if the heatmap missed one, we can just use the a default, aligned with the other
                data_disc[int(np.round(nx/2.0)), default, list_disc_z[iz] - 1] = list_disc_value[iz] + 1
    image_out.data = data_disc
    image_out.save('disc_posterior_tmp.nii.gz')
