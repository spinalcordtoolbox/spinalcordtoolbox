#!/usr/bin/env python
# -*- coding: utf-8
# Core functions dealing with vertebral labeling


# TODO: remove i/o as much as possible

import os
import numpy as np
import scipy.ndimage.measurements
from scipy.ndimage.filters import gaussian_filter

import sct_utils as sct
from sct_maths import mutual_information

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.metadata import get_file_label
from spinalcordtoolbox.math import dilate


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


def vertebral_detection(fname, fname_seg, contrast, param, init_disc, verbose=1, path_template='', path_output='../',
                        scale_dist=1.):
    """
    Find intervertebral discs in straightened image using template matching
    :param fname: file name of straigthened spinal cord
    :param fname_seg: file name of straigthened spinal cord segmentation
    :param contrast: t1 or t2
    :param param:  advanced parameters
    :param init_disc:
    :param verbose:
    :param path_template:
    :param path_output: output path for verbose=2 pictures
    :param scale_dist: float: Scaling factor to adjust average distance between two adjacent intervertebral discs
    :return:
    """
    sct.printv('\nLook for template...', verbose)
    sct.printv('Path template: ' + path_template, verbose)

    # adjust file names if MNI-Poly-AMU template is used (by default: PAM50)
    fname_level = get_file_label(os.path.join(path_template, 'template'), id_label=7, output='filewithpath')  # label = spinal cord mask with discrete vertebral levels
    id_label_dct = {'T1': 0, 'T2': 1, 'T2S': 2}
    fname_template = get_file_label(os.path.join(path_template, 'template'), id_label=id_label_dct[contrast.upper()], output='filewithpath')  # label = *-weighted template

    # Open template and vertebral levels
    sct.printv('\nOpen template and vertebral levels...', verbose)
    data_template = Image(fname_template).data
    data_disc_template = Image(fname_level).data

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
    centerline_level = data_disc_template[xct, yct, :]
    # attribute value to each disc. Starts from max level, then decrease.
    min_level = centerline_level[centerline_level.nonzero()].min()
    max_level = centerline_level[centerline_level.nonzero()].max()
    list_disc_value_template = list(range(min_level, max_level))
    # add disc above top one
    list_disc_value_template.insert(int(0), min_level - 1)
    sct.printv('\nDisc values from template: ' + str(list_disc_value_template), verbose)
    # get diff to find transitions (i.e., discs)
    diff_centerline_level = np.diff(centerline_level)
    # get disc z-values
    list_disc_z_template = diff_centerline_level.nonzero()[0].tolist()
    list_disc_z_template.reverse()
    sct.printv('Z-values for each disc: ' + str(list_disc_z_template), verbose)
    list_distance_template = (
        np.diff(list_disc_z_template) * (-1)).tolist()  # multiplies by -1 to get positive distances
    # Update distance with scaling factor
    list_distance_template = [i * scale_dist for i in list_distance_template]
    sct.printv('Distances between discs (in voxel): ' + str(list_distance_template), verbose)

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
    sct.printv('\nDetect intervertebral discs...', verbose)
    # assign initial z and disc
    current_z = init_disc[0]
    current_disc = init_disc[1]
    # create list for z and disc
    list_disc_z = []
    list_disc_value = []
    zrange = list(range(-10, 10))
    direction = 'superior'
    search_next_disc = True
    while search_next_disc:
        sct.printv('Current disc: ' + str(current_disc) + ' (z=' + str(current_z) + '). Direction: ' + direction, verbose)
        try:
            # get z corresponding to current disc on template
            current_z_template = list_disc_z_template[current_disc]
        except:
            # in case reached the bottom (see issue #849)
            sct.printv('WARNING: Reached the bottom of the template. Stop searching.', verbose, 'warning')
            break
        # find next disc
        # N.B. Do not search for C1/C2 disc (because poorly visible), use template distance instead
        if current_disc != 1:
            current_z = compute_corr_3d(data, data_template, x=xc, xshift=0, xsize=param.size_RL,
                                        y=yc, yshift=param.shift_AP, ysize=param.size_AP,
                                        z=current_z, zshift=0, zsize=param.size_IS,
                                        xtarget=xct, ytarget=yct, ztarget=current_z_template,
                                        zrange=zrange, verbose=verbose, save_suffix='_disc' + str(current_disc), gaussian_std=999, path_output=path_output)

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
            list_subject_to_template_distance = [float(list_distance_current[i]) / list_distance_template_identified[i] for i in range(len(list_distance_current))]
            # average across identified discs to obtain an average correcting factor
            correcting_factor = np.mean(list_subject_to_template_distance)
            sct.printv('.. correcting factor: ' + str(correcting_factor), verbose)
        else:
            correcting_factor = 1
        # update list_distance specific for the subject
        list_distance = [int(np.round(list_distance_template[i] * correcting_factor)) for i in range(len(list_distance_template))]

        # assign new current_z and disc value
        if direction == 'superior':
            try:
                approx_distance_to_next_disc = list_distance[list_disc_value_template.index(current_disc - 1)]
            except ValueError:
                sct.printv('WARNING: Disc value not included in template. Using previously-calculated distance: ' + str(approx_distance_to_next_disc))
            # assign new current_z and disc value
            current_z = current_z + approx_distance_to_next_disc
            current_disc = current_disc - 1
        elif direction == 'inferior':
            try:
                approx_distance_to_next_disc = list_distance[list_disc_value_template.index(current_disc)]
            except:
                sct.printv('WARNING: Disc value not included in template. Using previously-calculated distance: ' + str(approx_distance_to_next_disc))
            # assign new current_z and disc value
            current_z = current_z - approx_distance_to_next_disc
            current_disc = current_disc + 1

        # if current_z is larger than searching zone, switch direction (and start from initial z minus approximate
        # distance from updated template distance)
        if current_z >= nz or current_disc == 0:
            sct.printv('.. Switching to inferior direction.', verbose)
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
    sct.printv('Adding top disc based on adjusted template distance: #' + str(upper_disc - 1), verbose)
    approx_distance_to_next_disc = list_distance[list_disc_value_template.index(upper_disc - 1)]
    next_z = max(list_disc_z) + approx_distance_to_next_disc
    sct.printv('.. approximate distance: ' + str(approx_distance_to_next_disc), verbose)
    # make sure next disc does not go beyond FOV in superior direction
    if next_z > nz:
        list_disc_z.insert(0, nz)
    else:
        list_disc_z.insert(0, next_z)
    # assign disc value
    list_disc_value.insert(0, upper_disc - 1)

    # Label segmentation
    label_segmentation(fname_seg, list_disc_z, list_disc_value, verbose=verbose)


def center_of_mass(x):
    """
    :return: array center of mass
    """
    if (x == 0).all():
        raise ValueError("Array has no mass")
    return scipy.ndimage.measurements.center_of_mass(x)


def create_label_z(fname_seg, z, value, fname_labelz='labelz.nii.gz'):
    """
    Create a label at coordinates x_center, y_center, z
    :param fname_seg: segmentation
    :param z: int
    :param fname_labelz: string file name of output label
    :return: fname_labelz
    """
    nii = Image(fname_seg)
    orientation_origin = nii.orientation
    nii = nii.change_orientation("RPI")
    nx, ny, nz, nt, px, py, pz, pt = nii.dim  # Get dimensions
    # find x and y coordinates of the centerline at z using center of mass
    x, y = center_of_mass(np.array(nii.data[:, :, z]))
    x, y = int(np.round(x)), int(np.round(y))
    nii.data[:, :, :] = 0
    nii.data[x, y, z] = value
    # dilate label to prevent it from disappearing due to nearestneighbor interpolation
    nii.data = dilate(nii.data, 3, 'ball')
    nii.change_orientation(orientation_origin)  # put back in original orientation
    nii.save(fname_labelz)
    return fname_labelz


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


def clean_labeled_segmentation(fname_labeled_seg, fname_seg, fname_labeled_seg_new):
    """
    Clean labeled segmentation by:
      (i)  removing voxels in segmentation_labeled that are not in segmentation and
      (ii) adding voxels in segmentation that are not in segmentation_labeled
    :param fname_labeled_seg:
    :param fname_seg:
    :param fname_labeled_seg_new: output
    :return: none
    """
    # remove voxels in segmentation_labeled that are not in segmentation
    img_labeled_seg = Image(fname_labeled_seg)
    img_seg = Image(fname_seg)
    data_labeled_seg_mul = img_labeled_seg.data * img_seg.data
    # dilate to add voxels in segmentation that are not in segmentation_labeled
    data_labeled_seg_dil = dilate(img_labeled_seg.data, 2, 'ball')
    data_labeled_seg_mul_bin = data_labeled_seg_mul > 0
    data_diff = img_seg.data - data_labeled_seg_mul_bin
    ind_nonzero = np.where(data_diff)
    img_labeled_seg_corr = img_labeled_seg.copy()
    img_labeled_seg_corr.data = data_labeled_seg_mul
    for i_vox in range(len(ind_nonzero[0])):
        # assign closest label value for this voxel
        ix, iy, iz = ind_nonzero[0][i_vox], ind_nonzero[1][i_vox], ind_nonzero[2][i_vox]
        img_labeled_seg_corr.data[ix, iy, iz] = data_labeled_seg_dil[ix, iy, iz]
    # save new label file (overwrite)
    img_labeled_seg_corr.absolutepath = fname_labeled_seg_new
    img_labeled_seg_corr.save()


def compute_corr_3d(src, target, x, xshift, xsize, y, yshift, ysize, z, zshift, zsize, xtarget, ytarget, ztarget, zrange, verbose, save_suffix, gaussian_std, path_output):
    """
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
    # get dimensions from src
    nx, ny, nz = src.shape
    # Get pattern from template
    pattern = target[xtarget - xsize: xtarget + xsize + 1,
                     ytarget + yshift - ysize: ytarget + yshift + ysize + 1,
                     ztarget + zshift - zsize: ztarget + zshift + zsize + 1]
    pattern1d = pattern.ravel()
    # initializations
    I_corr = np.zeros(len(zrange))
    allzeros = 0
    # current_z = 0
    ind_I = 0
    # loop across range of z defined by src
    for iz in zrange:
        # if pattern extends towards the top part of the image, then crop and pad with zeros
        if z + iz + zsize + 1 > nz:
            # sct.printv('iz='+str(iz)+': padding on top')
            padding_size = z + iz + zsize + 1 - nz
            data_chunk3d = src[x - xsize: x + xsize + 1,
                               y + yshift - ysize: y + yshift + ysize + 1,
                               z + iz - zsize: z + iz + zsize + 1 - padding_size]
            data_chunk3d = np.pad(data_chunk3d, ((0, 0), (0, 0), (0, padding_size)), 'constant',
                                  constant_values=0)
        # if pattern extends towards bottom part of the image, then crop and pad with zeros
        elif z + iz - zsize < 0:
            # sct.printv('iz='+str(iz)+': padding at bottom')
            padding_size = abs(iz - zsize)
            data_chunk3d = src[x - xsize: x + xsize + 1,
                               y + yshift - ysize: y + yshift + ysize + 1,
                               z + iz - zsize + padding_size: z + iz + zsize + 1]
            data_chunk3d = np.pad(data_chunk3d, ((0, 0), (0, 0), (padding_size, 0)), 'constant',
                                  constant_values=0)
        else:
            data_chunk3d = src[x - xsize: x + xsize + 1,
                               y + yshift - ysize: y + yshift + ysize + 1,
                               z + iz - zsize: z + iz + zsize + 1]

        # convert subject pattern to 1d
        data_chunk1d = data_chunk3d.ravel()
        # check if data_chunk1d contains at least one non-zero value
        if (data_chunk1d.size == pattern1d.size) and np.any(data_chunk1d):
            I_corr[ind_I] = mutual_information(data_chunk1d, pattern1d, nbins=16, normalized=False)
        else:
            allzeros = 1
        ind_I = ind_I + 1
    # ind_y = ind_y + 1
    if allzeros:
        sct.printv('.. WARNING: Data contained zero. We probably hit the edge of the image.', verbose)

    # adjust correlation with Gaussian function centered at the right edge of the curve (most rostral point of FOV)
    from scipy.signal import gaussian
    gaussian_window = gaussian(len(I_corr) * 2, std=len(I_corr) * gaussian_std)
    I_corr_gauss = np.multiply(I_corr, gaussian_window[0:len(I_corr)])

    # Find global maximum
    if np.any(I_corr_gauss):
        # if I_corr contains at least a non-zero value
        ind_peak = [i for i in range(len(I_corr_gauss)) if I_corr_gauss[i] == max(I_corr_gauss)][0]  # index of max along z
        sct.printv('.. Peak found: z=' + str(zrange[ind_peak]) + ' (correlation = ' + str(I_corr_gauss[ind_peak]) + ')', verbose)
        # check if correlation is high enough
        if I_corr_gauss[ind_peak] < thr_corr:
            sct.printv('.. WARNING: Correlation is too low. Using adjusted template distance.', verbose)
            ind_peak = zrange.index(0)  # approx_distance_to_next_disc
    else:
        # if I_corr contains only zeros
        sct.printv('.. WARNING: Correlation vector only contains zeros. Using adjusted template distance.', verbose)
        ind_peak = zrange.index(0)  # approx_distance_to_next_disc

    # display patterns and correlation
    if verbose == 2:
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        fig = Figure(figsize=(15, 7))
        FigureCanvas(fig)
        # display template pattern
        ax = fig.add_subplot(131)
        ax.imshow(np.flipud(np.mean(pattern[:, :, :], axis=0).transpose()), origin='upper', cmap='gray',
                  interpolation='none')
        ax.set_title('Template pattern')
        # display subject pattern at best z
        ax = fig.add_subplot(132)
        iz = zrange[ind_peak]
        data_chunk3d = src[x - xsize: x + xsize + 1,
                           y + yshift - ysize: y + yshift + ysize + 1,
                           z + iz - zsize: z + iz + zsize + 1]
        ax.imshow(np.flipud(np.mean(data_chunk3d[:, :, :], axis=0).transpose()), origin='upper', cmap='gray',
                  clim=[0, 800], interpolation='none')
        ax.set_title('Subject at iz=' + str(iz))
        # display correlation curve
        ax = fig.add_subplot(133)
        ax.plot(zrange, I_corr)
        ax.plot(zrange, I_corr_gauss, 'black', linestyle='dashed')
        ax.legend(['I_corr', 'I_corr_gauss'])
        ax.set_title('Mutual Info, gaussian_std=' + str(gaussian_std))
        ax.plot(zrange[ind_peak], I_corr_gauss[ind_peak], 'ro')
        ax.axvline(x=zrange.index(0), linewidth=1, color='black', linestyle='dashed')
        ax.axhline(y=thr_corr, linewidth=1, color='r', linestyle='dashed')
        ax.grid()
        # save figure
        fig.savefig('fig_pattern' + save_suffix + '.png')

    # return z-origin (z) + z-displacement minus zshift (to account for non-centered disc)
    return z + zrange[ind_peak] - zshift


def label_segmentation(fname_seg, list_disc_z, list_disc_value, verbose=1):
    """
    Label segmentation image
    :param fname_seg: fname of the segmentation, no orientation expected
    :param list_disc_z: list of z that correspond to a disc
    :param list_disc_value: list of associated disc values
    :param verbose:
    :return:
    """

    # open segmentation
    seg = Image(fname_seg)
    init_orientation = seg.orientation
    seg.change_orientation("RPI")

    dim = seg.dim
    ny = dim[1]
    nz = dim[2]
    # loop across z
    for iz in range(nz):
        # get index of the disc right above iz
        try:
            ind_above_iz = max([i for i in range(len(list_disc_z)) if list_disc_z[i] > iz])
        except ValueError:
            # if ind_above_iz is empty, attribute value 0
            vertebral_level = 0
        else:
            # assign vertebral level (add one because iz is BELOW the disk)
            vertebral_level = list_disc_value[ind_above_iz] + 1
            # sct.printv(vertebral_level)
        # get voxels in mask
        ind_nonzero = np.nonzero(seg.data[:, :, iz])
        seg.data[ind_nonzero[0], ind_nonzero[1], iz] = vertebral_level
        # if verbose == 2:
        #     # move to OO. No time to finish... (JCA)
        #     from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        #     from matplotlib.figure import Figure
        #     fig = Figure()
        #     FigureCanvas(fig)
        #     ax = fig.add_subplot(111)
        #     ax.scatter(int(np.round(ny / 2)), iz, c=vertebral_level, vmin=min(list_disc_value),
        #                vmax=max(list_disc_value), cmap='prism', marker='_', s=200)
        #
        #     # TODO: the thing below crashes with the py3k move. Fix it when i have time...
        #     import matplotlib
        #     matplotlib.use('Agg')
        #     import matplotlib.pyplot as plt
        #     plt.figure(50)
        #     plt.scatter(int(np.round(ny / 2)), iz, c=vertebral_level, vmin=min(list_disc_value), vmax=max(list_disc_value), cmap='prism', marker='_', s=200)

    # write file
    seg.change_orientation(init_orientation).save(sct.add_suffix(fname_seg, '_labeled'))


def label_discs(fname_seg_labeled, verbose=1):
    """
    Label discs from labeled_segmentation. The convention is C2/C3-->3, C3/C4-->4, etc.
    :param fname_seg_labeld: fname of the labeled segmentation
    :param verbose:
    :return:
    """
    # open labeled segmentation
    im_seg_labeled = Image(fname_seg_labeled)
    orientation_native = im_seg_labeled.orientation
    im_seg_labeled.change_orientation("RPI")
    nx, ny, nz = im_seg_labeled.dim[0], im_seg_labeled.dim[1], im_seg_labeled.dim[2]
    data_disc = np.zeros([nx, ny, nz])
    vertebral_level_previous = np.max(im_seg_labeled.data)  # initialize with the max label value
    # loop across z in the superior direction (i.e. starts with the bottom slice), and each time the i/i+1 interface
    # between two levels is found, create a label at the center of the cord with the value corresponding to the
    # vertebral level below the point. E.g., at interface C3/C4, the value would be 4.
    for iz in range(nz):
        # get 2d slice
        slice = im_seg_labeled.data[:, :, iz]
        # check if at least one voxel is non-zero
        if np.any(slice):
            slice_one = np.copy(slice)
            # set all non-zero values to 1 (to compute center of mass)
            # Note: the reason we do this is because if one slice includes part of vertebral level i and i+1, the
            # center of mass will be shifted towards the i+1 level.We don't want that here (i.e. the goal is to be at
            # the center of the cord)
            slice_one[slice.nonzero()] = 1
            # compute center of mass
            cx, cy = [int(x) for x in np.round(center_of_mass(slice_one)).tolist()]
            # retrieve vertebral level
            vertebral_level = slice[cx, cy]
            # if smaller than previous level, then labeled as a disc
            if vertebral_level < vertebral_level_previous:
                # label disc
                data_disc[cx, cy, iz] = vertebral_level + 1
            # update variable
            vertebral_level_previous = vertebral_level
    # save disc labeled file
    im_seg_labeled.data = data_disc
    im_seg_labeled.change_orientation(orientation_native).save(sct.add_suffix(fname_seg_labeled, '_disc'))
