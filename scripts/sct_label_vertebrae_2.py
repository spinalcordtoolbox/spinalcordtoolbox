#!/usr/bin/env python
#########################################################################################
#
# Detect vertebral levels from centerline.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Eugenie Ullmann, Karun Raju, Tanguy Duval, Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: add user input option (show sagittal slice)
# TODO: make better distance template

# from msct_base_classes import BaseScript
import sys
from os import chdir
from time import strftime
import numpy as np
from scipy.signal import argrelextrema, gaussian
from sct_utils import extract_fname, printv, run, generate_output_file, slash_at_the_end
from msct_parser import Parser
from msct_image import Image
from scipy.optimize import minimize
import scipy.optimize as spo
from scipy.interpolate import interp1d
import scipy.constants


# PARSER
# ==========================================================================================
def get_parser():
    # parser initialisation
    parser = Parser(__file__)
    parser.usage.set_description('''This program automatically detect the spinal cord in a MR image and output a centerline of the spinal cord.''')
    parser.add_option(name="-i",
                      type_value="file",
                      description="input image.",
                      mandatory=True,
                      example="t2.nii.gz")
    parser.add_option(name="-seg",
                      type_value="file",
                      description="Segmentation or centerline of the spinal cord.",
                      mandatory=True,
                      deprecated_by='-s',
                      example="t2_seg.nii.gz")
    parser.add_option(name="-s",
                      type_value="file",
                      description="Segmentation or centerline of the spinal cord.",
                      mandatory=True,
                      example="t2_seg.nii.gz")
    parser.add_option(name="-initz",
                      type_value=[[','], 'int'],
                      description='Initialize labeling by providing slice number (in superior-inferior direction!!) and disc value. Value corresponds to vertebral level above disc (e.g., for C3/C4 disc, value=3). Separate with ","',
                      mandatory=False,
                      example=['125,3'])
    parser.add_option(name="-initcenter",
                      type_value='int',
                      description='Initialize labeling by providing the disc value centered in the rostro-caudal direction. If the spine is curved, then consider the disc that projects onto the cord at the center of the z-FOV',
                      mandatory=False)
    parser.add_option(name='-o',
                      type_value='file_output',
                      description='Output file',
                      mandatory=False,
                      default_value='',
                      example='t2_seg_labeled.nii.gz')
    parser.add_option(name="-denoise",
                      type_value="multiple_choice",
                      description="Apply denoising filter to the data. Sometimes denoising is too aggressive, so use with care.",
                      mandatory=False,
                      default_value='0',
                      example=['0', '1'])
    parser.add_option(name="-laplacian",
                      type_value="multiple_choice",
                      description="Apply Laplacian filtering. More accuracy but could mistake disc depending on anatomy.",
                      mandatory=False,
                      default_value='0',
                      example=['0', '1'])
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="Remove temporary files.",
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="""Verbose. 0: nothing. 1: basic. 2: extended.""",
                      mandatory=False,
                      default_value='1',
                      example=['0', '1', '2'])
    parser.add_option(name="-h",
                      type_value=None,
                      description="display this help",
                      mandatory=False)
    return parser


# MAIN
# ==========================================================================================
def main(args=None):

    # initializations
    initz = ''
    initcenter = ''

    # # check user arguments
    if not args:
        args = sys.argv[1:]

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    fname_in = arguments["-i"]
    fname_seg = arguments['-s']
    # contrast = arguments['-t']
    if '-o' in arguments:
        fname_out = arguments["-o"]
    else:
        fname_out = ''
    if '-initz' in arguments:
        initz = arguments['-initz']
    if '-initcenter' in arguments:
        initcenter = arguments['-initcenter']
    verbose = int(arguments['-v'])
    remove_tmp_files = int(arguments['-r'])
    denoise = int(arguments['-denoise'])
    laplacian = int(arguments['-laplacian'])

    # fname_in = "/Users/auton/sct_example_data/t2/t2.nii.gz" #111t2_spc_1mm_p3, t2, 141t2_spc_1mm_p3, 131t2_spc_1mm_p3, 71t2_spc_1mm_p3.nii
    # fname_seg = "/Users/auton/sct_example_data/t2/t2_seg.nii.gz"
    # fname_out = "t2_seg_test.nii.gz"
    # initcenter = 7
    # verbose = 2
    # denoise = 0
    # laplacian = 0
    # remove_tmp_files = 0

    # create temporary folder
    printv('\nCreate temporary folder...', verbose)
    path_tmp = slash_at_the_end('tmp.'+strftime("%y%m%d%H%M%S"), 1)
    # path_tmp = slash_at_the_end('/Users/auton/code/spinalcordtoolbox/scripts/sct_example_data/tmp.'+strftime("%y%m%d%H%M%S"), 1)
    run('mkdir '+path_tmp, verbose)

    # Copying input data to tmp folder
    printv('\nCopying input data to tmp folder...', verbose)
    run('sct_convert -i '+fname_in+' -o '+path_tmp+'data.nii')
    run('sct_convert -i '+fname_seg+' -o '+path_tmp+'segmentation.nii.gz')

    # Go go temp folder
    # path_tmp = '/Users/julien/data/biospective/20151013_demo_spinalcordv2.1.b9/200_006_s2_T2/tmp.151013175622/'
    chdir(path_tmp)

    # create label to identify disc
    printv('\nCreate label to identify disc...', verbose)
    if initz:
        create_label_z('segmentation.nii.gz', initz[0], initz[1])  # create label located at z_center
    elif initcenter:
        # find z centered in FOV
        nii = Image('segmentation.nii.gz')
        nii.change_orientation('RPI')  # reorient to RPI
        nx, ny, nz, nt, px, py, pz, pt = nii.dim  # Get dimensions
        z_center = int(round(nz/2))  # get z_center
        create_label_z('segmentation.nii.gz', z_center, initcenter)  # create label located at z_center
    else:
        printv('\nERROR: You need to initialize the disc detection algorithm using one of these two options: -initz, -initcenter\n', 1, 'error')

    # Straighten spinal cord
    printv('\nStraighten spinal cord...', verbose)
    run('sct_straighten_spinalcord -i data.nii -s segmentation.nii.gz -r 0 -qc 0')
    # run('sct_straighten_spinalcord -i data.nii -s segmentation.nii.gz -r 0 -param all_labels=0,bspline_meshsize=3x3x5 -qc 0')  # here using all_labels=0 because of issue #610

    # Apply straightening to segmentation
    # N.B. Output is RPI
    printv('\nApply straightening to segmentation...', verbose)
    run('sct_apply_transfo -i segmentation.nii.gz -d data_straight.nii -w warp_curve2straight.nii.gz -o segmentation_straight.nii.gz -x linear')
    # Threshold segmentation to 0.5
    run('sct_maths -i segmentation_straight.nii.gz -thr 0.5 -o segmentation_straight.nii.gz')

    # Apply straightening to z-label
    printv('\nDilate z-label and apply straightening...', verbose)
    run('sct_apply_transfo -i labelz.nii.gz -d data_straight.nii -w warp_curve2straight.nii.gz -o labelz_straight.nii.gz -x nn')

    # get z value and disk value to initialize labeling
    printv('\nGet z and disc values from straight label...', verbose)
    init_disc = get_z_and_disc_values_from_label('labelz_straight.nii.gz')
    printv('.. '+str(init_disc), verbose)

    # denoise data
    if denoise:
        printv('\nDenoise data...', verbose)
        run('sct_maths -i data_straight.nii -denoise h=0.05 -o data_straight.nii')

    # apply laplacian filtering
    if laplacian:
        printv('\nApply Laplacian filter...', verbose)
        run('sct_maths -i data_straight.nii -laplace 1 -o data_straight.nii')

    # detect vertebral levels on straight spinal cord
    vertebral_detection('data_straight.nii', 'segmentation_straight.nii.gz', init_disc, verbose)

    # un-straighten labelled spinal cord
    printv('\nUn-straighten labeling...', verbose)
    run('sct_apply_transfo -i segmentation_straight_labeled.nii.gz -d segmentation.nii.gz -w warp_straight2curve.nii.gz -o segmentation_labeled.nii.gz -x nn')

    # Clean labeled segmentation
    printv('\nClean labeled segmentation (correct interpolation errors)...', verbose)
    clean_labeled_segmentation('segmentation_labeled.nii.gz', 'segmentation.nii.gz', 'segmentation_labeled.nii.gz')

    # Build fname_out
    if fname_out == '':
        path_seg, file_seg, ext_seg = extract_fname(fname_seg)
        fname_out = path_seg+file_seg+'_labeled'+ext_seg

    # come back to parent folder
    chdir('..')

    # Generate output files
    printv('\nGenerate output files...', verbose)
    generate_output_file(path_tmp+'segmentation_labeled.nii.gz', fname_out)

    # Remove temporary files
    if remove_tmp_files == 1:
        printv('\nRemove temporary files...', verbose)
        run('rm -rf '+path_tmp)

    # to view results
    printv('\nDone! To view results, type:', verbose)
    printv('fslview '+fname_in+' '+fname_out+' -l Random-Rainbow -t 0.5 &\n', verbose, 'info')

# Detect vertebral levels
# ==========================================================================================
def vertebral_detection(fname, fname_seg, init_disc, verbose):

    shift_AP = 17  # shift the centerline towards the spine (in mm).
    size_AP = 4  # window size in AP direction (=y) in mm
    size_RL = 7  # window size in RL direction (=x) in mm
    size_IS = 7  # window size in IS direction (=z) in mm
    searching_window_for_maximum = 5  # size used for finding local maxima
    thr_corr = 0.2  # disc correlation threshold. Below this value, use template distance.
    # gaussian_std_factor = 5  # the larger, the more weighting towards central value. This value is arbitrary-- should adjust based on large dataset
    fig_anat_straight = 1 # handle for figure
    fig_anat_straight_labeled = 1 # handle for figure
    fig_pattern = 2  # handle for figure
    fig_corr = 3  # handle for figure
    # define mean distance between adjacent discs: C1/C2 -> C2/C3, C2/C3 -> C4/C5, ..., L1/L2 -> L2/L3.
    mean_distance = np.array([18, 16, 17.0000, 16.0000, 15.1667, 15.3333, 15.8333,   18.1667,   18.6667,   18.6667,
    19.8333,   20.6667,   21.6667,   22.3333,   23.8333,   24.1667,   26.0000,   28.6667,   30.5000,   33.5000,
    33.0000,   31.3330])


    if verbose == 2:
        import matplotlib.pyplot as plt
        plt.ion()  # enables interactive mode

    # open anatomical volume
    img = Image(fname)
    data = img.data

    # smooth data
    from scipy.ndimage.filters import gaussian_filter
    data = gaussian_filter(data, [3, 1, 0], output=None, mode="reflect")

    # get dimension
    nx, ny, nz, nt, px, py, pz, pt = img.dim


    #==================================================
    # Compute intensity profile across vertebrae
    #==================================================

    shift_AP = shift_AP * py
    size_AP = size_AP * py
    size_RL = size_RL * px

    # define z: vector of indices along spine
    z = range(nz)
    # define xc and yc (centered in the field of view)
    xc = int(round(nx/2))  # direction RL
    yc = int(round(ny/2))  # direction AP

    # display stuff
    if verbose == 2:
        import matplotlib.pyplot as plt
        plt.ion()
        plt.matshow(np.mean(data[xc-size_RL:xc+size_RL, :, :], axis=0).transpose(), fignum=fig_anat_straight, cmap=plt.cm.gray, origin='lower')
        plt.title('Anatomical image')
        plt.autoscale(enable=False)  # to prevent autoscale of axis when displaying plot
        plt.figure(fig_anat_straight) , plt.scatter(yc+shift_AP, init_disc[0], c='y', s=50)  # display init disc
        #plt.text(yc+shift_AP+4, init_disc[0], 'init', verticalalignment='center', horizontalalignment='left', color='yellow', fontsize=15), plt.draw()
        plt.close()

    # FIND DISCS
    # ===========================================================================
    printv('\nDetect intervertebral discs...', verbose)
    # assign initial z and disc
    current_z = init_disc[0]
    current_disc = init_disc[1]

    # AT_test: manual initialization
    # current_z = 165
    # current_disc = 2

    # adjust to pix size
    mean_distance = mean_distance * pz
    mean_distance_real = np.zeros(len(mean_distance))

    # do local adjustment to be at the center of the disc
    printv('.. local adjustment to center disc', verbose)
    # AT_del: pattern seems to be unused
    # pattern = data[xc-size_RL:xc+size_RL+1, yc+shift_AP-size_AP:yc+shift_AP+size_AP+1, current_z-size_IS:current_z+size_IS+1]
    current_z = local_adjustment(xc, yc, current_z, current_disc, data, size_RL, shift_AP, size_IS, searching_window_for_maximum, verbose)
    # if verbose == 2:
    #     plt.figure(fig_anat_straight), plt.scatter(yc+shift_AP, current_z, c='g', s=50)
    #     plt.text(yc+shift_AP+4, current_z, str(current_disc)+'/'+str(current_disc+1), verticalalignment='center', horizontalalignment='left', color='green', fontsize=15)
    #     # plt.draw()

    # AT_del: used for later
    # append value to main list
    # list_disc_z = np.append(list_disc_z, current_z).astype(int)
    # list_disc_value = np.append(list_disc_value, current_disc).astype(int)

    # update initial value (used when switching disc search to inferior direction)
    init_disc[0] = current_z
    # AT_add: reference pattern
    pattern_ref = data[xc-size_RL:xc+size_RL+1, yc+shift_AP-size_AP:yc+shift_AP+size_AP+1, current_z-size_IS:current_z+size_IS+1]
    pattern_ref1d = pattern_ref.ravel()

    # AT_add: find distance from current_disc based on mean_distance
    mean_distance_from_init_disc = np.zeros(len(mean_distance)+1)
    for idistance in range(0, len(mean_distance_from_init_disc)):
        sum_distance_pos = 0
        sum_distance_neg = 0
        if idistance < current_disc:
            for i in range(current_disc-1, idistance-1, -1):
                sum_distance_pos = sum_distance_pos + mean_distance[i]
            mean_distance_from_init_disc[len(mean_distance_from_init_disc)-idistance-1] = int(round(sum_distance_pos))
        if idistance > current_disc:
            for i in range(current_disc,idistance):
                sum_distance_neg = sum_distance_neg - mean_distance[i]
                mean_distance_from_init_disc[len(mean_distance_from_init_disc)-idistance-1] = int(round(sum_distance_neg))
        if idistance == current_disc:
            mean_distance_from_init_disc[len(mean_distance_from_init_disc)-idistance-1] = 0

    # old algorithm (reversed)
    # mean_distance_from_init_disc = np.zeros(len(mean_distance)+1)
    # for idistance in range(0, len(mean_distance_from_init_disc)):
    #     sum_distance = 0
    #     if idistance < current_disc:
    #         for i in range(current_disc-1,idistance-1, -1):
    #             sum_distance = sum_distance - mean_distance[i]
    #         mean_distance_from_init_disc[idistance] = int(round(sum_distance))
    #     if idistance > current_disc:
    #         for i in range(current_disc, idistance):
    #             sum_distance = sum_distance + mean_distance[i]
    #         mean_distance_from_init_disc[idistance] = int(round(sum_distance))
    #     if idistance == current_disc:
    #         mean_distance_from_init_disc[idistance+current_disc] = 0

    # AT_add: adjust mean_distance_from_init_disc based on the image dimensions
    ind_inf = 0
    ind_sup = len(mean_distance_from_init_disc)-current_disc
    for ind_dist in range(0, len(mean_distance_from_init_disc)):
        if ind_dist<len(mean_distance_from_init_disc)-current_disc:
            if mean_distance_from_init_disc[ind_dist]+current_z<=0:
                ind_inf=ind_inf+1
            else:
                ind_inf=ind_inf
        else:
            if mean_distance_from_init_disc[ind_dist]+current_z>nz:
                ind_sup = ind_sup
            else:
                ind_sup = ind_sup+1
    mean_distance_from_init_disc_append = mean_distance_from_init_disc[ind_inf:ind_sup]

    correlation_profile = [get_correlation_profile(pattern_ref, pattern_ref1d, z, xc, yc, size_RL, shift_AP, size_AP, size_IS, data) for z in range(0, nz)]
    correlation_profile = np.nan_to_num(correlation_profile)


    mean_distance_from_z0 = mean_distance_from_init_disc_append + current_z
    z_corr_max = np.zeros(len(mean_distance_from_z0))
    for ind_dist in range(0, len(mean_distance_from_z0)):
        if ind_dist == 0:
            lowerlim = 0
            upperlim = int(round(0.5*(mean_distance_from_z0[ind_dist+1]-mean_distance_from_z0[ind_dist])+mean_distance_from_z0[ind_dist]))
        elif ind_dist == len(mean_distance_from_z0)-1:
            lowerlim = int(round(0.5*(mean_distance_from_z0[ind_dist]-mean_distance_from_z0[ind_dist-1])+mean_distance_from_z0[ind_dist-1]))
            upperlim = len(correlation_profile)-1
        else:
            lowerlim = int(round(0.5*(mean_distance_from_z0[ind_dist]-mean_distance_from_z0[ind_dist-1])+mean_distance_from_z0[ind_dist-1]))
            upperlim = int(round(0.5*(mean_distance_from_z0[ind_dist+1]-mean_distance_from_z0[ind_dist])+mean_distance_from_z0[ind_dist]))
        if lowerlim < 0:
            lowerlim = 0
        if upperlim > len(correlation_profile)-1:
            upperlim = len(correlation_profile)-1
        correlation_profile_window = correlation_profile[lowerlim:upperlim]
        z_corr_max[ind_dist] = np.argmax(correlation_profile_window)+lowerlim

    # Initial guess for optimization
    z_corr_max_real = z_corr_max
    search_z_corr_max_real = True
    while search_z_corr_max_real:
        optimization = minimize(get_z_corr_max_real, z_corr_max_real, args=(z_corr_max, mean_distance_from_z0, correlation_profile), method='nelder-mead')
        optimization_result = optimization.x.astype(int)
        test = abs(optimization_result-z_corr_max_real)
        if all(test < 2):
            search_z_corr_max_real = False
        else:
            # correction_factor = np.average(np.divide((optimization_result+1).astype(float), (z_corr_max_real+1).astype(float)))
            # mean_distance_from_z0_real = correction_factor * mean_distance_from_z0_real
            z_corr_max_real = optimization_result


    # AT_add: find correlation (not working)
    # z_adjustment = np.zeros_like(mean_distance_from_init_disc_append)
    # optimization_result = minimize(get_correlation, z_adjustment,
    #                                args=(mean_distance_from_init_disc_append, pattern_ref, pattern_ref1d, current_z, xc, yc, size_RL, shift_AP, size_AP, size_IS, data),
    #                                method='nelder-mead')
    # z_adjustment_real = (optimization_result.x).astype(int)


    # create list for z
    list_disc_z = np.zeros(len(mean_distance_from_init_disc_append))
    list_disc_z = list_disc_z + z_corr_max_real

    # create list for disc
    disc_inf = len(mean_distance_from_init_disc) - ind_sup
    disc_sup = len(mean_distance_from_init_disc) - (ind_inf + 1)
    list_disc_value = np.linspace(disc_sup, disc_inf, num=disc_sup-disc_inf+1)

    if verbose == 2:
        import matplotlib.pyplot as plt
        plt.ion()
        plt.matshow(np.mean(data[xc-size_RL:xc+size_RL, :, :], axis=0).transpose(), fignum=fig_anat_straight_labeled, cmap=plt.cm.gray, origin='lower')
        plt.title('Anatomical image with labels')
        plt.autoscale(enable=False)  # to prevent autoscale of axis when displaying plot
        plt.figure(fig_anat_straight_labeled), plt.scatter(np.full(len(list_disc_z), yc+shift_AP), list_disc_z, c='y', s=50)
        for i_text in range(0, len(list_disc_z)):
            plt.text(yc+shift_AP+4, list_disc_z[i_text], list_disc_value[i_text].astype(str), verticalalignment='center', horizontalalignment='left', color='yellow', fontsize=15), plt.draw()
        plt.figure(fig_anat_straight_labeled), plt.savefig('../fig_anat_straight_with_all_labels.png')


    # LABEL SEGMENTATION
    # open segmentation
    seg = Image(fname_seg)
    for iz in range(nz):
        # get index of the disk above iz
        ind_above_iz = np.nonzero((list_disc_z-iz).clip(0))[0]
        if not ind_above_iz.size:
            # if ind_above_iz is empty, attribute value 0
            # vertebral_level = np.min(labeled_peaks)
            vertebral_level = 0
        else:
            # ind_disk_above = np.where(peaks-iz > 0)[0][0]
            ind_disk_above = min(ind_above_iz)
            # assign vertebral level (add one because iz is BELOW the disk)
            vertebral_level = list_disc_value[ind_disk_above] + 1
            # print vertebral_level
        # get voxels in mask
        ind_nonzero = np.nonzero(seg.data[:, :, iz])
        seg.data[ind_nonzero[0], ind_nonzero[1], iz] = vertebral_level
        if verbose == 2:
            plt.figure(fig_anat_straight)
            plt.scatter(int(round(ny/2)), iz, c=vertebral_level, vmin=min(list_disc_value), vmax=max(list_disc_value), cmap='prism', marker='_', s=200)

    # write file
    seg.file_name += '_labeled'
    seg.save()

    # save figure
    if verbose == 2:
        plt.figure(fig_anat_straight), plt.savefig('../fig_anat_straight_with_labels.png')
        plt.close()


# Create label
# ==========================================================================================
def create_label_z(fname_seg, z, value):
    """
    Create a label at coordinates x_center, y_center, z
    :param fname_seg: segmentation
    :param z: int
    :return: fname_label
    """
    fname_label = 'labelz.nii.gz'
    nii = Image(fname_seg)
    orientation_origin = nii.change_orientation('RPI')  # change orientation to RPI
    nx, ny, nz, nt, px, py, pz, pt = nii.dim  # Get dimensions
    # find x and y coordinates of the centerline at z using center of mass
    from scipy.ndimage.measurements import center_of_mass
    x, y = center_of_mass(nii.data[:, :, z])
    x, y = int(round(x)), int(round(y))
    nii.data[:, :, :] = 0
    nii.data[x, y, z] = value
    # dilate label to prevent it from disappearing due to nearestneighbor interpolation
    from sct_maths import dilate
    nii.data = dilate(nii.data, 3)
    nii.setFileName(fname_label)
    nii.change_orientation(orientation_origin)  # put back in original orientation
    nii.save()
    return fname_label


# Get z and label value
# ==========================================================================================
def get_z_and_disc_values_from_label(fname_label):
    """
    Find z-value and label-value based on labeled image
    :param fname_label: image that contains label
    :return: [z_label, value_label] int list
    """
    nii = Image(fname_label)
    # get center of mass of label
    from scipy.ndimage.measurements import center_of_mass
    x_label, y_label, z_label = center_of_mass(nii.data)
    x_label, y_label, z_label = int(round(x_label)), int(round(y_label)), int(round(z_label))
    # get label value
    value_label = int(nii.data[x_label, y_label, z_label])
    return [z_label, value_label]


# Do local adjustment to be at the center of the current disc
# ==========================================================================================
def local_adjustment(xc, yc, current_z, current_disc, data, size_RL, shift_AP, size_IS, searching_window_for_maximum, verbose):
    """
    Do local adjustment to be at the center of the current disc, using cross-correlation of mirrored disc
    :param current_z: init current_z
    :return: adjusted_z: adjusted current_z
    """
    if verbose == 2:
        import matplotlib.pyplot as plt

    size_AP_mirror = 1
    searching_window = range(-9, 13)
    fig_local_adjustment = 4  # fig number
    thr_corr = 0.15  # arbitrary-- should adjust based on large dataset
    gaussian_std_factor = 3  # the larger, the more weighting towards central value. This value is arbitrary-- should adjust based on large dataset

    # Get pattern centered at current_z = init_disc[0]
    pattern = data[xc-size_RL:xc+size_RL+1, yc+shift_AP-size_AP_mirror:yc+shift_AP+size_AP_mirror+1, current_z-size_IS:current_z+size_IS+1]
    # if pattern is missing data (because close to the edge), do not perform correlation and return current_z
    if not pattern.shape == (int(round(size_RL*2+1)), int(round(size_AP_mirror*2+1)), int(round(size_IS*2+1))):
        printv('.... WARNING: Pattern is missing data (because close to the edge). Using initial current_z provided.', verbose)
        return current_z
    pattern1d = pattern.ravel()
    # compute cross-correlation with mirrored pattern
    I_corr = np.zeros((len(searching_window)))
    ind_I = 0
    for iz in searching_window:
        # get pattern shifted
        pattern_shift = data[xc-size_RL:xc+size_RL+1, yc+shift_AP-size_AP_mirror:yc+shift_AP+size_AP_mirror+1, current_z+iz-size_IS:current_z+iz+size_IS+1]
        # if pattern is missing data (because close to the edge), do not perform correlation and return current_z
        if not pattern_shift.shape == (int(round(size_RL*2+1)), int(round(size_AP_mirror*2+1)), int(round(size_IS*2+1))):
            printv('.... WARNING: Pattern is missing data (because close to the edge). Using initial current_z provided.', verbose)
            return current_z
        # make it 1d
        pattern1d_shift = pattern_shift.ravel()
        # mirror it
        pattern1d_shift_mirr = pattern1d_shift[::-1]
        # compute correlation
        I_corr[ind_I] = np.corrcoef(pattern1d_shift_mirr, pattern1d)[0, 1]
        ind_I = ind_I + 1
    # adjust correlation with Gaussian function centered at 'approx_distance_to_next_disc'
    gaussian_window = gaussian(len(searching_window), std=len(searching_window)/gaussian_std_factor)
    I_corr_adj = np.multiply(I_corr, gaussian_window)
    # display
    if verbose == 2:
        plt.figure(fig_local_adjustment), plt.plot(I_corr), plt.plot(I_corr_adj, 'k')
        plt.legend(['I_corr', 'I_corr_adj'])
        plt.title('Correlation of pattern with mirrored pattern.')
    # Find peak within local neighborhood
    ind_peak = argrelextrema(I_corr_adj, np.greater, order=searching_window_for_maximum)[0]
    if len(ind_peak) == 0:
        printv('.... WARNING: No peak found. Using initial current_z provided.', verbose)
        adjusted_z = current_z
    else:
        # keep peak with maximum correlation
        ind_peak = ind_peak[np.argmax(I_corr_adj[ind_peak])]
        printv('.... Peak found: '+str(ind_peak)+' (correlation = '+str(I_corr_adj[ind_peak])+')', verbose)
        # check if correlation is too low
        if I_corr_adj[ind_peak] < thr_corr:
            printv('.... WARNING: Correlation is too low. Using initial current_z provided.', verbose)
            adjusted_z = current_z
        else:
            adjusted_z = int(current_z + round(searching_window[ind_peak]/2)) + 1
            printv('.... Update init_z position to: '+str(adjusted_z), verbose)
    if verbose == 2:
        # display peak
        plt.figure(fig_local_adjustment), plt.plot(ind_peak, I_corr_adj[ind_peak], 'ro')
        # save and close figure
        plt.figure(fig_local_adjustment), plt.savefig('../fig_local_adjustment_disc'+str(current_disc)+'.png'), plt.close()
    return adjusted_z


def get_correlation_profile(pattern_ref, pattern_ref1d, z, xc, yc, size_RL, shift_AP, size_AP, size_IS, data):
    pattern = data[xc-size_RL:xc+size_RL+1,
                  yc+shift_AP-size_AP:yc+shift_AP+size_AP+1,
                  z-size_IS:z+size_IS+1]
    padding_size = pattern_ref.shape[2] - pattern.shape[2]
    pattern = np.pad(pattern, ((0, 0), (0, 0), (0, padding_size)), 'constant', constant_values=0)
    pattern1d = pattern.ravel()
    return np.corrcoef(pattern1d, pattern_ref1d)[0, 1]


def get_correlation_value(z, correlation_profile):
    return correlation_profile[int(z)]


def get_z_corr_max_real(z_corr_max_real, z_corr_max, mean_distance_from_z0, correlation_profile):
    z_corr_diff = abs(z_corr_max_real - z_corr_max)
    constraint = abs(np.diff(z_corr_max_real) - np.diff(mean_distance_from_z0))
    last_constraint = constraint[-1]
    constraint = np.append(constraint, last_constraint)
    factor = np.zeros_like(constraint)
    for i in range(0, len(constraint)):
        factor[i] = 1.3 - get_correlation_value(z_corr_max[i], correlation_profile)
    constraint = np.multiply(constraint, factor)
    return np.sum(z_corr_diff + constraint)


# AT_test: unused functions
# def get_correlation_sum(z_adjustment, mean_distance_from_init_disc_append, correlation_profile, current_z):
#     z_adjustment = (np.around(z_adjustment * 100.0)).astype(int)
#     index = (mean_distance_from_init_disc_append+current_z+z_adjustment).astype(int)
#     correlation_values = correlation_profile[index]
#     return -np.sum(correlation_values)
#
#
# # AT_test: correlation optimization for one disc
# def test_get_correlation(z_adjustment, mean_distance_from_init_disc_append, pattern_ref, pattern_ref1d, current_z, xc, yc, size_RL, shift_AP, size_AP, size_IS, data):
#     I_corr = 0
#     approx_distance_to_next_disc = mean_distance_from_init_disc_append
#     pattern = data[xc-size_RL:xc+size_RL+1,
#                   yc+shift_AP-size_AP:yc+shift_AP+size_AP+1,
#                   current_z+approx_distance_to_next_disc-size_IS+z_adjustment:current_z+approx_distance_to_next_disc+size_IS+1+z_adjustment]
#     padding_size = pattern_ref.shape[2] - pattern.shape[2]
#     pattern = np.pad(pattern, ((0, 0), (0, 0), (0, padding_size)), 'constant', constant_values=0)
#     pattern1d = pattern.ravel()
#     I_corr = np.corrcoef(pattern1d, pattern_ref1d)[0, 1]
#     # result = I_corr
#     result = I_corr-I_corr*(z_adjustment)*get_constraint_factor(z_adjustment)
#     return -result
#
#
# # AT_add: Get correlation
# def get_correlation(z_adjustment, mean_distance_from_init_disc_append, pattern_ref, pattern_ref1d, current_z, xc, yc, size_RL, shift_AP, size_AP, size_IS, data):
#     # pattern_all = np.zeros((len(mean_distance_from_init_disc), len(pattern_ref1d)))
#     z_adjustment = (np.around(z_adjustment * 100000.0)).astype(int)
#
#     sum_I_corr = 0
#     I_corr = np.zeros_like(z_adjustment).astype(float)
#     for iz in range(0, len(mean_distance_from_init_disc_append)):
#         approx_distance_to_next_disc = mean_distance_from_init_disc_append[iz]
#         pattern = data[xc-size_RL:xc+size_RL+1,
#                   yc+shift_AP-size_AP:yc+shift_AP+size_AP+1,
#                   current_z+approx_distance_to_next_disc-size_IS+z_adjustment[iz]:current_z+approx_distance_to_next_disc+size_IS+1+z_adjustment[iz]]
#         padding_size = pattern_ref.shape[2] - pattern.shape[2]
#         pattern = np.pad(pattern, ((0, 0), (0, 0), (0, padding_size)), 'constant', constant_values=0)
#         pattern1d = pattern.ravel()
#         I_corr[iz] = np.corrcoef(pattern1d, pattern_ref1d)[0, 1]
#         # gaussian_window = gaussian(len(I_corr[iz]), std=len(I_corr[iz])/3)
#         # I_corr_adj = np.multiply(I_corr[iz].transpose(), gaussian_window).transpose()
#         if np.isnan(I_corr[iz]) == False:
#             sum_I_corr = sum_I_corr + (I_corr[iz])
#         else:
#             sum_I_corr = sum_I_corr
#
#     print z_adjustment, sum_I_corr
#     return -sum_I_corr
#
#
# # AT_add: sigmoid function to determine constraint factor
# def get_constraint_factor(z_adjustment):
#     # factor_gaussian = 0.001
#     z_adjustment = abs(z_adjustment)
#     sigma = 10
#     factor = (np.exp(-np.power(z_adjustment, 2.) / (2 * np.power(sigma, 2.))))/(sigma*np.sqrt((2*np.pi)))
#     # factor = 1
#     # z_adjustment = abs(z_adjustment)
#     # alpha = 10
#     # factor_sigm = 0.05
#     # shift = mean_distance_append[iz]
#     # factor = -factor_sigm/(1+np.exp(-alpha*(z_adjustment-shift)))+factor_sigm
#     return factor

# Clean labeled segmentation
# ==========================================================================================
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
    #run('sct_maths -i segmentation_labeled.nii.gz -bin -o segmentation_labeled_bin.nii.gz')
    run('sct_maths -i '+fname_labeled_seg+' -mul '+fname_seg+' -o segmentation_labeled_mul.nii.gz')
    # add voxels in segmentation that are not in segmentation_labeled
    run('sct_maths -i '+fname_labeled_seg+' -dilate 2 -o segmentation_labeled_dilate.nii.gz')  # dilate labeled segmentation
    data_label_dilate = Image('segmentation_labeled_dilate.nii.gz').data
    run('sct_maths -i segmentation_labeled_mul.nii.gz -bin -o segmentation_labeled_mul_bin.nii.gz')
    data_label_bin = Image('segmentation_labeled_mul_bin.nii.gz').data
    data_seg = Image(fname_seg).data
    data_diff = data_seg - data_label_bin
    ind_nonzero = np.where(data_diff)
    im_label = Image('segmentation_labeled_mul.nii.gz')
    for i_vox in range(len(ind_nonzero[0])):
        # assign closest label value for this voxel
        ix, iy, iz = ind_nonzero[0][i_vox], ind_nonzero[1][i_vox], ind_nonzero[2][i_vox]
        im_label.data[ix, iy, iz] = data_label_dilate[ix, iy, iz]
    # save new label file (overwrite)
    im_label.setFileName(fname_labeled_seg_new)
    im_label.save()


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # call main function
    main()

