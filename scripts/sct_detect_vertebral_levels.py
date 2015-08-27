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

# check if needed Python libraries are already installed or not

# from msct_base_classes import BaseScript
import sys
from os import chdir
from time import strftime

import numpy as np

from sct_utils import extract_fname, printv, run, generate_output_file, slash_at_the_end
from msct_parser import Parser
from msct_image import Image


class Param:
    def __init__(self):
        self.verbose = '1'
        self.remove_tmp_files = '1'


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
                      example="t2_seg.nii.gz")
    parser.add_option(name="-t",
                      type_value="multiple_choice",
                      description="Image contrast: t2: cord dark / CSF bright ; t1: cord bright / CSF dark",
                      mandatory=True,
                      example=["t1", "t2"])
    parser.add_option(name="-initz",
                      type_value=[[','], 'int'],
                      description='Initialize labeling by providing slice number (in superior-inferior direction!!) and disc value. Value corresponds to vertebral level above disc (e.g., for C3/C4 disc, value=3). Separate with ","',
                      mandatory=False,
                      example=['125,3'])
    parser.add_option(name="-initcenter",
                      type_value='int',
                      description='Initialize labeling by providing the disc value centered in the rostro-caudal direction. If the spine is curved, then consider the disc that projects onto the cord at the center of the z-FOV',
                      mandatory=False)
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="Remove temporary files.",
                      mandatory=False,
                      default_value=param.remove_tmp_files,
                      example=['0', '1'])
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="""Verbose. 0: nothing. 1: basic. 2: extended.""",
                      mandatory=False,
                      default_value=param.verbose,
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

    # check user arguments
    if not args:
        args = sys.argv[1:]

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    fname_in = arguments["-i"]
    fname_seg = arguments['-seg']
    contrast = arguments['-t']
    if '-o' in arguments:
        fname_out = arguments["-o"]
    else:
        fname_out = ''
    if '-initz' in arguments:
        initz = arguments['-initz']
    if '-initcenter' in arguments:
        initcenter = arguments['-initcenter']
    param.verbose = int(arguments['-v'])
    param.remove_tmp_files = int(arguments['-r'])

    # create temporary folder
    printv('\nCreate temporary folder...', param.verbose)
    path_tmp = slash_at_the_end('tmp.'+strftime("%y%m%d%H%M%S"), 1)
    run('mkdir '+path_tmp, param.verbose)

    # Copying input data to tmp folder
    printv('\nCopying input data to tmp folder...', param.verbose)
    run('sct_convert -i '+fname_in+' -o '+path_tmp+'data.nii')
    run('sct_convert -i '+fname_seg+' -o '+path_tmp+'segmentation.nii.gz')

    # Go go temp folder
    path_tmp = '/Users/julien/data/sct_debug/tmp.150826170018/'
    chdir(path_tmp)

    # create label to identify disc
    printv('\nCreate label to identify disc...', param.verbose)
    if initz:
        create_label_z('segmentation.nii.gz', initz[0], initz[1])  # create label located at z_center
    elif initcenter:
        # find z centered in FOV
        nii = Image(fname_seg)
        nii.change_orientation('RPI')  # reorient to RPI
        nx, ny, nz, nt, px, py, pz, pt = nii.dim  # Get dimensions
        z_center = int(round(nz/2))  # get z_center
        create_label_z('segmentation.nii.gz', z_center, initcenter)  # create label located at z_center

    # TODO: denoise data

    # # Straighten spinal cord
    # printv('\nStraighten spinal cord...', param.verbose)
    # run('sct_straighten_spinalcord -i data.nii -c segmentation.nii.gz')

    # Apply straightening to segmentation
    # N.B. Output is RPI
    printv('\nApply straightening to segmentation...', param.verbose)
    run('sct_apply_transfo -i segmentation.nii.gz -d data_straight.nii -w warp_curve2straight.nii.gz -o segmentation_straight.nii.gz -x linear')
    # Threshold segmentation to 0.5
    run('sct_maths -i segmentation_straight.nii.gz -thr 0.5 -o segmentation_straight.nii.gz')

    # Apply straightening to z-label
    printv('\nDilate z-label and apply straightening...', param.verbose)
    run('sct_apply_transfo -i labelz.nii.gz -d data_straight.nii -w warp_curve2straight.nii.gz -o labelz_straight.nii.gz -x nn')

    # get z value and disk value to initialize labeling
    printv('\nGet z and disc values from straight label...', param.verbose)
    init_disc = get_z_and_disc_values_from_label('labelz_straight.nii.gz')
    printv('.. '+str(init_disc), param.verbose)

    # detect vertebral levels on straight spinal cord
    printv('\nDetect inter-vertebral discs and label vertebral levels...', param.verbose)
    vertebral_detection('data_straight.nii', 'segmentation_straight.nii.gz', contrast, init_disc)

    # un-straighten spinal cord
    printv('\nUn-straighten labeling...', param.verbose)
    run('sct_apply_transfo -i segmentation_straight_labeled.nii.gz -d segmentation.nii.gz -w warp_straight2curve.nii.gz -o segmentation_labeled.nii.gz -x nn')

    # clean labeled segmentation
    # TODO: (i) find missing voxels wrt. original segmentation, and attribute value closest to neighboring label and (ii) remove additional voxels.

    # Build fname_out
    if fname_out == '':
        path_seg, file_seg, ext_seg = extract_fname(fname_seg)
        fname_out = path_seg+file_seg+'_labeled'+ext_seg

    # come back to parent folder
    chdir('..')

    # Generate output files
    printv('\nGenerate output files...', param.verbose)
    generate_output_file(path_tmp+'segmentation_labeled.nii.gz', fname_out)

    # Remove temporary files
    if param.remove_tmp_files == 1:
        printv('\nRemove temporary files...', param.verbose)
        run('rm -rf '+path_tmp)

    # to view results
    printv('\nDone! To view results, type:', param.verbose)
    printv('fslview '+fname_in+' '+fname_out+' &\n', param.verbose, 'info')



# Detect vertebral levels
# ==========================================================================================
def vertebral_detection(fname, fname_seg, contrast, init_disc):

    from scipy.signal import argrelextrema

    shift_AP = 15  # shift the centerline towards the spine (in mm).
    size_AP = 3  # mean around the centerline in the anterior-posterior direction in mm
    size_RL = 5  # mean around the centerline in the right-left direction in mm
    verbose = param.verbose

    if verbose == 2:
        import matplotlib.pyplot as plt
        plt.ion()  # enables interactive mode

    # open anatomical volume
    img = Image(fname)
    data = img.data
    # orient to RPI
    # img.change_orientation()
    # get dimension
    nx, ny, nz, nt, px, py, pz, pt = img.dim

    # matshow(img.data[:, :, 100]), show()

    #==================================================
    # Compute intensity profile across vertebrae
    #==================================================

    shift_AP = shift_AP * py
    size_AP = size_AP * py
    size_RL = size_RL * px

    # define z: vector of indices along spine
    z = range(nz)
    # define xc and yc (centered in the field of view)
    xc = round(nx/2)  # direction RL
    yc = round(ny/2)  # direction AP

    # JULIEN <<<<<<<<<<<<
    size_RL = 7  # x
    size_AP = 5  # y
    size_IS = 5  # z

    # define mean distance from C1 disc to XX
    mean_distance = [12.1600, 20.8300, 18.0000, 16.0000, 15.1667, 15.3333, 15.8333,   18.1667,   18.6667,   18.6667,
    19.8333,   20.6667,   21.6667,   22.3333,   23.8333,   24.1667,   26.0000,   28.6667,   30.5000,   33.5000,
    33.0000,   31.3330]

    if verbose == 2:
        plt.figure(1), plt.imshow(np.mean(data[xc-7:xc+7, :, :], axis=0).transpose(), cmap=plt.cm.gray, origin='lower'), plt.title('Anatomical image'), plt.draw()
        # plt.matshow(np.flipud(np.mean(data[xc-7:xc+7, :, :], axis=0).transpose()), fignum=1, cmap=plt.cm.gray), plt.title('Anatomical image'), plt.draw()
        # display init disc
        plt.figure(1), plt.scatter(yc+shift_AP, init_disc[0], c='y'), plt.draw()


    # FIND DISCS
    # ===========================================================================
    list_disc_z = []
    list_disc_value = []
    # adjust to pix size
    mean_distance = mean_distance * pz
    # search peaks along z direction
    current_z = init_disc[0]
    current_disc = init_disc[1]
    # append initial position to main list
    list_disc_z = np.append(list_disc_z, current_z)
    list_disc_value = np.append(list_disc_value, current_disc)
    direction = 'superior'
    # TODO: define correcting factor based on distance to previous discs
    correcting_factor = 1
    # define mean distance to next disc
    approx_distance_to_next_disc = int(round(mean_distance[current_disc] * correcting_factor))
    # find_disc(data, current_z, current_disc, approx_distance_to_next_disc, direction)
    # loop until potential new peak is inside of FOV
    search_next_disc = True
    while search_next_disc:
        # Get pattern centered at z = current_z
        pattern = data[xc-size_RL:xc+size_RL+1, yc+shift_AP-size_AP:yc+shift_AP+size_AP+1, current_z-size_IS:current_z+size_IS+1]
        pattern1d = pattern.ravel()
        if verbose == 2:
            plt.matshow(np.flipud(np.mean(pattern[:, :, :], axis=0).transpose()), cmap=plt.cm.gray), plt.title('Pattern in sagittal averaged across R-L'), plt.draw()
        # compute correlation between pattern and data within range of z defined by template distance
        length_z_corr = approx_distance_to_next_disc * 2
        I_corr = np.zeros((length_z_corr, 1))
        ind_I = 0
        for iz in range(current_z, current_z+length_z_corr):
            data_chunk1d = data[xc-size_RL:xc+size_RL+1, yc+shift_AP-size_AP:yc+shift_AP+size_AP+1, iz-size_IS:iz+size_IS+1].ravel()
            # in case data_chunk1d is cropped (because beginning/end of  data), crop pattern
            if len(data_chunk1d) < len(pattern1d):
                crop_size = len(pattern1d) - len(data_chunk1d)
                if direction == 'superior':
                    # if direction is superior, crop end of pattern
                    pattern1d = pattern1d[:-crop_size]
                elif direction == 'inferior':
                    # if direction is inferior, crop beginning of pattern
                    pattern1d = pattern1d[crop_size:]
            I_corr[ind_I] = np.corrcoef(data_chunk1d, pattern1d)[0, 1]
            ind_I = ind_I + 1

        if verbose == 2:
            plt.figure(), plt.plot(I_corr), plt.title('Correlation of pattern with data.'), plt.draw()

        # TODO: crop beginning of signal to remove correlation = 1 from pattern with data
        # Find peak within local neighborhood defined by mean distance template
        # TODO: adjust probability to be in the middle of I_corr (account for mean dist template)
        peaks = argrelextrema(I_corr, np.greater, order=10)[0]
        nb_peaks = len(peaks)
        printv('.. Peaks found: '+str(peaks)+' with correlations: '+str(I_corr[peaks]), verbose)
        if len(peaks) > 1:
            # retain the peak with maximum correlation
            peaks = peaks[np.argmax(I_corr[peaks])]
            printv('.. WARNING: More than one peak found. Keeping: '+str(peaks), verbose)
        if verbose == 2:
            plt.plot(peaks, I_corr[peaks], 'ro'), plt.draw()
        # assign new z_start and disc value
        # if direction is superior: sign = -1, if direction is inferior: sign = +1
        current_z = current_z + int(peaks)
        if direction == 'superior':
            current_disc = current_disc - 1
        elif direction == 'inferior':
            current_disc = current_disc + 1

        # append to main list
        list_disc_z = np.append(list_disc_z, current_z)
        list_disc_value = np.append(list_disc_value, current_disc)
        # define mean distance to next disc
        # TODO: define correcting factor
        approx_distance_to_next_disc = int(round(mean_distance[current_disc] * correcting_factor))
        # display
        if verbose == 2:
            plt.figure(1), plt.scatter(yc+shift_AP, current_z, c='r'), plt.draw()
        # if current_z is larger than searching zone, switch direction (and start from initial z)
        if current_z + approx_distance_to_next_disc >= nz:
            direction = 'inferior'
            current_z = init_disc[0]
            current_disc = init_disc[1]
        # if current_z is lower than searching zone, stop searching
        if current_z - approx_distance_to_next_disc <= 0:
            search_next_disc = False


    # >>>>>>>>>>>>>>>>>>>>>

    # I = np.zeros((nz, 1))
    # # data_masked = img.data
    # # data = img.data
    # for iz in range(nz):
    #     vox_in_spine = np.mgrid[xc-size_RL:xc+size_RL+1, yc+shift_AP-size_AP:yc+shift_AP+size_AP+1]
    #     # average intensity within box in the spine (shifted from spinal cord)
    #     I[iz] = np.mean(img.data[vox_in_spine[0, :, :].ravel().astype(int),
    #                              vox_in_spine[1, :, :].ravel().astype(int),
    #                              iz])
    #     # # just for visualization:
    #     # data_masked[vox_in_spine[0, :, :].ravel().astype(int),
    #     #             vox_in_spine[1, :, :].ravel().astype(int),
    #     #             iz] = 0
    #
    # # Display mask
    # if verbose == 2:
    #     plt.matshow(np.flipud(np.mean(img.data[xc-3:xc+3, :, :], axis=0).transpose()), cmap=plt.cm.gray)
    #     #plt.matshow(np.flipud(data[xc, :, :].transpose()), cmap=plt.cm.gray)
    #     plt.title('Anatomical image')
    #     plt.draw()
    #     # plt.matshow(np.flipud(data_masked[xc, :, :].transpose()), cmap=plt.cm.gray)
    #     # plt.title('Anatomical image with mask')
    #     # plt.draw()
    #
    # # display intensity along spine
    # if verbose == 2:
    #     plt.figure()
    #     plt.plot(I)
    #     plt.title('Averaged intensity within spine. x=0: most caudal.')
    #     plt.draw()
    #
    # # find local extrema
    # from scipy.signal import argrelextrema
    # peaks = argrelextrema(I, np.greater, order=10)[0]
    # nb_peaks = len(peaks)
    # printv('.. Number of peaks found: '+str(nb_peaks), verbose)
    #
    # if verbose == 2:
    #     plt.figure()
    #     plt.plot(I)
    #     plt.plot(peaks, I[peaks], 'ro')
    #     plt.draw()
    #
    # # LABEL PEAKS
    # labeled_peaks = np.array(range(nb_peaks, 0, -1)).astype(int)
    # # find peak index closest to user input
    # peak_ind_closest = np.argmin(abs(peaks-init_disc[0]))
    # # build vector of peak labels
    # # labeled_peaks = np.array(range(nb_peaks))
    # # add the difference between "peak_ind_closest" and the init_disc value
    # labeled_peaks = init_disc[1] - labeled_peaks[peak_ind_closest] + labeled_peaks
    #
    # # REMOVE WRONG LABELS (ASSUMING NO PEAK IS VISIBLE ABOVE C2/C3 DISK)
    # ind_true_labels = np.where(labeled_peaks>1)[0]
    # peaks = peaks[ind_true_labels]
    # labeled_peaks = labeled_peaks[ind_true_labels]
    #
    # # ADD C1 label (ASSUMING DISTANCE FROM THE ADULT TEMPLATE)
    # distance_c1_c2 = 20.8300/pz  # in mm
    # # check if C2 disk is there
    # if np.min(labeled_peaks) == 2:
    #     printv('.. C2 disk is present. Adding C2 vertebrae based on template...')
    #     peaks = np.append(peaks, (np.max(peaks) + distance_c1_c2).astype(int))
    #     labeled_peaks = np.append(labeled_peaks, 1)
    # printv('.. Labeled peaks: '+str(labeled_peaks[:-1]), verbose)

    # LABEL SEGMENTATION
    # open segmentation
    seg = Image(fname_seg)
    for iz in range(nz):
        # get value of the disk above iz
        ind_above_iz = np.nonzero((peaks-iz).clip(0))[0]
        if not ind_above_iz.size:
            # if ind_above_iz is empty, attribute value 0
            # vertebral_level = np.min(labeled_peaks)
            vertebral_level = 0
        else:
            # ind_disk_above = np.where(peaks-iz > 0)[0][0]
            ind_disk_above = min(ind_above_iz)
            # assign vertebral level (remove one because iz is BELOW the disk)
            vertebral_level = labeled_peaks[ind_disk_above] - 1
            # print vertebral_level
        # get voxels in mask
        ind_nonzero = np.nonzero(seg.data[:, :, iz])
        seg.data[ind_nonzero[0], ind_nonzero[1], iz] = vertebral_level

    # WRITE LABELED SEGMENTATION
    seg.file_name += '_labeled'
    seg.save()


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
    nii.data = dilate(nii.data, 3) * value  # multiplies by value because output of dilation is binary
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



# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # # initialize parameters
    param = Param()
    # call main function
    main()
