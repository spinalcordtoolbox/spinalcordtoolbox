#!/usr/bin/env python

import shutil
import os
import sct_utils as sct
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

import matplotlib.cm as cmx
import matplotlib.colors as colors

import numpy
def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also: 

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = numpy.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    s = x
    # print(len(s))
    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')

    y = numpy.convolve(w / w.sum(), s, mode='same')
    return y

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

labels_regions = {'PMJ': 50, 'PMG': 49,
                  'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5, 'C6': 6, 'C7': 7,
                  'T1': 8, 'T2': 9, 'T3': 10, 'T4': 11, 'T5': 12, 'T6': 13, 'T7': 14, 'T8': 15, 'T9': 16, 'T10': 17,
                  'T11': 18, 'T12': 19,
                  'L1': 20, 'L2': 21, 'L3': 22, 'L4': 23, 'L5': 24,
                  'S1': 25, 'S2': 26, 'S3': 27, 'S4': 28, 'S5': 29,
                  'Co': 30}

regions_labels = {'50': 'PMJ', '49': 'PMG',
                  '1': 'C1', '2': 'C2', '3': 'C3', '4': 'C4', '5': 'C5', '6': 'C6', '7': 'C7',
                  '8': 'T1', '9': 'T2', '10': 'T3', '11': 'T4', '12': 'T5', '13': 'T6', '14': 'T7', '15': 'T8',
                  '16': 'T9', '17': 'T10', '18': 'T11', '19': 'T12',
                  '20': 'L1', '21': 'L2', '22': 'L3', '23': 'L4', '24': 'L5',
                  '25': 'S1', '26': 'S2', '27': 'S3', '28': 'S4', '29': 'S5',
                  '30': 'Co'}
list_labels = [50, 49, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
               27, 28, 29, 30]

path_data_old = '/Volumes/data_processing/bdeleener/template/template_preprocessing_final/subjects/'
path_data_new = '/Users/benjamindeleener/data/PAM50_2017/'

"""
Deux autres sujets possibles pour remplacement:
AP et TT

Potentiellement a retirer
- JD (ne couvre pas toute la moelle)

                
                
                
                'errsm_03',
                'errsm_04',
                'errsm_05',
                'errsm_09',
                'errsm_10',
                'errsm_11',
                'errsm_12',
                'errsm_13',
                'errsm_14',
                'errsm_16',
                'errsm_17',
                'errsm_18',
                'errsm_20',
                'errsm_21',
                'errsm_23',
                'errsm_24',
                'errsm_25',
                'errsm_30',
                'errsm_31',
                'errsm_32',
                'errsm_33',
                'errsm_34',
                'errsm_35',
                'errsm_36',
                'errsm_37',
                'errsm_43',
                'errsm_44',
                'pain_pilot_1',
                'pain_pilot_2',
                'pain_pilot_3',
                'pain_pilot_4',
                'pain_pilot_7',
                'sct_001',
                'sct_002'
                
"""

list_subjects =[
                'ALT',
                'AM',
                'AP',
                'ED',
                'FR',
                'GB',
                'HB',
                'JW',
                'MLL',
                'MT',
                'PA',
                'T045',
                'T047',
                'VC',
                'VG',
                'VP'

                ]

PATH_OUTPUT = '/Users/benjamindeleener/data/PAM50_2017/output/'


def move_data():
    timer_move = sct.Timer(len(list_subjects))
    timer_move.start()
    for subject_name in list_subjects:
        sct.create_folder(path_data_new + subject_name + '/t1/')

        shutil.copy(path_data_old + subject_name + '/T1/data_RPI.nii.gz',
                    path_data_new + subject_name + '/t1/t1.nii.gz')

        sct.create_folder(path_data_new + subject_name + '/t2/')

        shutil.copy(path_data_old + subject_name + '/T2/data_RPI.nii.gz',
                    path_data_new + subject_name + '/t2/t2.nii.gz')

        timer_move.add_iteration()
    timer_move.stop()


def multisegment_spinalcord(contrast):
    print contrast
    num_initialisation = 10
    # initialisation_range = np.linspace(0, 1, num_initialisation + 2)[1:-1]
    #initialisation_range = np.linspace(0.25, 0.75, num_initialisation)
    x = 0.4 * np.logspace(0.1, 1, num_initialisation/2, endpoint=False) / max(np.logspace(0.1, 1, 5, endpoint=False))
    initialisation_range = 0.5 + np.concatenate([-x[::-1], [0.0], x])
    weights = np.concatenate([x, [0.5], x[::-1]])
    weights /= sum(weights)

    timer_segmentation = sct.Timer(len(initialisation_range) * len(list_subjects))
    timer_segmentation.start()

    foregroundValue = 1
    threshold = 0.95

    for subject_name in list_subjects:
        folder_output = path_data_new + subject_name + '/' + contrast + '/'
        list_files = [folder_output + contrast + '_seg_' + str(i+1) + '.nii.gz' for i in range(len(initialisation_range))]

        temp_fname = folder_output + contrast + '_seg_temp.nii.gz'
        for i, init in enumerate(initialisation_range):
            cmd_propseg = 'sct_propseg -i ' + path_data_new + subject_name + '/' + contrast + '/' + contrast + '.nii.gz -c ' + contrast + ' -init ' + str(init) + ' -ofolder ' + folder_output + ' -min-contrast 5'
            if i != 0:
                cmd_propseg += ' -init-centerline ' + folder_output + contrast + '_centerline_optic.nii.gz'
            sct.run(cmd_propseg, verbose=1)
            os.rename(folder_output + contrast + '_seg.nii.gz', list_files[i])

            """
            if i != 0:
                sct.run('fslmaths ' + list_files[i] + ' -mul ' + str(weights[i]) + ' -add ' + temp_fname + ' ' + temp_fname, verbose=0)
            else:
                sct.run('fslmaths ' + list_files[i] + ' -mul ' + str(weights[i]) + ' ' + temp_fname, verbose=0)
            """
            timer_segmentation.add_iteration()

        #sct.run('sct_maths -i ' + temp_fname + ' -thr ' + str(threshold) + ' -o ' + folder_output + contrast + '_seg.nii.gz', verbose=0)


        segmentations = [sitk.ReadImage(file_name, sitk.sitkUInt8) for file_name in list_files]
        reference_segmentation_STAPLE_probabilities = sitk.STAPLE(segmentations, foregroundValue)
        sitk.WriteImage(reference_segmentation_STAPLE_probabilities, folder_output + contrast + '_seg_prob.nii.gz')
        reference_segmentation_STAPLE = reference_segmentation_STAPLE_probabilities > threshold
        sitk.WriteImage(reference_segmentation_STAPLE, folder_output + contrast + '_seg.nii.gz')
    timer_segmentation.stop()


def segment_spinalcord(contrast):
    timer_segmentation = sct.Timer(len(list_subjects))
    timer_segmentation.start()
    for subject_name in list_subjects:
        sct.run('sct_propseg -i ' + path_data_new + subject_name + '/' + contrast + '/' + contrast + '.nii.gz -c t1 '
                '-ofolder ' + path_data_new + subject_name + '/' + contrast + '/', verbose=0)

        timer_segmentation.add_iteration()
    timer_segmentation.stop()


def generate_centerline(contrast):
    timer_centerline = sct.Timer(len(list_subjects))
    timer_centerline.start()
    for subject_name in list_subjects:
        sct.run('sct_process_segmentation -i ' + path_data_new + subject_name + '/' + contrast + '/' + contrast + '_seg_manual.nii.gz -p centerline '
                                         '-ofolder ' + path_data_new + subject_name + '/' + contrast + '/', verbose=1)

        timer_centerline.add_iteration()
    timer_centerline.stop()


def compute_ICBM152_centerline():
    from msct_image import Image

    im = Image('/Users/benjamindeleener/data/mni_icbm152_nlin_sym_09c/mni_icbm152_t1_tal_nlin_sym_09c_ground_truth.nii.gz')
    coord = im.getNonZeroCoordinates(sorting='z', reverse_coord=True)
    coord_physical = []

    for c in coord:
        if c.value <= 22 or c.value in [49, 50]:  # 22 corresponds to L2
            c_p = im.transfo_pix2phys([[c.x, c.y, c.z]])[0]
            c_p.append(c.value)
            coord_physical.append(c_p)

    from sct_straighten_spinalcord import smooth_centerline
    x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline(
        '/Users/benjamindeleener/data/mni_icbm152_nlin_sym_09c/mni_icbm152_t1_centerline_manual.nii.gz', algo_fitting='nurbs',
        verbose=0, nurbs_pts_number=300, all_slices=False, phys_coordinates=True, remove_outliers=False)
    from msct_types import Centerline
    centerline = Centerline(x_centerline_fit, y_centerline_fit, z_centerline,
                            x_centerline_deriv, y_centerline_deriv, z_centerline_deriv)

    centerline.compute_vertebral_distribution(coord_physical, label_reference='PMG')
    return centerline


def remove_file(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

def clean_segmentation(contrast):
    for subject_name in list_subjects:
        folder = path_data_new + subject_name + '/' + contrast + '/'
        remove_file(folder + contrast + '_seg.nii.gz')
        import os, glob
        for filename in glob.glob(folder + contrast + '_seg_*'):
            os.remove(filename)
        remove_file(folder + contrast + '_centerline.nii.gz')
        remove_file(folder + contrast + '_centerline_optic.nii.gz')





def average_centerline(contrast):
    centerline_icbm152 = compute_ICBM152_centerline()


    number_of_points_in_centerline = 4000
    height_of_template_space = 1100
    x_size_of_template_space = 201
    y_size_of_template_space = 201
    spacing = 0.5

    list_dist_disks = []
    list_centerline = []
    from sct_straighten_spinalcord import smooth_centerline
    from msct_image import Image

    timer_centerline = sct.Timer(len(list_subjects))
    timer_centerline.start()
    for subject_name in list_subjects:
        folder_output = path_data_new + subject_name + '/' + contrast + '/'

        # go to output folder
        print '\nGo to output folder ' + folder_output
        os.chdir(folder_output)

        im = Image(contrast + '_ground_truth.nii.gz')
        coord = im.getNonZeroCoordinates(sorting='z', reverse_coord=True)
        coord_physical = []


        C1_position, C2_position, C3_position = None, None, None
        for c in coord:
            if c.value <= 22 or c.value in [49, 50]:  # 22 corresponds to L2
                c_p = im.transfo_pix2phys([[c.x, c.y, c.z]])[0]
                c_p.append(c.value)
                coord_physical.append(c_p)
            #if c.value == 1:
            #    C1_position = c
            #elif c.value == 3:
            #    C3_position = c
        """
        # add C2 average position
        if C2_position is None:
            c_p = im.transfo_pix2phys([[(C1_position.x + C3_position.x) / 2.0, (C1_position.y + C3_position.y) / 2.0, (C1_position.z + C3_position.z) / 2.0]])[0]
            c_p.append(2)
            coord_physical.append(c_p)
        """

        x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline(
            contrast + '_centerline_manual.nii.gz', algo_fitting='nurbs',
            verbose=0, nurbs_pts_number=number_of_points_in_centerline, all_slices=False, phys_coordinates=True,
            remove_outliers=False)
        from msct_types import Centerline
        centerline = Centerline(x_centerline_fit, y_centerline_fit, z_centerline,
                                x_centerline_deriv, y_centerline_deriv, z_centerline_deriv)

        centerline.compute_vertebral_distribution(coord_physical)
        list_dist_disks.append(centerline.distance_from_C1label)
        list_centerline.append(centerline)
        timer_centerline.add_iteration()

    import numpy as np

    length_vertebral_levels = {}
    for dist_disks in list_dist_disks:
        for disk_label in dist_disks:
            if disk_label == 'C1':
                length = 0.0
            elif disk_label == 'PMJ':
                length = abs(dist_disks[disk_label] - dist_disks['PMG'])
            elif disk_label == 'PMG':
                length = abs(dist_disks[disk_label] - dist_disks['C1'])
            else:
                index_current_label = list_labels.index(labels_regions[disk_label])
                previous_label = regions_labels[str(list_labels[index_current_label - 1])]
                length = dist_disks[disk_label] - dist_disks[previous_label]

            if disk_label in length_vertebral_levels:
                length_vertebral_levels[disk_label].append(length)
            else:
                length_vertebral_levels[disk_label] = [length]
    print length_vertebral_levels

    average_length = {}
    for disk_label in length_vertebral_levels:
        mean = np.mean(length_vertebral_levels[disk_label])
        std = np.std(length_vertebral_levels[disk_label])
        average_length[disk_label] = [disk_label, mean, std]
    print average_length

    distances_disks_from_C1 = {'C1': 0.0, 'PMG': -average_length['PMG'][1],
                               'PMJ': -average_length['PMG'][1] - average_length['PMJ'][1]}
    for disk_number in list_labels:
        if disk_number not in [50, 49, 1] and regions_labels[str(disk_number)] in average_length:
            #if disk_number == 3:
            #    distances_disks_from_C1[regions_labels[str(disk_number)]] = distances_disks_from_C1[regions_labels[str(disk_number - 2)]] + average_length[regions_labels[str(disk_number)]][1]
            #else:
            distances_disks_from_C1[regions_labels[str(disk_number)]] = distances_disks_from_C1[regions_labels[str(disk_number - 1)]] + average_length[regions_labels[str(disk_number)]][1]
    print '\n', distances_disks_from_C1

    """
    distances_disks_from_C1 = {}
    for dist_disks in list_dist_disks:
        for disk_label in dist_disks:
            if disk_label in distances_disks_from_C1:
                distances_disks_from_C1[disk_label].append(dist_disks[disk_label])
            else:
                distances_disks_from_C1[disk_label] = [dist_disks[disk_label]]
    """

    average_distances = []
    for disk_label in distances_disks_from_C1:
        mean = np.mean(distances_disks_from_C1[disk_label])
        std = np.std(distances_disks_from_C1[disk_label])
        average_distances.append([disk_label, mean, std])

    #average_distances.append(['C2', np.mean(distances_disks_from_C1['C3'] / 2.0), 0.0])

    # create average space
    from operator import itemgetter
    average_distances = sorted(average_distances, key=itemgetter(1))
    import cPickle as pickle
    import bz2
    with bz2.BZ2File(PATH_OUTPUT + 'template_distances_from_C1_' + contrast + '.pbz2', 'w') as f:
        pickle.dump(average_distances, f)

    print '\nAverage distance\n', average_distances

    number_of_points_between_levels = 100
    disk_average_coordinates = {}
    points_average_centerline = []
    label_points = []
    average_positions_from_C1 = {}
    disk_position_in_centerline = {}

    for i in range(len(average_distances)):
        disk_label = average_distances[i][0]
        average_positions_from_C1[disk_label] = average_distances[i][1]

        for j in range(number_of_points_between_levels):
            relative_position = float(j) / float(number_of_points_between_levels)
            if disk_label in ['PMJ', 'PMG']:
                relative_position = 1.0 - relative_position
            list_coordinates = [[]] * len(list_centerline)
            for k, centerline in enumerate(list_centerline):
                idx_closest = centerline.get_closest_to_relative_position(disk_label, relative_position)
                if idx_closest is not None:
                    coordinate_closest = centerline.get_point_from_index(idx_closest[0])
                    list_coordinates[k] = coordinate_closest.tolist()
                else:
                    list_coordinates[k] = [np.nan, np.nan, np.nan]

            # average all coordinates
            average_coord = np.nanmean(list_coordinates, axis=0)
            # add it to averaged centerline list of points
            points_average_centerline.append(average_coord)
            label_points.append(disk_label)
            if j == 0:
                disk_average_coordinates[disk_label] = average_coord
                disk_position_in_centerline[disk_label] = i * number_of_points_between_levels

    """
    # compute average vertebral level length
    length_vertebral_levels = {}
    for i in range(len(list_labels) - 1):
        number_vert = list_labels[i]
        label_vert = regions_labels[str(number_vert)]
        label_vert_next = regions_labels[str(list_labels[i + 1])]
        if label_vert in average_positions_from_C1 and label_vert_next in average_positions_from_C1:
            length_vertebral_levels[label_vert] = average_positions_from_C1[label_vert_next] - average_positions_from_C1[label_vert]
    print length_vertebral_levels
    """
    with bz2.BZ2File(PATH_OUTPUT + 'template_vertebral_length_' + contrast + '.pbz2',
                     'w') as f:
        pickle.dump(length_vertebral_levels, f)

    cmap = get_cmap(len(list_centerline))
    from matplotlib.pyplot import cm
    color = iter(cm.rainbow(np.linspace(0, 1, len(list_centerline))))

    # generate averaged centerline
    plt.figure(1)
    # ax = plt.subplot(211)
    plt.subplot(211)
    for k, centerline in enumerate(list_centerline):
        col = cmap(k)
        col = next(color)
        position_C1 = centerline.points[centerline.index_disk['C1']]
        plt.plot([coord[2] - position_C1[2] for coord in centerline.points],
                 [coord[0] - position_C1[0] for coord in centerline.points], color=col)
        for label_disk in labels_regions:
            if label_disk in centerline.index_disk:
                point = centerline.points[centerline.index_disk[label_disk]]
                plt.scatter(point[2] - position_C1[2], point[0] - position_C1[0], color=col, s=5)

    position_C1 = disk_average_coordinates['C1']
    plt.plot([coord[2] - position_C1[2] for coord in points_average_centerline],
             [coord[0] - position_C1[0] for coord in points_average_centerline], color='g', linewidth=3)
    for label_disk in labels_regions:
        if label_disk in disk_average_coordinates:
            point = disk_average_coordinates[label_disk]
            plt.scatter(point[2] - position_C1[2], point[0] - position_C1[0], marker='*', color='green', s=25)

    plt.grid()
    MO_array = [[points_average_centerline[i][0], points_average_centerline[i][1], points_average_centerline[i][2]]
                for i in range(len(points_average_centerline)) if label_points[i] == 'PMG']
    PONS_array = [
        [points_average_centerline[i][0], points_average_centerline[i][1], points_average_centerline[i][2]] for i in
        range(len(points_average_centerline)) if label_points[i] == 'PMJ']
    # plt.plot([coord[2] for coord in MO_array], [coord[0] for coord in MO_array], 'mo')
    # plt.plot([coord[2] for coord in PONS_array], [coord[0] for coord in PONS_array], 'ko')

    color = iter(cm.rainbow(np.linspace(0, 1, len(list_centerline))))

    plt.title("X")
    # ax.set_aspect('equal')
    plt.xlabel('z')
    plt.ylabel('x')
    # ay = plt.subplot(212)
    plt.subplot(212)
    for k, centerline in enumerate(list_centerline):
        col = cmap(k)
        col = next(color)
        position_C1 = centerline.points[centerline.index_disk['C1']]
        plt.plot([coord[2] - position_C1[2] for coord in centerline.points],
                 [coord[1] - position_C1[1] for coord in centerline.points], color=col)
        for label_disk in labels_regions:
            if label_disk in centerline.index_disk:
                point = centerline.points[centerline.index_disk[label_disk]]
                plt.scatter(point[2] - position_C1[2], point[1] - position_C1[1], color=col, s=5)

    position_C1 = disk_average_coordinates['C1']
    plt.plot([coord[2] - position_C1[2] for coord in points_average_centerline], [coord[1] - position_C1[1] for coord in points_average_centerline], color='g', linewidth=3)
    for label_disk in labels_regions:
        if label_disk in disk_average_coordinates:
            point = disk_average_coordinates[label_disk]
            plt.scatter(point[2] - position_C1[2], point[1] - position_C1[1], marker='*', color='green', s=25)

    plt.grid()
    # plt.plot([coord[2] for coord in MO_array], [coord[1] for coord in MO_array], 'mo')
    # plt.plot([coord[2] for coord in PONS_array], [coord[1] for coord in PONS_array], 'ko')
    plt.title("Y")
    # ay.set_aspect('equal')
    plt.xlabel('z')
    plt.ylabel('y')
    plt.show()

    # create final template space
    #coord_C1 = disk_average_coordinates['C1']  # origin on average C1
    coord_C1 = np.copy(centerline_icbm152.points[centerline_icbm152.index_disk['PMG']])
    coord_C1[2] -= length_vertebral_levels['PMG'][1]
    position_template_disks = {}

    #average_positions_from_C1['C2'] = average_positions_from_C1['C3'] / 2.0

    for disk in average_length:
        if disk in ['PMJ', 'PMG']:
            position_template_disks[disk] = centerline_icbm152.points[centerline_icbm152.index_disk[disk]]
        elif disk == 'C1':
            position_template_disks[disk] = coord_C1.copy()
        else:
            coord_disk = coord_C1.copy()
            coord_disk[2] -= average_positions_from_C1[disk]
            position_template_disks[disk] = coord_disk
        print disk, disk_average_coordinates[disk], average_positions_from_C1[disk], position_template_disks[disk]

    # change centerline to be straight below C1
    index_C1 = disk_position_in_centerline['C1']
    index_PMG = disk_position_in_centerline['PMG']
    points_average_centerline_template = []
    for i in range(0, len(points_average_centerline)):
        current_label = label_points[i]
        if current_label in average_length:
            length_current_label = average_length[current_label][1]
            relative_position_from_disk = float(i - disk_position_in_centerline[current_label]) / float(number_of_points_between_levels)
            # print i, coord_C1[2], average_positions_from_C1[current_label], length_current_label
            temp_point = np.copy(coord_C1)
            #points_average_centerline[i][0] = coord_C1[0]
            if i >= index_PMG:
                temp_point[2] = coord_C1[2] - average_positions_from_C1[current_label] - relative_position_from_disk * length_current_label
                #points_average_centerline[i][1] = coord_C1[1]
                #points_average_centerline[i][2] = coord_C1[2] - average_positions_from_C1[current_label] - relative_position_from_disk * length_current_label
            #else:
            #    points_average_centerline[i][1] = coord_C1[1] + (points_average_centerline[i][1] - coord_C1[1]) * 2.0
            points_average_centerline_template.append(temp_point)
        #else:
        #    points_average_centerline[i] = None
    #points_average_centerline = [x for x in points_average_centerline if x is not None]

    # append icbm152 centerline from PMG
    points_icbm152 = centerline_icbm152.points[centerline_icbm152.index_disk['PMG']:]
    points_icbm152 = points_icbm152[::-1]
    #points_average_centerline = np.concatenate([points_average_centerline, points_icbm152])
    points_average_centerline = np.concatenate([points_icbm152, points_average_centerline_template])

    # generate averaged centerline
    plt.figure(1)
    # ax = plt.subplot(211)
    plt.subplot(211)
    for k, centerline in enumerate(list_centerline):
        plt.plot([coord[2] for coord in centerline.points], [coord[0] for coord in centerline.points], 'r')
    plt.plot([coord[2] for coord in points_average_centerline], [coord[0] for coord in points_average_centerline], 'bo')
    plt.title("X")
    # ax.set_aspect('equal')
    plt.xlabel('z')
    plt.ylabel('x')
    # ay = plt.subplot(212)
    plt.subplot(212)
    for k, centerline in enumerate(list_centerline):
        plt.plot([coord[2] for coord in centerline.points], [coord[1] for coord in centerline.points], 'r')
    plt.plot([coord[2] for coord in points_average_centerline], [coord[1] for coord in points_average_centerline], 'bo')
    plt.title("Y")
    # ay.set_aspect('equal')
    plt.xlabel('z')
    plt.ylabel('y')
    plt.show()

    # creating template space
    size_template_z = int(abs(points_average_centerline[0][2] - points_average_centerline[-1][2]) / spacing) + 15
    print points_average_centerline[0], points_average_centerline[-1]
    print size_template_z

    # saving template centerline and levels
    # generate template space
    from msct_image import Image
    from numpy import zeros
    template = Image('/Users/benjamindeleener/code/sct/dev/template_creation/template_landmarks-mm.nii.gz')
    template_space = Image([x_size_of_template_space, y_size_of_template_space, size_template_z])
    template_space.data = zeros((x_size_of_template_space, y_size_of_template_space, size_template_z))
    template_space.hdr = template.hdr
    template_space.hdr.set_data_dtype('float32')
    # origin = [(x_size_of_template_space - 1.0) / 4.0, -(y_size_of_template_space - 1.0) / 4.0, -((size_template_z / 4.0) - spacing)]
    origin = [points_average_centerline[-1][0] + x_size_of_template_space * spacing / 2.0,
              points_average_centerline[-1][1] - y_size_of_template_space * spacing / 2.0,
              (points_average_centerline[-1][2] - spacing)]
    print origin
    template_space.hdr.structarr['dim'] = [3.0, x_size_of_template_space, y_size_of_template_space, size_template_z,
                                           1.0, 1.0, 1.0, 1.0]
    template_space.hdr.structarr['pixdim'] = [-1.0, spacing, spacing, spacing, 1.0, 1.0, 1.0, 1.0]
    template_space.hdr.structarr['qoffset_x'] = origin[0]
    template_space.hdr.structarr['qoffset_y'] = origin[1]
    template_space.hdr.structarr['qoffset_z'] = origin[2]
    template_space.hdr.structarr['srow_x'][-1] = origin[0]
    template_space.hdr.structarr['srow_y'][-1] = origin[1]
    template_space.hdr.structarr['srow_z'][-1] = origin[2]
    template_space.hdr.structarr['srow_x'][0] = -spacing
    template_space.hdr.structarr['srow_y'][1] = spacing
    template_space.hdr.structarr['srow_z'][2] = spacing
    template_space.setFileName(PATH_OUTPUT + 'template_space.nii.gz')
    template_space.save()

    # generate template centerline
    image_centerline = template_space.copy()
    for coord in points_average_centerline:
        coord_pix = image_centerline.transfo_phys2pix([coord])[0]
        if 0 <= coord_pix[0] < image_centerline.data.shape[0] and 0 <= coord_pix[1] < image_centerline.data.shape[1] and 0 <= coord_pix[2] < image_centerline.data.shape[2]:
            image_centerline.data[int(coord_pix[0]), int(coord_pix[1]), int(coord_pix[2])] = 1
    image_centerline.setFileName(PATH_OUTPUT + 'template_centerline.nii.gz')
    image_centerline.save(type='uint8')

    # generate template disks position
    image_disks = template_space.copy()
    for disk in position_template_disks:
        label = labels_regions[disk]
        coord = position_template_disks[disk]
        coord_pix = image_disks.transfo_phys2pix([coord])[0]
        if 0 <= coord_pix[0] < image_disks.data.shape[0] and 0 <= coord_pix[1] < image_disks.data.shape[1] and 0 <= coord_pix[2] < image_disks.data.shape[2]:
            image_disks.data[int(coord_pix[0]), int(coord_pix[1]), int(coord_pix[2])] = label
        else:
            print coord_pix
            print 'ERROR: the disk label ' + str(disk) + ' is not in the template image.'
    image_disks.setFileName(PATH_OUTPUT + 'template_disks.nii.gz')
    image_disks.save(type='uint8')


def compute_csa(fname_segmentation, fname_disks, fname_centerline):
    # compute csa on the input segmentation
    # this function create a csv file (csa_per_slice.txt) containing csa for each slice in the image
    sct.run('sct_process_segmentation '
            '-i ' + fname_segmentation + ' '
            '-p csa')

    # read csv file to extract csa per slice
    csa_file = open('csa_per_slice.txt', 'r')
    csa = csa_file.read()
    csa_file.close()
    csa_lines = csa.split('\n')[1:-1]
    z_values, csa_values = [], []
    for l in csa_lines:
        s = l.split(',')
        z_values.append(int(s[0]))
        csa_values.append(float(s[1]))

    # compute a lookup table with continuous vertebral levels and slice position
    from sct_straighten_spinalcord import smooth_centerline
    from msct_image import Image
    im = Image(fname_disks)
    coord = im.getNonZeroCoordinates(sorting='z', reverse_coord=True)
    coord_physical = []
    for c in coord:
        c_p = im.transfo_pix2phys([[c.x, c.y, c.z]])[0]
        c_p.append(c.value)
        coord_physical.append(c_p)

    number_of_points_in_centerline = 4000
    x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline(
        fname_centerline,
        algo_fitting='nurbs',
        verbose=0, nurbs_pts_number=number_of_points_in_centerline, all_slices=False, phys_coordinates=True,
        remove_outliers=True)
    from msct_types import Centerline
    centerline = Centerline(x_centerline_fit, y_centerline_fit, z_centerline,
                            x_centerline_deriv, y_centerline_deriv, z_centerline_deriv)

    centerline.compute_vertebral_distribution(coord_physical)
    x, y, z, xd, yd, zd = centerline.average_coordinates_over_slices(im)
    coordinates = []
    for i in range(len(z)):
        nearest_index = centerline.find_nearest_indexes([[x[i], y[i], z[i]]])[0]
        disk_label = centerline.l_points[nearest_index]
        relative_position = centerline.dist_points_rel[nearest_index]
        if disk_label != 0:
            if centerline.labels_regions[disk_label] > centerline.last_label and centerline.labels_regions[disk_label] not in [49, 50]:
                print disk_label, centerline.labels_regions[disk_label]
                coordinates.append(float(labels_regions[disk_label]) + relative_position / centerline.average_vert_length[disk_label])
            else:
                coordinates.append(float(labels_regions[disk_label]) + relative_position)

    # concatenate results
    result_levels, result_csa = [], []
    z_pix = [int(im.transfo_phys2pix([[x[k], y[k], z[k]]])[0][2]) for k in range(len(z))]
    for i, zi in enumerate(z_values):
        try:
            corresponding_values = z_pix.index(int(zi))
        except ValueError as e:
            print 'got exception'
            continue

        if coordinates[corresponding_values] <= 26:
            result_levels.append(coordinates[corresponding_values])
            result_csa.append(csa_values[i])

    #print result_levels, result_csa
    #plt.plot(result_levels, result_csa)
    #plt.show()

    return result_levels, result_csa


def compare_csa(contrast, fname_segmentation, fname_disks, fname_centerline):
    timer_csa = sct.Timer(len(list_subjects))
    timer_csa.start()

    results_csa = {}

    for subject_name in list_subjects:
        folder_output = path_data_new + subject_name + '/' + contrast + '/'
        # go to output folder
        print '\nComparing CSA ' + folder_output
        os.chdir(folder_output)

        levels, csa = compute_csa(fname_segmentation, fname_disks, fname_centerline)
        results_csa[subject_name] = [levels, csa]
        timer_csa.add_iteration()
    timer_csa.stop()

    print results_csa
    import json
    with open(PATH_OUTPUT + 'csa.txt', 'w') as outfile:
        json.dump(results_csa, outfile)

    plt.figure()
    for subject in results_csa:
        plt.plot(results_csa[subject][0], results_csa[subject][1])
    plt.legend([subject for subject in results_csa])
    plt.show()



def compute_spinalcord_length(contrast, fname_segmentation):
    timer_length = sct.Timer(len(list_subjects))
    timer_length.start()

    results = {}

    for subject_name in list_subjects:
        folder_output = path_data_new + subject_name + '/' + contrast + '/'
        # go to output folder
        print '\nComputing length ' + folder_output
        os.chdir(folder_output)

        from sct_straighten_spinalcord import smooth_centerline

        number_of_points_in_centerline = 4000
        x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline(
            fname_segmentation,
            algo_fitting='nurbs',
            verbose=0, nurbs_pts_number=number_of_points_in_centerline, all_slices=False, phys_coordinates=True,
            remove_outliers=True)
        from msct_types import Centerline
        centerline = Centerline(x_centerline_fit, y_centerline_fit, z_centerline,
                                x_centerline_deriv, y_centerline_deriv, z_centerline_deriv)

        results[subject_name] = centerline.length
        timer_length.add_iteration()
    timer_length.stop()

    print results
    import json
    with open(PATH_OUTPUT + 'length.txt', 'w') as outfile:
        json.dump(results, outfile)

def straighten_all_subjects(contrast):
    # straightening of each subject on the new template
    timer_straightening = sct.Timer(len(list_subjects))
    timer_straightening.start()
    for subject_name in list_subjects:
        folder_output = path_data_new + subject_name + '/' + contrast + '/'

        # go to output folder
        print '\nStraightening ' + folder_output
        os.chdir(folder_output)
        sct.run('sct_straighten_spinalcord'
                ' -i ' + contrast + '.nii.gz'
                ' -s ' + contrast + '_centerline_manual.nii.gz'
                ' -disks-input ' + contrast + '_ground_truth.nii.gz'
                ' -ref ' + PATH_OUTPUT + 'template_centerline.nii.gz'
                ' -disks-ref ' + PATH_OUTPUT + 'template_disks.nii.gz'
                ' -disable-straight2curved'
                ' -param threshold_distance=1', verbose=1)

        sct.run('cp ' + contrast + '_straight.nii.gz ' + PATH_OUTPUT + 'final/' + subject_name + '_' + contrast + '.nii.gz')
        timer_straightening.add_iteration()
    timer_straightening.stop()

#clean_segmentation('t1')
#multisegment_spinalcord('t1')
#generate_centerline('t1')
#average_centerline('t1')
#straighten_all_subjects('t1')

#compare_csa(contrast='t1', fname_segmentation='t1_seg_manual.nii.gz', fname_disks='t1_ground_truth.nii.gz', fname_centerline='t1_centerline_manual.nii.gz')
#compute_spinalcord_length(contrast='t1', fname_segmentation='t1_seg_manual.nii.gz')


def smooth_cubicsplines(x, y):
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(x, y, bc_type='natural')
    return [cs(xi) for xi in x]

def smooth_UnivariateSpline(x, y):
    from scipy.interpolate import UnivariateSpline

    def moving_average(series, sigma=5):
        from scipy.signal import gaussian
        from scipy.ndimage import filters
        b = gaussian(39, sigma)
        average = filters.convolve1d(series, b / b.sum())
        var = filters.convolve1d(np.power(series - average, 2), b / b.sum())
        return average, var

    _, var = moving_average(y)
    sp = UnivariateSpline(x, y, w=1 / np.sqrt(var))
    return [sp(xi) for xi in x]

def smooth_spline(x, y):
    from scipy.interpolate import splrep, splev
    tck = splrep(x, y)
    return splev(x, tck)

def smooth_gmm(x, y):
    from sklearn.mixture import GMM
    x_gmm, y_gmm = np.array(x), np.array(y)
    y_gmm = y_gmm.reshape(len(y_gmm), 1)
    x_gmm = x_gmm.reshape(len(x_gmm), 1)

    model = GMM(20).fit(y_gmm)
    return -model.score_samples(x_gmm)

def smooth_kde(x, y):
    # Get the data
    obs_wave, obs_flux = np.array(x), np.array(y)

    # Center the x data in zero and normalized the y data to the area of the curve
    #n_wave = obs_wave - obs_wave[np.argmax(obs_flux)]
    n_flux = obs_flux / sum(obs_flux)

    # Generate a distribution of points matcthing the curve
    line_distribution = np.random.choice(a=obs_wave, size=100000, p=n_flux)
    number_points = len(line_distribution)

    """
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=30)
    gmm.fit(np.reshape(line_distribution, (number_points, 1)))

    from scipy.stats import norm
    gauss_mixt = np.array([p * norm.pdf(obs_wave, mu, sd) for mu, sd, p in zip(gmm.means_.flatten(), np.sqrt(gmm.covariances_.flatten()), gmm.weights_)])
    gauss_mixt_t = np.sum(gauss_mixt, axis=0)
    
    return gauss_mixt_t
    """

    from sklearn.neighbors import KernelDensity

    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(line_distribution)
    log_dens = kde.score_samples(obs_wave)
    return np.exp(log_dens)


import json
with open(PATH_OUTPUT + 'csa.txt') as data_file:
    results_csa = json.load(data_file)


plt.figure()
for i, subject in enumerate(results_csa):
    ax = plt.subplot(4, 4, i+1)
    x, y = results_csa[subject][0][::-1], results_csa[subject][1][::-1]
    y_smooth = y
    #y_smooth = smooth_UnivariateSpline(x, y)
    #y_smooth = smooth(np.array(results_csa[subject][1]), window_len=30)
    #y_smooth = smooth_kde(x, y)
    plt.plot(x, y_smooth)
    plt.title(subject)
plt.show()

plt.figure()
for i, subject in enumerate(results_csa):
    x, y = results_csa[subject][0][::-1], results_csa[subject][1][::-1]
    y_smooth = y
    #y_smooth = smooth_UnivariateSpline(x, y)
    #y_smooth = smooth(np.array(results_csa[subject][1]), window_len=40)
    #y_smooth = smooth_kde(x, y)
    plt.plot(x, y_smooth)
plt.legend([subject for subject in results_csa])
plt.show()




"""
sct_make_ground_truth.py -i t1.nii.gz -save-as niigz
Produce centerline along spinal cord (above and below segmentation) called {contrast}_centerline_manual.nii.gz
"""
