#!/usr/bin/env python

import sys, io, os, glob, shutil
import numpy as np

#import SimpleITK as sitk
import matplotlib.pyplot as plt

import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy

import sct_utils as sct
from msct_types import Centerline
from sct_straighten_spinalcord import smooth_centerline
from msct_image import Image


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
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
    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')

    y = numpy.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len+1]


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
                'VP',
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

list_subjects =['ALT',
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
                'VP',
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
                ]

PATH_OUTPUT = '/Users/benjamindeleener/data/PAM50_2017/output/'


def move_data():
    timer_move = sct.Timer(len(list_subjects))
    timer_move.start()
    for subject_name in list_subjects:
        sct.create_folder(os.path.join(path_data_new, subject_name, 't1'))

        sct.copy(os.path.join(path_data_old, subject_name, 'T1', 'data_RPI.nii.gz'),
                    os.path.join(path_data_new, subject_name, 't1', 't1.nii.gz'))

        sct.create_folder(os.path.join(path_data_new, subject_name, 't2'))

        sct.copy(os.path.join(path_data_old, subject_name, 'T2', 'data_RPI.nii.gz'),
                    os.path.join(path_data_new, subject_name, 't2', 't2.nii.gz'))

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
        folder_output = os.path.join(path_data_new, subject_name, contrast)
        list_files = [os.path.join(folder_output, contrast + '_seg_' + str(i+1) + '.nii.gz') for i in range(len(initialisation_range))]

        temp_fname = os.path.join(folder_output, contrast + '_seg_temp.nii.gz')
        for i, init in enumerate(initialisation_range):
            cmd_propseg = 'sct_propseg -i ' + os.path.join(path_data_new, subject_name, contrast, contrast + '.nii.gz') + ' -c ' + contrast + ' -init ' + str(init) + ' -ofolder ' + folder_output + ' -min-contrast 5'
            if i != 0:
                cmd_propseg += ' -init-centerline ' + os.path.join(folder_output, contrast + '_centerline_optic.nii.gz')
            sct.run(cmd_propseg, verbose=1)
            os.rename(os.path.join(folder_output, contrast + '_seg.nii.gz'), list_files[i])

            """
            if i != 0:
                sct.run('fslmaths ' + list_files[i] + ' -mul ' + str(weights[i]) + ' -add ' + temp_fname + ' ' + temp_fname, verbose=0)
            else:
                sct.run('fslmaths ' + list_files[i] + ' -mul ' + str(weights[i]) + ' ' + temp_fname, verbose=0)
            """
            timer_segmentation.add_iteration()

        #sct.run('sct_maths -i ' + temp_fname + ' -thr ' + str(threshold) + ' -o ' + os.path.join(folder_output, contrast + '_seg.nii.gz'), verbose=0)


        segmentations = [sitk.ReadImage(file_name, sitk.sitkUInt8) for file_name in list_files]
        reference_segmentation_STAPLE_probabilities = sitk.STAPLE(segmentations, foregroundValue)
        sitk.WriteImage(reference_segmentation_STAPLE_probabilities, os.path.join(folder_output, contrast + '_seg_prob.nii.gz'))
        reference_segmentation_STAPLE = reference_segmentation_STAPLE_probabilities > threshold
        sitk.WriteImage(reference_segmentation_STAPLE, os.path.join(folder_output, contrast + '_seg.nii.gz'))
    timer_segmentation.stop()


def segment_spinalcord(contrast):
    timer_segmentation = sct.Timer(len(list_subjects))
    timer_segmentation.start()
    for subject_name in list_subjects:
        sct.run('sct_propseg -i ' + os.path.join(path_data_new, subject_name, contrast, contrast + '.nii.gz') + ' -c t1 '
                '-ofolder ' + os.path.join(path_data_new, subject_name, contrast), verbose=0)

        timer_segmentation.add_iteration()
    timer_segmentation.stop()


def generate_centerline(contrast):
    timer_centerline = sct.Timer(len(list_subjects))
    timer_centerline.start()
    for subject_name in list_subjects:
        sct.run('sct_process_segmentation -i ' + os.path.join(path_data_new, subject_name, contrast, contrast + '_seg_manual.nii.gz') + ' -p centerline '
                                         '-ofolder ' + os.path.join(path_data_new, subject_name, contrast), verbose=1)

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
        folder = os.path.join(path_data_new, subject_name, contrast)
        remove_file(folder + contrast + '_seg.nii.gz')
        import os, glob
        for filename in glob.glob(os.path.join(folder, contrast + '_seg_*')):
            os.remove(filename)
        remove_file(folder + contrast + '_centerline.nii.gz')
        remove_file(folder + contrast + '_centerline_optic.nii.gz')


def average_centerline(contrast):
    number_of_points_in_centerline = 4000


    list_dist_disks = []
    list_centerline = []
    from sct_straighten_spinalcord import smooth_centerline
    from msct_image import Image

    timer_centerline = sct.Timer(len(list_subjects))
    timer_centerline.start()
    for subject_name in list_subjects:
        folder_output = os.path.join(path_data_new, subject_name, contrast)

        # go to output folder
        print '\nGo to output folder ' + folder_output
        os.chdir(folder_output)

        im = Image(contrast + '_ground_truth.nii.gz')
        coord = im.getNonZeroCoordinates(sorting='z', reverse_coord=True)
        coord_physical = []

        for c in coord:
            if c.value <= 22 or c.value in [49, 50]:  # 22 corresponds to L2
                c_p = im.transfo_pix2phys([[c.x, c.y, c.z]])[0]
                c_p.append(c.value)
                coord_physical.append(c_p)

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


    #def generate_average_centerline():
    centerline_icbm152 = compute_ICBM152_centerline()
    height_of_template_space = 1100
    x_size_of_template_space = 201
    y_size_of_template_space = 201
    spacing = 0.5

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
                idx_closest = centerline.get_closest_to_absolute_position(disk_label, relative_position)
                if idx_closest is not None:
                    coordinate_closest = centerline.get_point_from_index(idx_closest)
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


def compute_csa(fname_segmentation, fname_disks, fname_centerline_image, force_csa_computation=False):
    # compute csa on the input segmentation
    # this function create a csv file (csa_per_slice.txt) containing csa for each slice in the image

    fname_csa = 'csa_per_slice.txt'
    if not os.path.isfile(fname_csa) or force_csa_computation:
        sct.run('sct_process_segmentation '
                '-i ' + fname_segmentation + ' '
                '-p csa')

    # read csv file to extract csa per slice
    csa_file = open(fname_csa, 'r')
    csa = csa_file.read()
    csa_file.close()
    csa_lines = csa.split('\n')[1:-1]
    z_values, csa_values = [], []
    for l in csa_lines:
        s = l.split(',')
        z_values.append(int(s[0]))
        csa_values.append(float(s[1]))

    im = Image(fname_disks)

    fname_centerline = 'centerline.npz'
    if os.path.isfile(fname_centerline):
        centerline = Centerline(fname=fname_centerline)
    else:
        # compute a lookup table with continuous vertebral levels and slice position
        coord = im.getNonZeroCoordinates(sorting='z', reverse_coord=True)
        coord_physical = []
        for c in coord:
            c_p = im.transfo_pix2phys([[c.x, c.y, c.z]])[0]
            c_p.append(c.value)
            coord_physical.append(c_p)

        number_of_points_in_centerline = 4000
        x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline(
            fname_centerline_image,
            algo_fitting='nurbs',
            verbose=0, nurbs_pts_number=number_of_points_in_centerline, all_slices=False, phys_coordinates=True,
            remove_outliers=True)

        centerline = Centerline(x_centerline_fit, y_centerline_fit, z_centerline,
                                x_centerline_deriv, y_centerline_deriv, z_centerline_deriv)

        centerline.compute_vertebral_distribution(coord_physical)
        centerline.save_centerline(fname_output='centerline')

    #centerline.display(mode='relative')
    lookuptable_coordinates = centerline.get_lookup_coordinates(im)

    result_levels, result_csa = [], []
    for i, zi in enumerate(z_values):
        if lookuptable_coordinates[zi] is not None and lookuptable_coordinates[zi] < 30:
            result_levels.append(lookuptable_coordinates[zi])
            result_csa.append(csa_values[i])

    #print result_levels, result_csa
    #plt.plot(result_levels, result_csa)
    #plt.show()

    return result_levels, result_csa


def compare_csa(contrast, fname_segmentation, fname_disks, fname_centerline_image):
    timer_csa = sct.Timer(len(list_subjects))
    timer_csa.start()

    results_csa = {}

    for subject_name in list_subjects:
        folder_output = os.path.join(path_data_new, subject_name, contrast)
        # go to output folder
        print '\nComparing CSA ' + folder_output
        os.chdir(folder_output)

        levels, csa = compute_csa(fname_segmentation, fname_disks, fname_centerline_image)
        results_csa[subject_name] = [levels, csa]
        timer_csa.add_iteration()
    timer_csa.stop()

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
        folder_output = os.path.join(path_data_new, subject_name, contrast)
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
        folder_output = os.path.join(path_data_new, subject_name, contrast)

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


def normalize_intensity(contrast, fname_disks, fname_centerline_image):
    timer_normalize = sct.Timer(len(list_subjects))
    timer_normalize.start()

    average_length = 10
    smooth_window = 20

    results_intensity = {}
    slices_subjects = {}
    average_intensity = 0.0

    for subject_name in list_subjects:
        #folder_output = os.path.join(path_data_new, subject_name, contrast)
        #fname_image = contrast + '.nii.gz'

        folder_output = PATH_OUTPUT + 'final'
        fname_image = subject_name + '_' + contrast + '.nii.gz'

        print '\nExtracting intensity ' + folder_output
        os.chdir(folder_output)
        image_input = Image(fname_image)
        nx, ny, nz, nt, px, py, pz, pt = image_input.dim

        fname_centerline = 'centerline.npz'
        if os.path.isfile(fname_centerline):
            centerline = Centerline(fname=fname_centerline)
        else:
            # compute a lookup table with continuous vertebral levels and slice position
            image_disks = Image(fname_disks)
            coord = image_disks.getNonZeroCoordinates(sorting='z', reverse_coord=True)
            coord_physical = []
            for c in coord:
                c_p = image_input.transfo_pix2phys([[c.x, c.y, c.z]])[0]
                c_p.append(c.value)
                coord_physical.append(c_p)

            number_of_points_in_centerline = 4000
            x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline(
                fname_centerline_image,
                algo_fitting='nurbs',
                verbose=0, nurbs_pts_number=number_of_points_in_centerline, all_slices=False, phys_coordinates=True,
                remove_outliers=True)

            centerline = Centerline(x_centerline_fit, y_centerline_fit, z_centerline,
                                    x_centerline_deriv, y_centerline_deriv, z_centerline_deriv)

            centerline.compute_vertebral_distribution(coord_physical)
            centerline.save_centerline(fname_output='centerline')

        #centerline.display(mode='relative')

        x, y, z, xd, yd, zd = centerline.average_coordinates_over_slices(image_input)
        lookuptable_coordinates = centerline.get_lookup_coordinates(image_input)

        slices, levels, intensities = [], [], []
        for i in range(len(z)):
            coord_z = image_input.transfo_phys2pix([[x[i], y[i], z[i]]])[0]
            if 0 <= coord_z[2] < nz and lookuptable_coordinates[coord_z[2]] is not None and lookuptable_coordinates[coord_z[2]] < 30 and lookuptable_coordinates[coord_z[2]] not in levels:
                slices.append(coord_z[2])
                levels.append(lookuptable_coordinates[coord_z[2]])
                #intensities.append(image_input.data[coord_z[0], coord_z[1], coord_z[2]])
                intensities.append(np.mean(image_input.data[coord_z[0]-1:coord_z[0]+2, coord_z[1]-1:coord_z[1]+2, coord_z[2]]))

        intensities = intensities[::-1]
        y = [np.mean(intensities[0:average_length])] * smooth_window + intensities + [np.mean(intensities[-average_length:])] * smooth_window
        intensities = smooth(np.array(y), window_len=smooth_window)[smooth_window:-smooth_window]
        results_intensity[subject_name] = [levels[::-1], intensities]
        slices_subjects[subject_name] = slices

        average_intensity += np.mean(intensities)

        timer_normalize.add_iteration()
    timer_normalize.stop()

    average_intensity /= float(len(list_subjects))
    print 'Average intensity =', average_intensity

    timer_normalize = sct.Timer(len(list_subjects))
    timer_normalize.start()
    for subject_name in list_subjects:
        #folder_output = os.path.join(path_data_new, subject_name, contrast)
        #fname_image = contrast + '.nii.gz'

        folder_output = PATH_OUTPUT + 'final/'
        fname_image = subject_name + '_' + contrast + '.nii.gz'

        print '\nExtracting intensity ' + folder_output
        os.chdir(folder_output)
        image_input = Image(fname_image)
        image_output = image_input.copy()
        image_output.data += average_intensity - np.mean(results_intensity[subject_name][1])
        image_output.setFileName(subject_name + '_' + contrast + '_norm.nii.gz')
        image_output.save()

        timer_normalize.add_iteration()
    timer_normalize.stop()

    #import json
    #with open(PATH_OUTPUT + 'intensity_profiles.txt', 'w') as outfile:
    #    json.dump(results_intensity, outfile)

    #from msct_nurbs import NURBS

    #plt.figure()
    #for subject in results_intensity:
    """
    data = [[y_smooth[n], results_intensity[subject][0][n]] for n in range(len(results_intensity[subject][1]))]
    nurbs = NURBS(precision=200, liste=data, nbControl=nbControl, verbose=1,
                  twodim=True, all_slices=False,
                  weights=False, maxControlPoints=25)
    P = nurbs.getCourbe2D()
    levels_fit = P[1]
    intensities_fit = P[0]
    #plt.plot(levels_fit, intensities_fit)
    """
    #    plt.plot(results_intensity[subject][0], results_intensity[subject][1] + average_intensity - np.mean(results_intensity[subject][1]))

    #plt.legend([subject for subject in results_intensity])
    #plt.show()


def normalize_intensity_template():
    #fname_template_image = '/Users/benjamindeleener/data/PAM50_2017/output/PAM50/PAM50_t1_12.nii'  # t1
    fname_template_image = '/Users/benjamindeleener/data/PAM50_2017/output/PAM50/avg_t2.nii'  # t2
    fname_template_centerline_image = '/Users/benjamindeleener/data/PAM50_2017/output/PAM50/PAM50_centerline_man.nii.gz'
    fname_template_centerline = '/Users/benjamindeleener/data/PAM50_2017/output/PAM50/centerline.npz'
    path_segmentations = '/Users/benjamindeleener/data/PAM50_2017/output/PAM50/seg/'


    # open centerline from template
    number_of_points_in_centerline = 4000
    if os.path.isfile(fname_template_centerline):
        centerline_template = Centerline(fname=fname_template_centerline)
    else:
        x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline(
            fname_template_centerline_image, algo_fitting='nurbs', verbose=0,
            nurbs_pts_number=number_of_points_in_centerline,
            all_slices=False, phys_coordinates=True, remove_outliers=True)
        centerline_template = Centerline(x_centerline_fit, y_centerline_fit, z_centerline,
                                         x_centerline_deriv, y_centerline_deriv, z_centerline_deriv)

        centerline_template.save_centerline(fname_output=fname_template_centerline)

    image_template = Image(fname_template_image)
    nx, ny, nz, nt, px, py, pz, pt = image_template.dim
    x, y, z, xd, yd, zd = centerline_template.average_coordinates_over_slices(image_template)
    z_values, intensities = [], []

    for i in range(len(z)):
        coord_z = image_template.transfo_phys2pix([[x[i], y[i], z[i]]])[0]
        z_values.append(coord_z[2])
        intensities.append(np.mean(image_template.data[coord_z[0] - 1:coord_z[0] + 2, coord_z[1] - 1:coord_z[1] + 2, coord_z[2]]))

    min_z, max_z = min(z_values), max(z_values)
    from copy import copy
    intensities_temp = copy(intensities)
    z_values_temp = copy(z_values)
    for cz in range(nz):
        if cz not in z_values:
            z_values_temp.append(cz)
            if cz < min_z:
                intensities_temp.append(intensities[z_values.index(min_z)])
            elif cz > max_z:
                intensities_temp.append(intensities[z_values.index(max_z)])
            else:
                print 'error...', cz

    print intensities[z_values.index(min_z)], intensities[z_values.index(max_z)]
    intensities = intensities_temp
    z_values = z_values_temp

    arr_int = [[z_values[i], intensities[i]] for i in range(len(z_values))]
    arr_int.sort(key=lambda x: x[0])

    intensities = [c[1] for c in arr_int]
    int_smooth = smooth(np.array(intensities), window_len=50)
    mean_int = np.mean(int_smooth)
    mean_int = 500.0
    plt.figure()
    plt.plot(intensities)
    plt.plot(int_smooth)
    plt.show()

    image_template_new = image_template.copy()
    for i in range(nz):
        image_template_new.data[:, :, i] *= mean_int / int_smooth[i]
    #image_template_new.setFileName('/Users/benjamindeleener/data/PAM50_2017/output/PAM50/PAM50_t1.nii.gz')
    image_template_new.setFileName('/Users/benjamindeleener/data/PAM50_2017/output/PAM50/PAM50_t2.nii.gz')
    image_template_new.save()


def warp_segmentation(contrast):
    for subject_name in list_subjects:
        folder_output = os.path.join(path_data_new, subject_name, contrast)
        image_destination = PATH_OUTPUT + 'final/' + subject_name + '_' + contrast + '.nii.gz'
        folder_output_mnc = PATH_OUTPUT + 'mnc/'
        fname_image = 't1_seg_manual.nii.gz'

        print '\nWarping segmentation ' + folder_output
        os.chdir(folder_output)

        sct.run('sct_apply_transfo -i ' + fname_image + ' -w warp_curve2straight.nii.gz -d ' + image_destination + ' -x nn -o ' + subject_name + '_' + contrast + '_seg_straight.nii.gz')

        sct.run('nii2mnc ' + subject_name + '_' + contrast + '_seg_straight.nii.gz' + ' ' + os.path.join(folder_output_mnc, subject_name + '_' + contrast + '_seg.mnc'))


def create_mask_template():
    subject_name = list_subjects[0]
    template = Image(PATH_OUTPUT + 'final/' + subject_name + '_t1.nii.gz')
    template.data *= 0.0
    template.data += 1.0
    template.setFileName(PATH_OUTPUT + 'final/template_mask.nii.gz')
    template.save()

    folder_output = PATH_OUTPUT + 'mnc/'
    sct.run('nii2mnc ' + PATH_OUTPUT + 'final/template_mask.nii.gz ' + ' ' + os.path.join(folder_output, "template_mask.mnc"))


def convert_nii2mnc(contrast):
    for subject_name in list_subjects:
        folder_output = PATH_OUTPUT + 'mnc/'
        fname_image = subject_name + '_' + contrast + '_norm.nii.gz'
        fname_im_output = subject_name + '_' + contrast + '.mnc'

        sct.run('nii2mnc ' + PATH_OUTPUT + 'final/' + fname_image + ' ' + os.path.join(folder_output, fname_im_output))


def compute_vertebral_levels(contrast):
    import pandas as pd
    levels_per_subject = {}

    timer_levels = sct.Timer(len(list_subjects))
    timer_levels.start()

    centerlines = {}

    for subject_name in list_subjects:
        folder_output = os.path.join(path_data_new, subject_name, contrast)
        fname_image = contrast + '.nii.gz'

        print '\nExtracting lengths ' + folder_output
        os.chdir(folder_output)
        #image_input = Image(fname_image)
        #nx, ny, nz, nt, px, py, pz, pt = image_input.dim

        fname_centerline = 'centerline.npz'
        centerline = Centerline(fname=fname_centerline)
        centerline.compute_vertebral_distribution(disks_levels=centerline.disks_levels, label_reference='PMJ')
        centerlines[subject_name] = centerline

        levels, labels = [], []
        for label in centerline.labels_regions:
            if label in centerline.index_disk:
                levels.append(centerline.dist_points[centerline.index_disk[label]])
                labels.append(label)

        levels_series = pd.Series(levels, index=labels)
        levels_per_subject[subject_name] = levels_series

        timer_levels.add_iteration()
    timer_levels.stop()

    df_levels = pd.DataFrame(levels_per_subject)
    df_levels.to_csv('/Users/benjamindeleener/data/PAM50_2017/output/levels.txt')

    df_levels2 = df_levels.T
    del df_levels2['C2']
    print df_levels2.describe()
    df_levels = df_levels2.T

    C4C5_levels = df_levels2['C5']
    T11T12_levels = df_levels2['T12']

    import matplotlib.mlab as mlab
    plt.figure()
    #plt.subplot(2, 1, 1)
    for index, row in df_levels.iterrows():
        mu = row.mean()
        if mu != 0:
            sigma = row.std()
            variance = sigma**2
            print row.name, mu, sigma
            x = np.linspace(mu - 3 * variance, mu + 3 * variance, 5*variance)
            plt.plot(x, mlab.normpdf(x, mu, sigma))
    plt.xlim(0, 600)
    plt.show()

    plt.figure()
    #plt.subplot(2, 1, 2)
    import csv
    with open(PATH_OUTPUT + 'enlargements.txt') as data_file:
        results_enlargements = csv.reader(data_file, delimiter=' ')
        names, cervical, lumbar = [], [], []
        for row in results_enlargements:
            names.append(row[0])

            # cervical
            label_level = centerlines[row[0]].regions_labels[str(int(float(row[1])))]
            closest = centerlines[row[0]].get_closest_to_absolute_position(vertebral_level=label_level, relative_position=float(row[1])-int(float(row[1])))
            cervical.append(centerlines[row[0]].dist_points[closest])

            # lumbar
            label_level = centerlines[row[0]].regions_labels[str(int(float(row[2])))]
            closest = centerlines[row[0]].get_closest_to_absolute_position(vertebral_level=label_level, relative_position=float(row[2])-int(float(row[2])))
            lumbar.append(centerlines[row[0]].dist_points[closest])
        cervical_series = pd.Series(cervical, index=names)
        lumbar_series = pd.Series(lumbar, index=names)

    mu_cervical = cervical_series.mean()
    sigma_cervical = cervical_series.std()
    variance_cervical = sigma_cervical**2
    mu_lumbar = lumbar_series.mean()
    sigma_lumbar = lumbar_series.std()
    variance_lumbar = sigma_lumbar**2
    x = np.linspace(mu_cervical - 3 * variance_cervical, mu_cervical + 3 * variance_cervical, 5 * variance_cervical)
    plt.plot(x, mlab.normpdf(x, mu_cervical, sigma_cervical), 'r')
    x = np.linspace(mu_lumbar - 3 * variance_lumbar, mu_lumbar + 3 * variance_lumbar, 5 * variance_lumbar)
    plt.plot(x, mlab.normpdf(x, mu_lumbar, sigma_lumbar), 'r')
    print 'cervical enlargement', mu_cervical, sigma_cervical
    print 'lumbar enlargement', mu_lumbar, sigma_lumbar

    plt.xlim(0, 600)
    plt.show()

    # correlation between enlargements
    df_enl = pd.DataFrame({'cervical': cervical_series, 'lumbar': lumbar_series})
    print df_enl

    from scipy.stats.stats import pearsonr
    import seaborn

    corrcoef, pvalue_corr = pearsonr(cervical, lumbar)
    print 'correlation enlargements', corrcoef, pvalue_corr
    plt.figure()
    seaborn.set_style("whitegrid")
    #scat1 = seaborn.regplot(x='cervical', y='lumbar', fit_reg=True, data=df_enl)
    plt.plot(cervical, lumbar, 'o')
    plt.xlabel('Cervical enlargement position from PMJ')
    plt.ylabel('Lumbar enlargement position from PMJ')
    plt.show()

    # correlation between enlargements and vertebral levels
    # cervical enlargement and C4-C5 intervertebral disk
    df_cervical = pd.DataFrame({'cervical': cervical_series, 'C4-C5': C4C5_levels})
    corrcoef, pvalue_corr = pearsonr(np.array(cervical_series.tolist()), np.array(C4C5_levels.tolist()))
    print 'correlation cervical vs C4-C5', corrcoef, pvalue_corr
    plt.figure()
    seaborn.set_style("whitegrid")
    #scat1 = seaborn.regplot(x='cervical', y='C4-C5', fit_reg=True, data=df_cervical)
    plt.plot(cervical_series.tolist(), C4C5_levels.tolist(), 'o')
    plt.xlabel('Distance between PMJ and cervical enlargement [mm]')
    plt.ylabel('Distance between PMJ and C4-C5 intervertebral disk [mm]')
    plt.show()

    # lumbar enlargement and T12-L1 intervertebral disk
    df_lumbar = pd.DataFrame({'lumbar': lumbar_series, 'T11-T12': T11T12_levels})
    corrcoef, pvalue_corr = pearsonr(lumbar_series.tolist(), T11T12_levels.tolist())
    print 'correlation lumbar vs T11-T12', corrcoef, pvalue_corr
    plt.figure()
    seaborn.set_style("whitegrid")
    #scat1 = seaborn.regplot(x='lumbar', y='T11-T12', fit_reg=True, data=df_lumbar)
    plt.plot(lumbar_series.tolist(), T11T12_levels.tolist(), 'o')
    plt.xlabel('Distance between PMJ and lumbar enlargement [mm]')
    plt.ylabel('Distance between PMJ and T11-T12 intervertebral disk [mm]')
    plt.show()


def display_csa_length():
    import pandas as pd
    import matplotlib.pyplot as plt
    df_levels = pd.read_csv('/Users/benjamindeleener/data/PAM50_2017/output/levels.txt', index_col=0)
    df_levels['avg'] = df_levels.mean(axis=1)
    df_levels['std'] = df_levels.std(axis=1)
    levels_position = []
    for index, row in df_levels['avg'].iteritems():
        levels_position.append([labels_regions[index], row])
    levels_position.sort(key=lambda x: x[1])
    print levels_position
    from scipy.interpolate import interp1d
    x_l = [l[0] for l in levels_position]
    y_l = [l[1] for l in levels_position]
    f = interp1d(x_l, y_l, fill_value=0.0)

    labels = [regions_labels[str(l[0])] for l in levels_position]

    import json
    with open(PATH_OUTPUT + 'csa.txt') as data_file:
        results_csa = json.load(data_file)

    plt.figure()
    for i, subject in enumerate(results_csa):
        ax = plt.subplot(10, 5, i + 1)
        x, y = results_csa[subject][0], results_csa[subject][1]
        #y_smooth = y
        y_smooth = smooth(np.array(results_csa[subject][1]), window_len=20)
        plt.plot(f(x), y_smooth)
        plt.title(subject)
        plt.xlim((0, 550))
        plt.xticks(y_l, labels, rotation='horizontal')
    plt.show()

    path_template = os.path.join(path_data_new, 'output', 'PAM50')
    levels_template, csa_template = compute_csa(os.path.join(path_template, 'PAM50_seg.nii.gz'),
                                                os.path.join(path_template, 'PAM50_disks.nii.gz'),
                                                os.path.join(path_template, 'PAM50_centerline.nii.gz'))

    enlargements = {'FR': [4.65782333434, 17.8022309316],
'JW': [ 5.64666867651, 19.3457943925],
'VC': [ 4.99547784142, 19.2252034971],
'HB': [ 4.56135061803, 18.1640036177],
'pain_pilot_7': [ 4.73017787157, 19.4181489298],
'pain_pilot_4': [ 4.60958697618, 19.0804944227],
'pain_pilot_2': [ 4.82665058788, 19.6834488996],
'pain_pilot_3': [ 5.33313234851, 19.5387398251],
'pain_pilot_1': [ 5.2366596322, 18.6946035574],
'MT': [ 4.99547784142, 19.490503467],
'ED': [ 5.26077781128, 19.5146216461],
'errsm_23': [ 4.34428700633, 19.4181489298],
'errsm_20': [ 4.34428700633, 19.2493216762],
'PA': [ 4.29605064818, 19.2252034971],
'errsm_09': [ 4.41664154356, 19.7558034368],
'errsm_24': [ 4.68194151342, 19.3699125716],
'errsm_25': [ 4.97135966235, 18.7187217365],
'errsm_04': [ 4.89900512511, 18.863430811],
'errsm_05': [ 4.82665058788, 19.5146216461],
'errsm_03': [ 5.04371419958, 19.1287307808],
'T045': [ 4.00663249925, 19.5628580042],
'errsm_43': [ 4.44075972264, 19.0563762436],
'errsm_44': [ 4.22369611094, 18.7187217365],
'VG': [ 4.29605064818, 18.9599035273],
'errsm_21': [ 4.75429605065, 19.8763943322],
'errsm_18': [ 4.85076876696, 19.3940307507],
'AM': [ 4.56135061803, 19.0081398854],
'MLL': [ 4.392523364494, 19.68344889965],
'VP': [ 4.29605064818, 18.9599035273],
'AP': [ 5.0195960205, 19.3699125716],
'GB': [ 4.70605969249, 18.6222490202],
'sct_001': [ 4.65782333434, 19.3699125716],
'sct_002': [ 4.56135061803, 18.9116671691],
'T047': [ 4.53723243895, 18.9116671691],
'errsm_35': [ 4.53723243895, 18.8875489901],
'errsm_34': [ 4.73017787157, 19.6110943624],
'errsm_37': [ 4.82665058788, 19.1287307808],
'errsm_36': [ 4.00663249925, 17.1269219174],
'errsm_31': [ 5.18842327404, 17.7057582153],
'errsm_30': [ 4.68194151342, 19.8763943322],
'errsm_33': [ 4.44075972264, 18.1398854386],
'errsm_32': [ 4.32016882725, 19.9246306904],
'errsm_17': [ 4.65782333434, 19.1287307808],
'errsm_16': [ 4.85076876696, 19.6593307205],
'ALT': [ 4.70605969249, 19.4422671088],
'errsm_14': [ 4.22369611094, 18.1398854386],
'errsm_13': [ 4.63370515526, 19.6834488996],
'errsm_12': [5.11606873681, 19.2975580344],
'errsm_11': [4.271932469, 19.1287307808],
'errsm_10': [4.51311425987, 19.5146216461]}

    position_cervical = []
    position_lumbar = []
    csa_cervical = []
    csa_lumbar = []

    plt.figure()
    for i, subject in enumerate(results_csa):
        index_min_cervical = np.argmin(abs(np.array(results_csa[subject][0]) - enlargements[subject][0]))
        index_min_lumbar = np.argmin(abs(np.array(results_csa[subject][0]) - enlargements[subject][1]))
        print subject, index_min_cervical, index_min_lumbar
        print enlargements[subject][0], results_csa[subject][0][index_min_cervical], f(results_csa[subject][0][index_min_cervical])
        print enlargements[subject][1], results_csa[subject][0][index_min_lumbar], f(results_csa[subject][0][index_min_lumbar])

        csa_cervical.append(results_csa[subject][1][index_min_cervical])
        position_cervical.append(f(results_csa[subject][0][index_min_cervical]))
        csa_lumbar.append(results_csa[subject][1][index_min_lumbar])
        position_lumbar.append(f(results_csa[subject][0][index_min_lumbar]))
        x, y = results_csa[subject][0], results_csa[subject][1]
        #y_smooth = y
        y_smooth = smooth(np.array(results_csa[subject][1]), window_len=20)
        plt.plot(f(x), y_smooth)
    #plt.legend([subject for subject in results_csa])
    plt.xticks(y_l, labels, rotation='horizontal')
    plt.xlim((levels_position[2][1], 500))
    plt.grid()
    plt.show()

    print 'cervical', np.mean(csa_cervical), np.std(csa_cervical), np.mean(position_cervical), np.std(position_cervical)
    print 'lumbar', np.mean(csa_lumbar), np.std(csa_lumbar), np.mean(position_lumbar), np.std(position_lumbar)

    # AVERAGE CSA
    number_points = 1000
    x_new = np.linspace(levels_position[2][1], 500, num=number_points)
    csa_average = []
    for i, subject in enumerate(results_csa):
        x, y = results_csa[subject][0], results_csa[subject][1]
        f_average = interp1d(f(x), y, bounds_error=False, fill_value=0.0)
        csa_average.append(f_average(x_new))

    mean = np.mean(csa_average, axis=0)
    std = np.std(csa_average, axis=0)

    plt.figure()
    plt.plot(x_new, mean, color='b', linewidth=3)
    plt.fill_between(x_new, mean + std, mean - std, facecolor='blue', alpha=0.2)

    csa_smooth = smooth(np.array(csa_template), window_len=10)
    plt.plot(f(levels_template), csa_smooth, color='r', linewidth=3)

    plt.xticks(y_l, labels, rotation='horizontal')
    plt.xlim((levels_position[2][1], 500))
    plt.grid()
    plt.show()

    interp_mean = interp1d(x_new, mean, bounds_error=False, fill_value=0.0)
    interp_std = interp1d(x_new, std, bounds_error=False, fill_value=0.0)
    for i, l in enumerate(y_l):
        print labels[i] + ' ' + str(round(interp_mean(l), 2)) + ' +- ' + str(round(interp_std(l), 2))


def select_enlargements():
    import json
    with open(PATH_OUTPUT + 'csa.txt') as data_file:
        results_csa = json.load(data_file)

    import matplotlib.pyplot as plt
    for i, subject in enumerate(results_csa):
        def onclick(event):
            print subject, event.xdata, event.ydata

        fig = plt.figure()
        x, y = results_csa[subject][0], results_csa[subject][1]
        # y_smooth = y
        y_smooth = smooth(np.array(results_csa[subject][1]), window_len=20)
        plt.plot(x, y_smooth)
        plt.title(subject)
        plt.xlim((0, 26))
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()


def validate_centerline():
    import pandas as pd

    compute = False
    if compute:
        results = {}
        results_array = []
        result_mean_t1, result_mean_t2 = [], []
        result_max_t1, result_max_t2 = [], []

        fname_template_centerline_image = '/Users/benjamindeleener/data/PAM50_2017/output/template_centerline.nii.gz'
        fname_template_centerline = '/Users/benjamindeleener/data/PAM50_2017/output/final/centerline.npz'
        path_segmentations = '/Users/benjamindeleener/data/PAM50_2017/output/PAM50/seg/'

        folder_output = path_segmentations
        print '\nValidate centerlines ' + folder_output
        os.chdir(folder_output)

        # open centerline from template
        number_of_points_in_centerline = 4000
        if os.path.isfile(fname_template_centerline):
            centerline_template = Centerline(fname=fname_template_centerline)
        else:
            x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline(
                fname_template_centerline_image, algo_fitting='nurbs', verbose=0, nurbs_pts_number=number_of_points_in_centerline,
                all_slices=False, phys_coordinates=True, remove_outliers=True)
            centerline_template = Centerline(x_centerline_fit, y_centerline_fit, z_centerline,
                                             x_centerline_deriv, y_centerline_deriv, z_centerline_deriv)

            centerline_template.save_centerline(fname_output=fname_template_centerline)

        results_centerline = pd.DataFrame()
        for contrast in ['t2', 't1']:
            for subject_name in list_subjects:
                fname_image = subject_name + '_' + contrast + '_seg_t.nii.gz'

                fname_centerline = subject_name + '_' + contrast + '_centerline.npz'
                if os.path.isfile(fname_centerline):
                    centerline = Centerline(fname=fname_centerline)
                else:

                    x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline(
                        fname_image, algo_fitting='nurbs', verbose=0, nurbs_pts_number=number_of_points_in_centerline,
                        all_slices=False, phys_coordinates=True, remove_outliers=True)

                    centerline = Centerline(x_centerline_fit, y_centerline_fit, z_centerline,
                                            x_centerline_deriv, y_centerline_deriv, z_centerline_deriv)

                    centerline.save_centerline(fname_output=subject_name + '_' + contrast + '_centerline')

                mse, mean, std, max, distances = centerline.compare_centerline(other=centerline_template, reference_image=Image(fname_image))
                df_sub = pd.DataFrame([[d, subject_name, contrast] for d in distances], columns=['distance', 'subject', 'contrast'])
                if results_centerline.empty:
                    results_centerline = df_sub
                else:
                    results_centerline = pd.concat([results_centerline, df_sub])
                results[subject_name] = distances
                results_array.append(distances)
                if contrast == 't1':
                    result_mean_t1.append(mean)
                    result_max_t1.append(max)
                else:
                    result_mean_t2.append(mean)
                    result_max_t2.append(max)
                print subject_name, mse, mean, std, max

        print 'T1 mean =', str(round(np.average(result_mean_t1), 2)) + ' +- ' + str(round(np.std(result_mean_t1), 2))
        print 'T1 max =', str(round(np.average(result_max_t1), 2)) + ' +- ' + str(round(np.std(result_max_t1), 2))
        print 'T2 mean =', str(round(np.average(result_mean_t2), 2)) + ' +- ' + str(round(np.std(result_mean_t2), 2))
        print 'T2 max =', str(round(np.average(result_max_t2), 2)) + ' +- ' + str(round(np.std(result_max_t2), 2))

        results_mean_max = pd.DataFrame({'mean_t1': result_mean_t1, 'max_t1': result_max_t1,
                                         'mean_t2': result_mean_t2, 'max_t2': result_max_t2},
                                        index=list_subjects)
        results_mean_max.to_pickle('/Users/benjamindeleener/data/PAM50_2017/output/PAM50/accuracy_centerline_mean_max.pkl')
        results_centerline.to_pickle('/Users/benjamindeleener/data/PAM50_2017/output/PAM50/accuracy_centerline.pkl')
    else:
        results_mean_max = pd.read_pickle('/Users/benjamindeleener/data/PAM50_2017/output/PAM50/accuracy_centerline_mean_max.pkl')
        results_centerline = pd.read_pickle('/Users/benjamindeleener/data/PAM50_2017/output/PAM50/accuracy_centerline.pkl')


    import seaborn as sns
    sns.set_style("whitegrid")
    """
    plt.figure()
    plt.subplot(5, 1, 1)
    data_subplot = results_centerline.loc[results_centerline['subject'].isin(list_subjects[0:10])]
    ax = sns.violinplot(x='subject', y='distance', hue='contrast', data=data_subplot, palette="muted", split=True)
    plt.subplot(5, 1, 2)
    data_subplot = results_centerline.loc[results_centerline['subject'].isin(list_subjects[10:20])]
    ax = sns.violinplot(x='subject', y='distance', hue='contrast', data=data_subplot, palette="muted", split=True)
    plt.subplot(5, 1, 3)
    data_subplot = results_centerline.loc[results_centerline['subject'].isin(list_subjects[20:30])]
    ax = sns.violinplot(x='subject', y='distance', hue='contrast', data=data_subplot, palette="muted", split=True)
    plt.subplot(5, 1, 4)
    data_subplot = results_centerline.loc[results_centerline['subject'].isin(list_subjects[30:40])]
    ax = sns.violinplot(x='subject', y='distance', hue='contrast', data=data_subplot, palette="muted", split=True)
    plt.subplot(5, 1, 5)
    data_subplot = results_centerline.loc[results_centerline['subject'].isin(list_subjects[40:])]
    ax = sns.violinplot(x='subject', y='distance', hue='contrast', data=data_subplot, palette="muted", split=True)
    plt.ylim(0, 2)
    plt.show()
    """

    print results_mean_max.describe()

    plt.figure()
    plt.subplot(1, 2, 1)
    print results_mean_max
    data_subplot = results_mean_max[['mean_t1', 'mean_t2']]
    ax = sns.violinplot(data=data_subplot, palette="muted")
    plt.ylim(0, 0.75)
    plt.subplot(1, 2, 2)
    data_subplot = results_mean_max[['max_t1', 'max_t2']]
    ax = sns.violinplot(data=data_subplot, palette="muted")
    plt.ylim(0, 2.5)
    plt.show()


def convert_segmentations():
    path_data = '/Users/benjamindeleener/data/PAM50_2017/output/PAM50/out/'
    path_out = '/Users/benjamindeleener/data/PAM50_2017/output/PAM50/seg/'
    fname_template = '/Users/benjamindeleener/data/PAM50_2017/output/PAM50/PAM50_t1.nii'
    for subject_name in list_subjects:
        sct.run('mnc2nii ' + os.path.join(path_data, subject_name + '_t1_seg_d.mnc') + ' ' + os.path.join(path_data, subject_name + '_t1_seg_d.nii'))

    for subject_name in list_subjects:
        sct.run('sct_crop_image -i ' + os.path.join(path_data, subject_name + '_t1_seg_d.nii') + ' -o ' + os.path.join(path_out, subject_name + '_t1_seg_t.nii.gz') + ' -b 0 -start 0 -end 1047 -dim 2')
        #sct.run('fslview -m single,single ' + fname_template + ' ' + os.path.join(path_data, subject_name + '_t1_seg_d.nii') + ' -l Red -b 0,0.00001')


def warp_centerline():
    path_data = '/Users/benjamindeleener/data/PAM50_2017/'
    path_out = '/Users/benjamindeleener/data/PAM50_2017/output/PAM50/seg/'
    contrast = 't1'
    image_destination = '/Users/benjamindeleener/data/PAM50_2017/output/PAM50/PAM50_t1.nii'
    for subject_name in list_subjects:
        path_subject = os.path.join(path_data, subject_name, contrast)
        fname_centerline = os.path.join(path_subject, contrast + '_centerline_manual.nii.gz')
        sct.run('sct_apply_transfo -i ' + fname_centerline + ' -w ' + os.path.join(path_subject, 'warp_curve2straight.nii.gz') + ' -d ' + image_destination + ' -x nn -o ' + os.path.join(path_out, subject_name + '_' + contrast + '_centerline_straight.nii.gz'))
        sct.run('sct_crop_image -i ' + os.path.join(path_out, subject_name + '_' + contrast + '_centerline_straight.nii.gz') + ' -o ' + os.path.join(path_out, subject_name + '_' + contrast + '_seg_t.nii.gz') + ' -b 0 -start 200 -end 1047 -dim 2')
        os.remove(os.path.join(path_out, subject_name + '_' + contrast + '_centerline_straight.nii.gz'))


#clean_segmentation('t1')
#multisegment_spinalcord('t1')
#generate_centerline('t1')
#average_centerline('t1')
#straighten_all_subjects('t2')

#compare_csa(contrast='t1', fname_segmentation='t1_seg_manual.nii.gz', fname_disks='t1_ground_truth.nii.gz', fname_centerline_image='t1_centerline_manual.nii.gz')
#compute_spinalcord_length(contrast='t1', fname_segmentation='t1_seg_manual.nii.gz')

#normalize_intensity(contrast='t2', fname_disks=PATH_OUTPUT + 'template_disks.nii.gz', fname_centerline_image=PATH_OUTPUT + 'template_centerline.nii.gz')
#convert_nii2mnc(contrast='t2')
#warp_segmentation('t1')

#create_mask_template()

"""
#folder = '/mnt/parallel_scratch_mp2_wipe_on_august_2017/jcohen/bedelb/template_generation_t1/data/'
folder = '/gs/project/rrp-355-aa/data/'
contrast = 't1'
for subject_name in list_subjects:
    #print folder + subject_name + '_' + contrast + '.mnc,' + folder + subject_name + '_' + contrast + '_seg.mnc'
    print folder + subject_name + '_' + contrast + '.mnc,' + folder + 'template_mask.mnc'
"""

display_csa_length()
#compute_vertebral_levels(contrast='t1')
#select_enlargements()
#validate_centerline()
#normalize_intensity_template()
#convert_segmentations()
#warp_centerline()

"""
for subject_name in list_subjects:
    fname_seg = '/Users/benjamindeleener/data/PAM50_2017/' + subject_name + '/t2/t2_seg.nii.gz'
    fname_out = '/Users/benjamindeleener/data/PAM50_2017/' + subject_name + '/t2/'
    sct.run('sct_process_segmentation -i ' + fname_seg + ' -p centerline -ofolder ' + fname_out)
"""

"""
sct_process_segmentation -i t2_seg.nii.gz -p centerline
fslview -m single,single t2.nii.gz -b 0,700 t2_seg_centerline.nii.gz -l Red &

sct_make_ground_truth.py -i t1.nii.gz -save-as niigz
Produce centerline along spinal cord (above and below segmentation) called {contrast}_centerline_manual.nii.gz
"""
