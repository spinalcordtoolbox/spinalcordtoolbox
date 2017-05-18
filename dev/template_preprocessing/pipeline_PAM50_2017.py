#!/usr/bin/env python

import shutil
import os
import sct_utils as sct
import numpy as np

path_data_old = '/Volumes/data_processing/bdeleener/template/template_preprocessing_final/subjects/'
path_data_new = '/Users/benjamindeleener/data/PAM50_2017/'

list_subjects =['ALT',
                'AM',
                'ED',
                'FR',
                'GB',
                'HB',
                'JD',
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
                'sct_002']


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


def multisegment_spinalcord(contrast):
    num_initialisation = 10
    timer_segmentation = sct.Timer(num_initialisation * len(list_subjects))
    timer_segmentation.start()

    initialisation_range = np.linspace(0, 1, num_initialisation + 2)[1:-1]
    for subject_name in list_subjects:
        folder_output = path_data_new + subject_name + '/' + contrast + '/'
        list_files = [folder_output + contrast + '_seg' + str(i) + '.nii.gz' for i in range(len(initialisation_range))]

        for i, init in enumerate(initialisation_range):
            cmd_propseg = 'sct_propseg -i ' + path_data_new + subject_name + '/' + contrast + '/' + contrast + '.nii.gz -c ' + contrast + ' -init ' + str(init) + ' -ofolder ' + folder_output
            if i != 0:
                cmd_propseg += ' -init-centerline ' + folder_output + contrast + '_centerline_optic.nii.gz'
            sct.run(cmd_propseg, verbose=0)
            os.rename(folder_output + contrast + '_seg.nii.gz', list_files[i])

            timer_segmentation.add_iteration()

        sct.run('sct_image -i ' + ','.join(list_files) + ' -concat t -o ' + folder_output + contrast + '_multiseg.nii.gz', verbose=0)
        sct.run('seg_LabFusion -in ' + folder_output + contrast + '_multiseg.nii.gz -STAPLE -out ' + folder_output + contrast + '_seg.nii.gz', verbose=0)


def segment_spinalcord(contrast):
    timer_segmentation = sct.Timer(len(list_subjects))
    timer_segmentation.start()
    for subject_name in list_subjects:
        sct.run('sct_propseg -i ' + path_data_new + subject_name + '/' + contrast + '/' + contrast + '.nii.gz -c t1 '
                '-ofolder ' + path_data_new + subject_name + '/' + contrast + '/', verbose=0)

        timer_segmentation.add_iteration()

multisegment_spinalcord('t2')
#segment_spinalcord('t2')
