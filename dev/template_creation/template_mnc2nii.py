#!/usr/bin/env python

path_data = '/mnt/parallel_scratch_mp2_wipe_on_august_2017/jcohen/bedelb/data/'
path_out = '/mnt/parallel_scratch_mp2_wipe_on_august_2017/jcohen/bedelb/template_generation_t1/data/'


# folder to dataset - not useful here
folder_data_errsm = '/Volumes/data_shared/montreal_criugm/errsm'
folder_data_sct = '/Volumes/data_shared/montreal_criugm/sct'
folder_data_marseille = '/Volumes/data_shared/marseille'
folder_data_pain = '/Volumes/data_shared/montreal_criugm/simon'
SUBJECTS_LIST = [
    ['ALT', folder_data_marseille+'/ALT/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-15', folder_data_marseille+'/ALT/01_0100_space-composing'],
    ['errsm_11', folder_data_errsm + '/errsm_11/24-SPINE_T1/echo_2.09', folder_data_errsm + '/errsm_11/09-SPINE_T2'],
    ['errsm_18', folder_data_errsm + '/errsm_18/36-SPINE_T1/echo_2.09', folder_data_errsm + '/errsm_18/33-SPINE_T2'],
    ['MLL', folder_data_marseille+'/MLL_1016/01_0008_sc-mprage-1mm-2palliers-fov384-comp-sp-7', folder_data_marseille+'/MLL_1016/01_0100_t2-compo'],
    ['errsm_03', folder_data_errsm+'/errsm_03/32-SPINE_all/echo_2.09', folder_data_errsm+'/errsm_03/38-SPINE_all_space'],
    ['errsm_14', folder_data_errsm+'/errsm_14/5002-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_14/5003-SPINE_T2'],
    ['errsm_25', folder_data_errsm+'/errsm_25/25-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_25/26-SPINE_T2'],
    ['errsm_37', folder_data_errsm+'/errsm_37/19-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_37/20-SPINE_T2'],
    ['sct_001', folder_data_sct+'/sct_001/17-SPINE_T1/echo_2.09', folder_data_sct+'/sct_001/16-SPINE_T2'],
    ['AM', folder_data_marseille+'/AM/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', folder_data_marseille+'/AM/01_0100_compo-t2-spine'],
    ['MT', folder_data_marseille+'/MT/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', folder_data_marseille+'/MT/01_0100_t2composing'],
    ['errsm_04', folder_data_errsm+'/errsm_04/16-SPINE_memprage/echo_2.09', folder_data_errsm+'/errsm_04/18-SPINE_space'],
    ['errsm_16', folder_data_errsm + '/errsm_16/23-SPINE_T1/echo_2.09', folder_data_errsm + '/errsm_16/39-SPINE_T2'],
    ['errsm_30', folder_data_errsm + '/errsm_30/51-SPINE_T1/echo_2.09', folder_data_errsm + '/errsm_30/50-SPINE_T2'],
    ['errsm_43', folder_data_errsm + '/errsm_43/22-SPINE_T1/echo_2.09', folder_data_errsm + '/errsm_43/18-SPINE_T2'],
    ['sct_002', folder_data_sct + '/sct_002/12-SPINE_T1/echo_2.09', folder_data_sct + '/sct_002/18-SPINE_T2'],
    ['ED', folder_data_marseille+'/ED/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-101', folder_data_marseille+'/ED/01_0008_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-65'],
    ['PA', folder_data_marseille+'/PA/01_0034_sc-mprage-1mm-2palliers-fov384-comp-sp-5', folder_data_marseille+'/PA/01_0038_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-7'],
    ['errsm_05', folder_data_errsm+'/errsm_05/23-SPINE_MEMPRAGE/echo_2.09', folder_data_errsm+'/errsm_05/24-SPINE_SPACE'],
    ['errsm_17', folder_data_errsm+'/errsm_17/41-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_17/42-SPINE_T2'],
    ['errsm_31', folder_data_errsm+'/errsm_31/31-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_31/32-SPINE_T2'],
    ['errsm_44', folder_data_errsm+'/errsm_44/18-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_44/19-SPINE_T2'],
    ['FR', folder_data_marseille+'/FR_T080/01_0039_sc-mprage-1mm-3palliers-fov384-comp-sp-13', folder_data_marseille+'/FR_T080/01_0104_spine2'],
    ['T045', folder_data_marseille+'/T045/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', folder_data_marseille+'/T045/01_0101_t2-3d-composing'],
    ['errsm_09', folder_data_errsm+'/errsm_09/34-SPINE_MEMPRAGE2/echo_2.09', folder_data_errsm+'/errsm_09/33-SPINE_SPACE'],
    ['errsm_32', folder_data_errsm+'/errsm_32/16-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_32/19-SPINE_T2'],
    ['pain_pilot_1', folder_data_pain+'/d_sp_pain_pilot1/24-SPINE_T1/echo_2.09', folder_data_pain+'/d_sp_pain_pilot1/25-SPINE'],
    ['pain_pilot_2', folder_data_pain+'/d_sp_pain_pilot2/13-SPINE_T1/echo_2.09', folder_data_pain+'/d_sp_pain_pilot2/30-SPINE_T2'],
    ['pain_pilot_3', folder_data_pain+'/d_sp_pain_pilot3/16-SPINE_T1/echo_2.09', folder_data_pain+'/d_sp_pain_pilot3/31-SPINE_T2'],
    ['pain_pilot_4', folder_data_pain+'/d_sp_pain_pilot4/33-SPINE_T1/echo_2.09', folder_data_pain+'/d_sp_pain_pilot4/32-SPINE_T2'],
    ['pain_pilot_7', folder_data_pain+'/d_sp_pain_pilot7/32-SPINE_T1/echo_2.09', folder_data_pain+'/d_sp_pain_pilot7/33-SPINE_T2'],
    ['GB', folder_data_marseille+'/GB_T083/01_0029_sc-mprage-1mm-2palliers-fov384-comp-sp-5', folder_data_marseille+'/GB_T083/01_0033_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-7'],
    ['HB', folder_data_marseille+'/HB/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-29', folder_data_marseille+'/HB/01_0100_t2-compo'],
    ['VC', folder_data_marseille+'/VC/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-23', folder_data_marseille+'/VC/01_0008_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-113'],
    ['T047', folder_data_marseille+'/T047/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', folder_data_marseille+'/T047/01_0100_t2-3d-composing'],
    ['JD', folder_data_marseille+'/JD/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-23', folder_data_marseille+'/JD/01_0100_compo-space'],
    ['VG', folder_data_marseille+'/VG/T1/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-15', folder_data_marseille+'/VG/T2/01_0024_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-11'],
    ['JW', folder_data_marseille+'/JW/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', folder_data_marseille+'/JW/01_0100_compo-space'],
    ['VP', folder_data_marseille+'/VP/01_0011_sc-mprage-1mm-2palliers-fov384-comp-sp-25', folder_data_marseille+'/VP/01_0100_space-compo'],
    ['errsm_10', folder_data_errsm+'/errsm_10/13-SPINE_MEMPRAGE/echo_2.09', folder_data_errsm+'/errsm_10/20-SPINE_SPACE'],
    ['errsm_20', folder_data_errsm+'/errsm_20/12-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_20/34-SPINE_T2'],
    ['errsm_33', folder_data_errsm+'/errsm_33/30-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_33/31-SPINE_T2'],
    ['errsm_21', folder_data_errsm+'/errsm_21/27-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_21/30-SPINE_T2'],
    ['errsm_34', folder_data_errsm+'/errsm_34/41-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_34/40-SPINE_T2'],
    ['errsm_12', folder_data_errsm+'/errsm_12/19-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_12/18-SPINE_T2'],
    ['errsm_23', folder_data_errsm+'/errsm_23/29-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_23/28-SPINE_T2'],
    ['errsm_35', folder_data_errsm+'/errsm_35/37-SPINE_T1/echo_2.09', folder_data_errsm+'/errsm_35/38-SPINE_T2'],
    ['errsm_13', folder_data_errsm + '/errsm_13/33-SPINE_T1/echo_2.09', folder_data_errsm + '/errsm_13/34-SPINE_T2'],
    ['errsm_24', folder_data_errsm + '/errsm_24/20-SPINE_T1/echo_2.09', folder_data_errsm + '/errsm_24/24-SPINE_T2'],
    ['errsm_36', folder_data_errsm + '/errsm_36/30-SPINE_T1/echo_2.09', folder_data_errsm + '/errsm_36/31-SPINE_T2']
]

import os
for i in range(0, len(SUBJECTS_LIST)):
    subject = SUBJECTS_LIST[i][0]
    os.system('nii2mnc ' + path_data + 'template_mask.nii.gz ' + path_out + 'template_mask.mnc')
    os.system('nii2mnc ' + path_data + subject + '_final_T1.nii.gz ' + path_out + subject + '_final_T1.mnc')
    os.system('nii2mnc ' + path_data + subject + '_final_T2.nii.gz ' + path_out + subject + '_final_T2.mnc')
