#!/usr/bin/env python

from msct_gmseg_utils import *

path = '.'

for subject_dir in os.listdir(path):
    subject_path = path + '/' + subject_dir
    if os.path.isdir(subject_path):
        subject_andyy_seg = ''
        subject_sc_seg = ''

        for file_name in os.listdir(subject_path):
            if 'andyy' in file_name.lower():
                subject_andyy_seg = file_name
            if '_seg_corrected' in file_name:
                subject_sc_seg = file_name

        andyy_seg_im = Image(subject_path + '/' + subject_andyy_seg)
        name_andy_seg = subject_dir + '_andyy_manual_gmseg'
        inverse_gm_seg = inverse_wmseg_to_gmseg(andyy_seg_im.data, name_andy_seg)

        subject_sc_seg_im = Image(subject_path + '/' + subject_sc_seg)

        gm_seg = np.absolute(inverse_gm_seg - subject_sc_seg_im.data)
        gm_seg = (gm_seg == 0).astype(int)
        res_gm_seg_im = Image(param=gm_seg, absolutepath=subject_path + '/' + subject_dir + '_andyy_manual_gmseg.nii.gz')
        res_gm_seg_im.save()