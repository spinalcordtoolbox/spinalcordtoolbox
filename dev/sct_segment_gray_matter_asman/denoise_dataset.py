#!/usr/bin/env python

from msct_gmseg_utils import *

'''
path = '.'
original_path = os.getcwd() + path
for subject_dir in os.listdir(path):
    if os.path.isdir(path + '/' + subject_dir):
        subject_path = path + '/' + subject_dir
        os.chdir(subject_path + '/')
        if subject_dir + '_t2star_denoised.nii.gz' not in os.listdir('.'):
            print 'denoising ...'
            sct.run('sct_denoising_onlm.py -i ' + subject_dir + '_t2star.nii.gz')
        else:
            print 'already denoised'
        os.chdir('../')
'''

path = '.'
original_path = os.getcwd() + path
for subject_dir in os.listdir(path):
    if os.path.isdir(path + '/' + subject_dir):
        subject_path = path + '/' + subject_dir
        os.chdir(subject_path + '/')
        if subject_dir + '_im_denoised.nii.gz' not in os.listdir('.'):
            print 'denoising ...'
            sct.run('sct_denoising_onlm.py -i ' + subject_dir + '_im.nii.gz')
        else:
            print 'already denoised'
        os.chdir('../')