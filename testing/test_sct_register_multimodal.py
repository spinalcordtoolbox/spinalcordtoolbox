#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_register_multimodal script
#
#   replace the shell test script in sct 1.0
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014/09/28
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os
import sct_utils as sct
import test_all
import time
import shutil

class bcolors:
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    red = '\033[91m'

class Param:
    def __init__(self):
        self.contrasts = ['mt']
        self.subjects = []
        self.files = [['mt1.nii.gz', 'segmentation_binary.nii.gz']]


def test(path_data):
    '''
    # initialize parameters
    status = 0
    param = Param()
    results_dir = 'results_sct_register_multimodal'

    if os.path.isdir(results_dir):
        shutil.rmtree(results_dir)

    os.makedirs(results_dir)
    os.chdir(results_dir)

    begin_log_file = "test ran at "+time.strftime("%y%m%d%H%M%S")+"\n"
    fname_log = "sct_register_multimodal.log"

    test_all.write_to_log_file(fname_log, begin_log_file, 'w')

    for contrast in param.contrasts:
        test_all.write_to_log_file(fname_log, bcolors.red+"Contrast: "+contrast+"\n", 'a')
        for f in param.files:
            cmd = 'sct_register_multimodal -i ../' + data_file_path + '/' + contrast + '/label/template/MNI-Poly-AMU_T2.nii.gz' \
                + ' -d ../' + data_file_path + '/' + contrast + '/' + f[0] \
                + ' -s ../' + data_file_path + '/' + contrast + '/label/template/MNI-Poly-AMU_cord.nii.gz' \
                + ' -t ../' + data_file_path + '/' + contrast + '/' + f[1] \
                + ' -q ../' + data_file_path + '/' + contrast + '/label/template/warp_template2anat.nii.gz'\
                + ' -x 0' \
                + ' -o template2' + f[0] \
                + ' -n 10x3' \
                + ' -r 1' \
                + ' -v 1'

            s, output = sct.run(cmd, 0)
            test_all.write_to_log_file(fname_log, cmd + '\n' + output, 'a')
            status += s

    os.chdir('..')
    return status
    '''



    folder_data = 'mt/'
    folder_template = 'template/'
    file_data = ['mt1.nii.gz', 'mt1_seg.nii.gz', 'warp_template2mt.nii.gz']
    file_template = ['MNI-Poly-AMU_T2.nii.gz', 'MNI-Poly-AMU_cord.nii.gz']

    cmd = 'sct_register_multimodal -i ' + path_data + folder_template + file_template[0] \
          + ' -d ' + path_data + folder_data + file_data[0] \
          + ' -s ' + path_data + folder_template + file_template[1] \
          + ' -t ' + path_data + folder_data + file_data[1] \
          + ' -q ' + path_data + folder_data + file_data[2] \
          + ' -x 0' \
          + ' -o template2' + file_data[0] \
          + ' -n 10x3' \
          + ' -r 1' \
          + ' -v 1'

    return sct.run(cmd, 0)


if __name__ == "__main__":
    # call main function
    test()