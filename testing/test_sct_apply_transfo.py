#!/usr/bin/env python
#########################################################################################
#
# Test function sct_apply_transfo
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014/10/30
#
# About the license: see the file LICENSE.TXT
#########################################################################################



def init(param_test):
    """
    Initialize testing.
    Parameters
    ----------
    param_test: Class defined in sct_testing.py

    Returns
    -------
    param_test
    """
    # initialization
    default_args = '-i template/template/PAM50_small_t2.nii.gz -d t2/t2.nii.gz -w t2/warp_template2anat.nii.gz'  # default parameters

    # assign default params
    if not param_test.args:
        param_test.args = default_args

    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    Parameters
    ----------
    param_test: Class defined in sct_testing.py

    Returns
    -------
    param_test
    """
    param_test.output += '\nNot implemented.'
    return param_test
#
#
# def test(data_path):
#
#     # parameters
#     folder_data = [ 'template/template/', 't2/', 'dmri/']
#     file_data = [get_file_label(data_path+'template/template/','T2-weighted'),
#                  't2.nii.gz',
#                  'warp_template2anat.nii.gz',
#                  'dmri.nii.gz']
#
#     output = ''
#     status = 0
#
#     # test function
#     cmd = 'sct_apply_transfo -i ' + data_path + folder_data[0] + file_data[0] \
#           + ' -d ' + data_path + folder_data[1] + file_data[1] \
#           + ' -w ' + data_path + folder_data[1] + file_data[2]
#     output += cmd+'\n'  # copy command
#     s, o = commands.getstatusoutput(cmd)
#     status += s
#     output += o
#
#     # test with 4d input
#     cmd = 'sct_apply_transfo -i ' + data_path + folder_data[2] + file_data[3] \
#           + ' -d ' + data_path + folder_data[1] + file_data[1] \
#           + ' -w ' + data_path + folder_data[1] + file_data[2]
#     output += cmd+'\n'  # copy command
#     s, o = commands.getstatusoutput(cmd)
#     status += s
#     output += o
#
#     # return
#     #return sct.run(cmd, 0)
#     return status, output
#
#
# if __name__ == "__main__":
#     # call main function
#     test()