#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_sctraighten_spinalcord script
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

import commands

import sct_utils as sct


def test(path_data):

    folder_data = 't2/'
    file_data = ['t2.nii.gz', 't2_seg.nii.gz']

    cmd = 'sct_straighten_spinalcord -i '+ path_data + folder_data + file_data[0] \
          + ' -c ' + path_data + folder_data + file_data[1] \
          + ' -r 0' \
          + ' -v 1'
    # return sct.run(cmd, 0)
    return commands.getstatusoutput(cmd)


if __name__ == "__main__":
    # call main function
    test()