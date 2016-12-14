#!/usr/bin/env python
#########################################################################################
#
# Test function sct_resample
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
# modified: 2014-10-10
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sct_utils as sct
import commands


def test(path_data):

    folder_data = 'fmri/'
    file_data = ['fmri.nii.gz']


    cmd = 'sct_resample -i ' + path_data + folder_data + file_data[0] \
                + ' -f 0.5x0.5x1' \
                + ' -v 1'

    # return
    #return sct.run(cmd, 0)
    return commands.getstatusoutput(cmd)


if __name__ == "__main__":
    # call main function
    test()