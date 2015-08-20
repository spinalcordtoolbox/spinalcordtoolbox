#!/usr/bin/env python
#########################################################################################
#
# Test function sct_fmri_moco
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
# modified: 2014-10-06
#
# About the license: see the file LICENSE.TXT
#########################################################################################

#import sct_utils as sct
import commands


def test(path_data):

    folder_data = 'fmri/'
    file_data = ['fmri.nii.gz']


    cmd = 'sct_fmri_moco -i ' + path_data + folder_data + file_data[0] \
                + ' -g 5' \
                + ' -x spline' \
                + ' -r 0' \
                + ' -v 2'

    # return
    #return sct.run(cmd, 0)
    return commands.getstatusoutput(cmd)


if __name__ == "__main__":
    # call main function
    test()