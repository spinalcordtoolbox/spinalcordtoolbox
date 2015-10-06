#!/usr/bin/env python
#########################################################################################
#
# Test function sct_fmri_moco
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Sara Dupont
# modified: 2015-10-06
#
# About the license: see the file LICENSE.TXT
#########################################################################################

#import sct_utils as sct
import commands


def test(path_data):

    folder_data = 'mt/'
    file_data = ['mtr.nii.gz']

    output = ''
    status = 0

    pad = 2
    cmd = 'sct_image -i ' + path_data + folder_data + file_data[0] \
                + ' -o test.nii.gz' \
                + ' -pad 0,0,'+str(pad)
    output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command

    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o

    if status == 0:
        from msct_image import Image
        nx, ny, nz, nt, px, py, pz, pt = Image(path_data + folder_data + file_data[0]).dim
        nx2, ny2, nz2, nt2, px2, py2, pz2, pt2 = Image("test.nii.gz").dim

        if nz2 != nz+2*pad:
            status = 99
            output += '\nResulting image\'s dimension differs from expected:\n'
            output += 'dim : '+ str(nx2) +'x'+ str(ny2) +'x'+ str(nz2)+'\n'
            output += 'expected : '+ str(nx) +'x'+ str(ny) +'x'+ str(nz+2*pad)+'\n'

    return status, output

if __name__ == "__main__":
    # call main function
    test()