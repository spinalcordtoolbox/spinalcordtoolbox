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


def test(data_path):

    folder_data = ['mt/', 't2/', 'dmri/']
    file_data = ['mtr.nii.gz', 't2.nii.gz', 'dmri.nii.gz']

    output = ''
    status = 0

    pad = 2
    cmd = 'sct_image -i ' + data_path + folder_data[0] + file_data[0] \
                + ' -o test.nii.gz' \
                + ' -pad 0,0,'+str(pad)
    output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command

    s0, o0 = commands.getstatusoutput(cmd)
    status += s0
    output += o0


    # test 3d data
    cmd = 'sct_image -i ' + data_path + folder_data[1] + file_data[1] + ' -getorient '
    output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
    s1, o1 = commands.getstatusoutput(cmd)

    status += s1
    output += o1

    from time import sleep
    sleep(1)  # here add one second, otherwise the next test will try to create a temporary folder with the same name (because it runs in less than one second)


    # test 4d data
    if status == 0:
        cmd = 'sct_image -i ' + data_path + folder_data[2] + file_data[2] + ' -getorient '
        output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
        s2, o2 = commands.getstatusoutput(cmd)

        status += s2
        output += o2

    if s0 == 0:
        from msct_image import Image
        nx, ny, nz, nt, px, py, pz, pt = Image(data_path + folder_data[0] + file_data[0]).dim
        nx2, ny2, nz2, nt2, px2, py2, pz2, pt2 = Image("test.nii.gz").dim

        if nz2 != nz+2*pad:
            status = 99
            output += '\nResulting pad image\'s dimension differs from expected:\n'
            output += 'dim : '+ str(nx2) +'x'+ str(ny2) +'x'+ str(nz2)+'\n'
            output += 'expected : '+ str(nx) +'x'+ str(ny) +'x'+ str(nz+2*pad)+'\n'
    if s1 == 0:
        if o1 != "AIL":
            status = 99
            output += '\nResulting orientation differs from expected:\n'
    if s2 == 0:
        if o2 != "RPI":
            status = 99
            output += '\nResulting orientation differs from expected:\n'

    return status, output

if __name__ == "__main__":
    # call main function
    test()