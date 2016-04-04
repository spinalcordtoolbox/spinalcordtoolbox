#!/usr/bin/env python
#########################################################################################
#
# Test function sct_image
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

    # TEST PADDING
    pad = 2
    cmd = 'sct_image -i ' + data_path + folder_data[0] + file_data[0] \
                + ' -o test.nii.gz' \
                + ' -pad 0,0,'+str(pad)
    output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
    s0, o0 = commands.getstatusoutput(cmd)
    status += s0
    output += o0

    ## Removed orientation test: see issue #765 for guidelines of what to do to put it back
    # TEST ORIENTATION
    # test 3d data
    cmd = 'sct_image -i ' + data_path + folder_data[1] + file_data[1] + ' -getorient '
    output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
    s1, o1 = commands.getstatusoutput(cmd)
    status += s1
    output += o1

    from time import sleep
    sleep(1)  # here add one second, otherwise the next test will try to create a temporary folder with the same name (because it runs in less than one second)

    # test 4d data
    cmd = 'sct_image -i ' + data_path + folder_data[2] + file_data[2] + ' -getorient '
    output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
    s2, o2 = commands.getstatusoutput(cmd)
    status += s2
    output += o2

    # TEST SPLIT DATA
    cmd = 'sct_image -i '+ data_path + folder_data[2] + file_data[2] +' -split t'
    output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
    s3, o3 = commands.getstatusoutput(cmd)
    status += s3
    output += o3

    # TEST CONCATENATE DATA
    l = file_data[2].split('.')
    file_name = l[0]
    ext = '.'+'.'.join(l[1:])
    dmri_t_slices = [data_path+folder_data[2]+file_name+'_T'+str(i).zfill(4)+ext for i in range(7)]
    input_concat = ','.join(dmri_t_slices)
    cmd = 'sct_image -i '+input_concat+' -concat t -o dmri_concat.nii.gz'
    output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
    s4, o4 = commands.getstatusoutput(cmd)
    status += s4
    output += o4

    # INTEGRITY CHECKS
    if s0 == 0:
        from msct_image import Image
        nx, ny, nz, nt, px, py, pz, pt = Image(data_path + folder_data[0] + file_data[0]).dim
        nx2, ny2, nz2, nt2, px2, py2, pz2, pt2 = Image("test.nii.gz").dim

        if nz2 != nz+2*pad:
            status = 99
            output += '\nResulting pad image\'s dimension differs from expected:\n'
            output += 'dim : '+ str(nx2) +'x'+ str(ny2) +'x'+ str(nz2)+'\n'
            output += 'expected : '+ str(nx) +'x'+ str(ny) +'x'+ str(nz+2*pad)+'\n'
    '''
    if s1 == 0:
        if o1 != "AIL":
            status = 99
            output += '\nResulting orientation differs from expected:\n' \
                      'orientation: '+o1+'\n' \
                      'expected: AIL'
    if s2 == 0:
        if o2 != "RPI":
            status = 99
            output += '\nResulting orientation differs from expected:\n' \
                      'orientation: '+o2+'\n' \
                      'expected: RPI'
    '''
    if s3 == 0:
        from msct_image import Image
        from numpy import sum
        threshold = 1e-3
        ref = Image(dmri_t_slices[0])
        new = Image(file_name+'_T0000'+ext)
        diff = ref.data-new.data
        if sum(diff) > threshold:
            status = 99
            output += '\nResulting split image differs from gold-standard.\n'

    if s4 == 0:
        from msct_image import Image
        from numpy import sum
        threshold = 1e-3
        ref = Image(data_path + folder_data[2] + file_data[2])
        new = Image('dmri_concat.nii.gz')
        diff = ref.data-new.data
        if sum(diff) > threshold:
            status = 99
            output += '\nResulting concatenated image differs from gold-standard (original dmri image).\n'

    return status, output

if __name__ == "__main__":
    # call main function
    test()