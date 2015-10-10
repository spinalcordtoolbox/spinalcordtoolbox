#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_get_centerline script
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
from msct_image import Image
from sct_get_centerline import ind2sub
import math
import sct_utils as sct
import numpy as np


def test(path_data):

    # parameters
    folder_data = 't2/'
    file_data = ['t2.nii.gz', 't2_centerline_init.nii.gz', 't2_centerline_labels.nii.gz', 't2_seg_manual.nii.gz']

    output = ''
    status = 0

    # define command
    cmd = 'sct_get_centerline -i ' + path_data + folder_data + file_data[0] \
        + ' -method auto' \
        + ' -t t2 ' \
        + ' -v 1'
    output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o
    # small integrity test on scad
    try :
        if status == 0:
            manual_seg = Image(path_data + folder_data + file_data[3])
            centerline_scad = Image(path_data + folder_data + file_data[0])
            centerline_scad.change_orientation()
            manual_seg.change_orientation()

            from scipy.ndimage.measurements import center_of_mass
            # find COM
            iterator = range(manual_seg.data.shape[2])
            com_x = [0 for ix in iterator]
            com_y = [0 for iy in iterator]

            for iz in iterator:
                com_x[iz], com_y[iz] = center_of_mass(manual_seg.data[:, :, iz])
            max_distance = {}
            distance = {}
            for iz in range(1, centerline_scad.data.shape[2]-1):
                ind1 = np.argmax(centerline_scad.data[:, :, iz])
                X,Y = ind2sub(centerline_scad.data[:, :, iz].shape,ind1)
                com_phys = np.array(manual_seg.transfo_pix2phys([[com_x[iz], com_y[iz], iz]]))
                scad_phys = np.array(centerline_scad.transfo_pix2phys([[X, Y, iz]]))
                distance_magnitude = np.linalg.norm([com_phys[0][0]-scad_phys[0][0], com_phys[0][1]-scad_phys[0][1], 0])
                if math.isnan(distance_magnitude):
                    print "Value is nan"
                else:
                    distance[iz] = distance_magnitude

            max_distance = max(distance.values())
            #if max_distance > 5:
                #sct.printv("Max distance between scad and manual centerline is greater than 5 mm", type="warning")

    except Exception, e:
        sct.printv("Exception found while testing scad integrity")
        sct.printv(e.message, type="error")

    # define command: DOES NOT RUN IT BECAUSE REQUIRES FSL FLIRT
    # cmd = 'sct_get_centerline -i ' + path_data + folder_data + file_data[0] \
    #     + ' -method point' \
    #     + ' -p ' + path_data + folder_data + file_data[1] \
    #     + ' -g 1'\
    #     + ' -k 4'
    # output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
    # s, o = commands.getstatusoutput(cmd)
    # status += s
    # output += o

    # define command
    cmd = 'sct_get_centerline -i ' + path_data + folder_data + file_data[0] \
        + ' -method labels ' \
        + ' -l ' + path_data + folder_data + file_data[2] \
        + ' -v 1'
    output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o

    return status, output


if __name__ == "__main__":
    # call main function
    test()