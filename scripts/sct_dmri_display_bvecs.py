#!/usr/bin/env python
#########################################################################################
#
# Display bvecs
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sys
from msct_parser import Parser
from dipy.data.fetcher import read_bvals_bvecs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sct_utils as sct

bzero = 0.0001  # b-zero threshold


# PARSER
# ==========================================================================================
def get_parser():

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Display scatter plot of gradient directions from bvecs file.')
    parser.add_option(name='-bvec',
                      type_value='file',
                      description='bvecs file.',
                      mandatory=True,
                      example='bvecs.txt')
    return parser


def plot_2dscatter(fig_handle=None, subplot=None, x=None, y=None, xlabel='X', ylabel='Y'):
    ax = fig_handle.add_subplot(subplot, aspect='equal')
    for i in range(0, len(x)):
        # if b=0, do not plot
        if not(abs(x[i]) < bzero and abs(x[i]) < bzero):
            ax.scatter(x[i], y[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.grid()

# MAIN
# ==========================================================================================
def main():

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    fname_bvecs = arguments['-bvec']

    # Read bvecs
    bvecs = read_bvals_bvecs(fname_bvecs, None)
    bvecs = bvecs[0]
    x, y, z = bvecs[0], bvecs[1], bvecs[2]

    # Get total number of directions
    n_dir = len(x)

    # Get effective number of directions
    bvecs_eff = []
    n_b0 = 0
    for i in range(0, n_dir):
        add_direction = True
        # check if b=0
        if abs(x[i]) < bzero and abs(x[i]) < bzero and abs(x[i]) < bzero:
            n_b0 += 1
            add_direction = False
        else:
            # loop across bvecs_eff
            for j in range(0, len(bvecs_eff)):
                # if bvalue already present, then do not add to bvecs_eff
                if bvecs_eff[j] == [x[i], y[i], z[i]]:
                    add_direction = False
        if add_direction:
            bvecs_eff.append([x[i], y[i], z[i]])
    n_dir_eff = len(bvecs_eff)

    # Display scatter plot
    fig = plt.figure(facecolor='white', figsize=(9, 8))
    fig.suptitle('Number of b=0: '+str(n_b0)+', Number of b!=0: '+str(n_dir-n_b0)+', Number of effective directions (without duplicates): '+str(n_dir_eff))
    # plt.ion()

    # Display three views
    plot_2dscatter(fig_handle=fig, subplot=221, x=bvecs[0][:], y=bvecs[1][:], xlabel='X', ylabel='Y')
    plot_2dscatter(fig_handle=fig, subplot=222, x=bvecs[0][:], y=bvecs[2][:], xlabel='X', ylabel='Z')
    plot_2dscatter(fig_handle=fig, subplot=223, x=bvecs[1][:], y=bvecs[2][:], xlabel='Y', ylabel='Z')

    # 3D
    ax = fig.add_subplot(224, projection='3d')
    # ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])
    for i in range(0, n_dir):
        # x, y, z = bvecs[0], bvecs[1], bvecs[2]
        # if b=0, do not plot
        if not(abs(x[i]) < bzero and abs(x[i]) < bzero and abs(x[i]) < bzero):
            ax.scatter(x[i], y[i], z[i])
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    plt.title('3D view (use mouse to rotate)')
    plt.axis('off')
    # plt.draw()

    # Save image
    sct.printv("Saving figure: bvecs.png\n")
    plt.savefig('bvecs.png')
    plt.show()


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.start_stream_logger()
    # call main function
    main()
