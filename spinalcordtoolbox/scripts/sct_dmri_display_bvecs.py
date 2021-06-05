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

import os
import sys

import matplotlib.pyplot as plt
from dipy.data.fetcher import read_bvals_bvecs

from spinalcordtoolbox.utils import SCTArgumentParser, Metavar, init_sct, printv, set_loglevel

bzero = 0.0001  # b-zero threshold


def get_parser():
    parser = SCTArgumentParser(
        description='Display scatter plot of gradient directions from bvecs file.'
    )
    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        '-bvec',
        metavar=Metavar.file,
        required=True,
        help="Input bvecs file. Example: bvecs.txt",
    )
    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit."
    )
    optional.add_argument(
        '-v',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode"
    )

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


def main(argv=None):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)

    fname_bvecs = arguments.bvec

    # Read bvecs
    bvecs = read_bvals_bvecs(fname_bvecs, None)
    bvecs = bvecs[0]
    # if first dimension is not equal to 3 (x,y,z), transpose bvecs file
    if not bvecs.shape[0] == 3:
        bvecs = bvecs.transpose()
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
    fig.suptitle('Number of b=0: ' + str(n_b0) + ', Number of b!=0: ' + str(n_dir - n_b0) + ', Number of effective directions (without duplicates): ' + str(n_dir_eff))
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
    printv("Saving figure: bvecs.png\n")
    plt.savefig('bvecs.png')
    plt.show()


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])

