#!/usr/bin/env python
#########################################################################################
#
# All sort of utilities for labels.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Sara Dupont
# Modified: 2015-02-17
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from __future__ import absolute_import, division

import sys
import os
import argparse

import sct_utils as sct
from spinalcordtoolbox.utils import Metavar, SmartFormatter


# DEFAULT PARAMETERS
class Param:
    # The constructor
    def __init__(self):
        self.debug = 0
        self.verbose = 1
        self.t1 = 0


class ErnstAngle:
    # The constructor
    def __init__(self, t1, tr=None, fname_output=None):
        self.t1 = t1
        self.tr = tr
        self.fname_output = fname_output

    # compute and return the Ernst Angle
    def getErnstAngle(self, tr):
        from numpy import arccos
        from numpy import exp
        from math import pi
        angle_rad = arccos(exp(-tr / self.t1))
        angle_deg = angle_rad * 180 / pi
        return angle_deg

    # draw the graph
    def draw(self, tr_min, tr_max):
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from numpy import arange
        step = (tr_max - tr_min) / 50
        tr_range = arange(tr_min, tr_max, step)
        theta_E = self.getErnstAngle(tr_range)

        sct.printv("\nDrawing", type='info')
        plt.plot(tr_range, theta_E, linewidth=1.0)
        plt.xlabel("TR (in $ms$)")
        plt.ylabel("$\Theta_E$ (in degree)")
        plt.ylim(min(theta_E), max(theta_E) + 2)
        plt.title("Ernst Angle with T1=" + str(self.t1) + "ms")
        plt.grid(True)

        if self.tr is not None:
            plt.plot(self.tr, self.getErnstAngle(self.tr), 'ro')
        if self.fname_output is not None :
            sct.printv("\nSaving figure", type='info')
            plt.savefig(self.fname_output, format='png')
        plt.show()


def get_parser():
    # Initialize the parser
    parser = argparse.ArgumentParser(
        description='Function to compute the Ernst Angle. For examples of T1 values in the brain, see Wansapura et al. '
                    'NMR relaxation times in the human brain at 3.0 tesla. Journal of magnetic resonance imaging : '
                    'JMRI (1999) vol. 9 (4) pp. 531-8. \nT1 in WM: 832ms\nT1 in GM: 1331ms',
        add_help=None,
        formatter_class=SmartFormatter,
        prog=os.path.basename(__file__).strip(".py"))

    mandatoryArguments = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatoryArguments.add_argument(
        "-tr",
        type=float,
        required=True,
        help='Value of TR (in ms) to get the Ernst Angle. Example: 2000',
        metavar=Metavar.float,
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")
    optional.add_argument(
        "-t1",
        type=float,
        help='T1 value (in ms). Example: 832.3',
        required=False,
        metavar=Metavar.float,
        default=832.0)
    optional.add_argument(
        "-b",
        type=float,
        nargs='*',
        metavar=Metavar.float,
        help='Min/Max range of TR (in ms) separated with space. Only use with -v 2. Example: 500 3500',
        required=False)
    optional.add_argument(
        "-o",
        help="Name of the output file containing Ernst angle result.",
        required=False,
        metavar=Metavar.str,
        default="ernst_angle.txt")
    optional.add_argument(
        "-ofig",
        help="Name of the output graph. Only use with -v 2.",
        required=False,
        metavar=Metavar.str,
        default="ernst_angle.png")
    optional.add_argument(
        "-v",
        type=int,
        help="Verbose: 0 = nothing, 1 = classic, 2 = expended (graph)",
        required=False,
        choices=(0, 1, 2),
        default=1)

    return parser


# main
#=======================================================================================================================
def main():
    parser = get_parser()
    arguments = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    # Initialization
    param = Param()
    input_t1 = arguments.t1
    input_fname_output = None
    input_tr_min = 500
    input_tr_max = 3500
    input_tr = None
    verbose = 1
    fname_output_file = arguments.o
    if arguments.ofig is not None:
        input_fname_output = arguments.ofig
    if arguments.b is not None:
        input_tr_min = arguments.b[0]
        input_tr_max = arguments.b[1]
    if arguments.tr is not None:
        input_tr = arguments.tr
    verbose = arguments.v
    sct.init_sct(log_level=verbose, update=True)  # Update log level

    graph = ErnstAngle(input_t1, tr=input_tr, fname_output=input_fname_output)
    if input_tr is not None:
        sct.printv("\nValue of the Ernst Angle with T1=" + str(graph.t1) + "ms and TR=" + str(input_tr) + "ms :", verbose=verbose, type='info')
        sct.printv(str(graph.getErnstAngle(input_tr)))
        if input_tr > input_tr_max:
            input_tr_max = input_tr + 500
        elif input_tr < input_tr_min:
            input_tr_min = input_tr - 500
        # save text file
        try:
            f = open(fname_output_file, 'w')
            f.write(str(graph.getErnstAngle(input_tr)))
            f.close()
        except:
            sct.printv('\nERROR: Cannot open file'+fname_output_file, '1', 'error')

    if verbose == 2:
        graph.draw(input_tr_min, input_tr_max)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    sct.init_sct()
    main()
