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
from msct_parser import Parser

import sys
import sct_utils as sct
#import numpy as np


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
    parser = Parser(__file__)
    parser.usage.set_description('Function to get the Ernst Angle. For examples of T1 values, see Stikov et al. MRM 2015. Example in the white matter at 3T: 850ms.')
    parser.add_option(name="-t1",
                      type_value="float",
                      description="T1 value (in ms).",
                      mandatory=False,
                      default_value=850.0,
                      example='800')
    parser.add_option(name="-b",
                      type_value=[[','], 'float'],
                      description="Boundaries TR parameter (in ms) in case -v 2 is used.",
                      mandatory=False,
                      example='500,3500')
    parser.add_option(name="-tr",
                      type_value='float',
                      description="Value of TR (in ms) to get the Ernst Angle. ",
                      mandatory=False,
                      example='2000')
    parser.add_option(name="-d",
                      type_value=None,
                      description="Display option. The graph isn't display if 0.  ",
                      deprecated_by="-v",
                      mandatory=False)
    parser.add_option(name="-o",
                      type_value="file_output",
                      description="Name of the output graph of the Ernst angle.",
                      mandatory=False,
                      example="ernst_angle.png")
    parser.add_option(name="-v",
                      type_value='multiple_choice',
                      description="verbose: 0 = nothing, 1 = classic, 2 = expended (graph)",
                      mandatory=False,
                      example=['0', '1', '2'],
                      default_value='1')
    return parser

#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    sct.start_stream_logger()
    # initialize parameters
    param = Param()
    param_default = Param()

    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])

    input_t1 = arguments["-t1"]
    input_fname_output = None
    input_tr_min = 500
    input_tr_max = 3500
    input_tr = None
    verbose = 1
    if "-o" in arguments:
        input_fname_output = arguments["-o"]
    if "-b" in arguments:
        input_tr_min = arguments["-b"][0]
        input_tr_max = arguments["-b"][1]
    if "-tr" in arguments :
        input_tr = arguments["-tr"]
    if "-v" in arguments :
        verbose = int(arguments["-v"])

    graph = ErnstAngle(input_t1, tr=input_tr, fname_output=input_fname_output)
    if input_tr is not None:
        sct.printv("\nValue of the Ernst Angle with T1=" + str(graph.t1) + "ms and TR=" + str(input_tr) + "ms :", verbose=verbose, type='info')
        sct.printv(str(graph.getErnstAngle(input_tr)))
        if input_tr > input_tr_max:
            input_tr_max = input_tr + 500
        elif input_tr < input_tr_min:
            input_tr_min = input_tr - 500
    if verbose == 2 :
        graph.draw(input_tr_min, input_tr_max)
