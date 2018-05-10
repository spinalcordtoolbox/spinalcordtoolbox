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
    parser.usage.set_description('Function to compute the Ernst Angle. For examples of T1 values in the brain, see Wansapura et al. NMR relaxation times in the human brain at 3.0 tesla. Journal of magnetic resonance imaging : JMRI (1999) vol. 9 (4) pp. 531-8. \nT1 in WM: 832ms\nT1 in GM: 1331ms')
    parser.add_option(name="-tr",
                      type_value='float',
                      description="Value of TR (in ms) to get the Ernst Angle. ",
                      mandatory=True,
                      example='2000')
    parser.add_option(name="-t1",
                      type_value="float",
                      description="T1 value (in ms).",
                      mandatory=False,
                      default_value=832.0,
                      example='832')
    parser.add_option(name="-b",
                      type_value=[[','], 'float'],
                      description="Boundaries TR parameter (in ms) in case -v 2 is used.",
                      mandatory=False,
                      example='500,3500')
    parser.add_option(name="-d",
                      type_value=None,
                      description="Display option. The graph isn't display if 0.  ",
                      deprecated_by="-v",
                      mandatory=False)
    parser.add_option(name="-o",
                      type_value="file_output",
                      description="Name of the output file containing Ernst angle result.",
                      mandatory=False,
                      default_value="ernst_angle.txt")
    parser.add_option(name="-ofig",
                      type_value="file_output",
                      description="Name of the output graph (only if -v 2 is used).",
                      mandatory=False,
                      default_value="ernst_angle.png")
    parser.add_option(name="-v",
                      type_value='multiple_choice',
                      description="verbose: 0 = nothing, 1 = classic, 2 = expended (graph)",
                      mandatory=False,
                      example=['0', '1', '2'],
                      default_value='1')
    return parser


# main
#=======================================================================================================================
def main(args=None):

    # Initialization
    param = Param()

    # check user arguments
    if not args:
        args = sys.argv[1:]

    # Check input parameters
    parser = get_parser()
    arguments = parser.parse(args)

    input_t1 = arguments["-t1"]
    input_fname_output = None
    input_tr_min = 500
    input_tr_max = 3500
    input_tr = None
    verbose = 1
    fname_output_file = arguments['-o']
    if "-ofig" in arguments:
        input_fname_output = arguments["-ofig"]
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
