#!/usr/bin/env python
#########################################################################################
#
# Flip data in a specified direction. Note: this will NOT change the header, but it will change the way the data are stored.
#
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
# Modified: 2014-04-18
#
# About the license: see the file LICENSE.TXT
#########################################################################################



def usage():
    """
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  Flip data in a specified dimension (x,y,z or t).
  N.B. This script will NOT modify the header but the way the data are stored (so be careful!!).

USAGE
  `basename ${0}` -i <input> -d <x|y|z|t>

MANDATORY ARGUMENTS
  -i <input>                   image
  -d <dimension>               dimension: x|y|z|t

    """