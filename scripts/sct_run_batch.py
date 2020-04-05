#!/usr/bin/env python
##############################################################################
#
# Wrapper to processing scripts, which loops across subjects. Data should be
# organized according to the BIDS structure:
# https://github.com/sct-pipeline/spine_generic#file-structure
#
# ----------------------------------------------------------------------------
# Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad, Chris Hammill
#
# About the license: see the file LICENSE.TXT
##############################################################################

from __future__ import absolute_import

import os
import sys
import argparse
import numpy as np

import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.image import Image, concat_data
from spinalcordtoolbox.utils import Metavar, SmartFormatter

parser = argparse.ArgumentParser(
  description = 'Wrapper to processing scripts, which loops across subjects. Data should be '
                'organized according to the BIDS structure: '
                'https://github.com/sct-pipeline/spine_generic#file-structure',
  formatter_class = SmartFormatter,
  prog=os.path.basename(__file__).strip('.py'))

parser.add_argument("--jobs", type = int)
parser.add_argument("task")
