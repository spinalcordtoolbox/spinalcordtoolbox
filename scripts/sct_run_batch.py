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
import multiprocessing

import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.image import Image, concat_data
from spinalcordtoolbox.utils import Metavar, SmartFormatter

parser = argparse.ArgumentParser(
  description = 'Wrapper to processing scripts, which loops across subjects. Data should be '
                'organized according to the BIDS structure: '
                'https://github.com/sct-pipeline/spine_generic#file-structure',
  formatter_class = SmartFormatter,
  prog=os.path.basename(__file__).strip('.py'))

parser.add_argument("--jobs", type = int, default = 1
                    help = 'The number of jobs to run in parallel. '
                           'either an integer greater than or equal to one '
                           'specifying the number of cores, 0 or a negative integer '
                           'specifying number of cores minus that number. For example '
                           '--jobs -1 indicates run ncores - 1 jobs in parallel.')
parser.add_argument("--data-path", help = 'Path containing subject directories in the BIDS format')
parser.add_argument("--subject-prefix", default = "sub-",
                    help = 'Subject prefix, defaults to "sub-"')
parser.add_argument("--out-path", default = "./",
                    help = 'Output path, subdirectories for results, logs, and QC '
                           'will be generated here')
parser.add_argument("--include",
                    help = 'Optional regex used to filter the list of subject directories. Only process '
                           'a subject if they match the regex. Inclusions are processed before exclusions')
parser.add_argument("--exclude",
                    help = 'Optional regex used to filter the list of subject directories. Only process '
                           'a subject if they do not match the regex. Exclusions are processed '
                           'after inclusions')
parser.add_argument("--segmentation",
                    help = 'A path containing manual segmentations to be used by the task program.')                
parser.add_argument("--itk-threads",
                    help = 'Number of threads to use for ITK based programs including ANTs. Set to a low '
                           'number to avoid a large increase in memory. Defaults to 1')
parser.add_argument("task",
                    help = 'Script used to process the data, probably process_data.sh or similar')

args = parser.parse_args()

# Find subjects and process inclusion/exclusions
data_dir = os.path.expanduser(args.data_path)
subject_dirs = [ f for f in os.listdir(data_dir) if f.startswith(args.prefix) ]

if args.include is not None:
  subject_dirs = [ f for f in subject_dirs if re.match(args.include, f) is not None ]

if args.exclude is not None:
  subject_dirs = [ f for f in subject_dirs if re.match(args.include, f) is None ]

# Determine the number of jobs we can run simulataneously
if args.jobs < 1:
  jobs = multiprocessing.cpu_count() + args.jobs
else:
  jobs = args.jobs

