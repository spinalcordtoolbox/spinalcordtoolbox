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
import subprocess

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
parser.add_argument("--path-segmanual", default = "."
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

out_path = args.out_path
results_path = os.path.join(out_path, "results")
log_path = os.path.join(out_path, "log")
qc_path = os.path.join(out_path, "qc_path")

for pth in [out_path, results_path, log_path, qc_path]:
  if not os.exists(pth):
    os.mkdir(out_path)

task = args.task
task_base = re.sub("\.sh$", "", os.path.basename(task))
    
def run_single(subj_dir):
  subject = os.path.basename(subj_dir)
  log_file = os.path.join(log_path, "{}_{}.log".format(task_base, subject))
  err_file = os.path.join(log_path, "err.{}_{}.log".format(task_base, subject))
                          
  res = subprocess.run([args.task, subj_dir],
                       env = {"PATH_SEGMANUAL" : args.path_segmanual,
                              "PATH_RESULTS"   : results_path,
                              "PATH_LOG"       : log_path,
                              "PATH_QC"        : qc_path
                              "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS" : args.itk_threads
                       },
                       stdout = open(log_file, "w")
                       stderr = STDOUT)

  if res.returncode != 0:
    os.rename(log_file, err_file)

  assert res.returncode == 0, "Processing of subject {} failed".format(subject)

# Determine the number of jobs we can run simulataneously
if args.jobs < 1:
  jobs = multiprocessing.cpu_count() + args.jobs
else:
  jobs = args.jobs

with Pool(jobs) as p:
  p.map(run_single, subject_dirs)

