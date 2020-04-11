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
import re
import time

import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.image import Image, concat_data
from spinalcordtoolbox.utils import Metavar, SmartFormatter

parser = argparse.ArgumentParser(
  description = 'Wrapper to processing scripts, which loops across subjects. Data should be '
                'organized according to the BIDS structure: '
                'https://github.com/sct-pipeline/spine_generic#file-structure',
  formatter_class = SmartFormatter,
  prog=os.path.basename(__file__).strip('.py'))

parser.add_argument("--jobs", type = int, default = 1,
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
parser.add_argument("--path-segmanual", default = ".",
                    help = 'A path containing manual segmentations to be used by the task program.')                
parser.add_argument("--itk-threads", default = 1,
                    help = 'Number of threads to use for ITK based programs including ANTs. Set to a low '
                           'number to avoid a large increase in memory. Defaults to 1')
parser.add_argument("task",
                    help = 'Script used to process the data, probably process_data.sh or similar')

args = parser.parse_args()

# Find subjects and process inclusion/exclusions
data_dir = os.path.expanduser(args.data_path)
subject_dirs = [ f for f in os.listdir(data_dir) if f.startswith(args.subject_prefix) ]

if args.include is not None:
  subject_dirs = [ f for f in subject_dirs if re.search(args.include, f) is not None ]

if args.exclude is not None:
  subject_dirs = [ f for f in subject_dirs if re.search(args.exclude, f) is None ]

# Set up output directories and create them if they don't already exist
out_path = os.path.abspath(args.out_path)
results_path = os.path.join(out_path, "results")
log_path = os.path.join(out_path, "log")
qc_path = os.path.join(out_path, "qc")

for pth in [out_path, results_path, log_path, qc_path]:
  if not os.path.exists(pth):
    os.mkdir(pth)

# Strip the `.sh` extension from the task for building error logs
# TODO: we should probably strip all extensions
task = args.task
task_base = re.sub("\.sh$", "", os.path.basename(task))

## Job function for mapping with multiprocessing
def run_single(subj_dir):
  subject = os.path.basename(subj_dir)
  log_file = os.path.join(log_path, "{}_{}.log".format(task_base, subject))
  err_file = os.path.join(log_path, "err.{}_{}.log".format(task_base, subject))

  print("Running {}. See log file {}".format(subject, log_file))
  
  # A full copy of the environment is needed otherwise sct programs won't necessarily be found
  envir = os.environ.copy()
  # Add the script relevant environment variables
  envir.update({
    "PATH_SEGMANUAL" : args.path_segmanual,
    "PATH_DATA"      : data_dir,
    "PATH_RESULTS"   : results_path,
    "PATH_LOG"       : log_path,
    "PATH_QC"        : qc_path ,
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS" : str(args.itk_threads)
  })

  # Ship the job out, merging stdout/stderr and piping to log file
  res = subprocess.run([args.task, subj_dir],
                       env = envir,
                       stdout = open(log_file, "w"),
                       stderr = subprocess.STDOUT)

  # Check the return code, if it failed rename the log file to indicate
  # an error
  if res.returncode != 0:
    os.rename(log_file, err_file)

  # Assert that the command needs to have run successfully
  assert res.returncode == 0, "Processing of subject {} failed".format(subject)

# Determine the number of jobs we can run simulataneously
if args.jobs < 1:
  jobs = multiprocessing.cpu_count() + args.jobs
else:
  jobs = args.jobs

# Run the jobs, recording start and end times  
start = time.strftime('%H:%M', time.localtime(time.time()))
with multiprocessing.Pool(jobs) as p:
  p.map(run_single, subject_dirs)
end = time.strftime('%H:%M', time.localtime(time.time()))

print("Finished :-)\n"
      "Started: {}\n"
      "Ended: {}\n"
      "".format(start, end)
)

open_cmd = "open" if sys.platform == "darwin" else "xdg-open"

print("To open Quality Control (QC) report on a web-browser, run the following:\n"
      "{} {}/index.html".format(open_cmd, qc_path)
)
