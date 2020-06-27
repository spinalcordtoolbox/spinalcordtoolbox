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
import multiprocessing
import subprocess
import re
import time

from spinalcordtoolbox.utils import Metavar, SmartFormatter

parser = argparse.ArgumentParser(
    description='Wrapper to processing scripts, which loops across subjects. Subjects '
    'should be organized as folders within a single directory. We recommend '
                'following the BIDS convention (https://bids.neuroimaging.io/). '
                'The processing script (task) should accept a subject directory as its only argument. '
                'Additional information is passed via environment variables and the arguments '
                'passed via `-task-args`',
    formatter_class=SmartFormatter,
    prog=os.path.basename(__file__).strip('.py'))

parser.add_argument("-jobs", type=int, default=1,
                    help='The number of jobs to run in parallel. '
                    'Either an integer greater than or equal to one '
                    'specifying the number of cores, 0 or a negative integer '
                    'specifying number of cores minus that number. For example '
                    '\'-jobs -1\' indicates run ncores - 1 jobs in parallel. Set \'-jobs 0\''
                    'to use all available cores.',
                    metavar=Metavar.int)
parser.add_argument("-path-data", help='R|Setting for environment variable: PATH_DATA\n'
                    'Path containing subject directories in a consistent format')
parser.add_argument('-subject-prefix', default="sub-",
                    help='Subject prefix, defaults to "sub-" which is the prefix used for BIDS directories. '
                    'If the subject directories do not share a common prefix, an empty string can be '
                    'passed here.')
parser.add_argument('-path-output', default="./",
                    help='R|Base directory for environment variables:\n'
                    'PATH_RESULTS=' + os.path.join('<path-output>', 'results') + '\n'
                    'PATH_QC=' + os.path.join('<path-output>', 'QC') + '\n'
                    'PATH_LOG=' + os.path.join('<path-output>', 'log') + '\n'
                    'Which are respectively output paths for results, QC and logs')
parser.add_argument('-include',
                    help='Optional regex used to filter the list of subject directories. Only process '
                    'a subject if they match the regex. Inclusions are processed before exclusions.')
parser.add_argument('-exclude',
                    help='Optional regex used to filter the list of subject directories. Only process '
                    'a subject if they do not match the regex. Exclusions are processed '
                    'after inclusions.')
parser.add_argument('-path-segmanual', default='.',
                    help='R|Setting for environment variable: PATH_SEGMANUAL\n'
                    'A path containing manual segmentations to be used by the task program.')
parser.add_argument('-itk-threads', type=int, default=1,
                    help='R|Setting for environment variable: ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS\n'
                    'Number of threads to use for ITK based programs including ANTs. Set to a low '
                    'number to avoid a large increase in memory. Defaults to 1',
                    metavar=Metavar.int)
parser.add_argument('-task-args', default='',
                    help='A quoted string with extra flags and arguments to pass to the task script. '
                    'For example \'sct_run_batch -path-data data/ -task-args "-foo bar -baz /qux" process_data.sh \'')
parser.add_argument('task',
                    help='Shell script used to process the data.')

args = parser.parse_args()

# Find subjects and process inclusion/exclusions
path_data = os.path.abspath(os.path.expanduser(args.path_data))
subject_dirs = [f for f in os.listdir(path_data) if f.startswith(args.subject_prefix)]

if args.include is not None:
    subject_dirs = [f for f in subject_dirs if re.search(args.include, f) is not None]

if args.exclude is not None:
    subject_dirs = [f for f in subject_dirs if re.search(args.exclude, f) is None]

# Set up output directories and create them if they don't already exist
path_output = os.path.abspath(os.path.expanduser(args.path_output))
path_results = os.path.join(path_output, 'results')
path_log = os.path.join(path_output, 'log')
path_qc = os.path.join(path_output, 'qc')

for pth in [path_output, path_results, path_log, path_qc]:
    if not os.path.exists(pth):
        os.mkdir(pth)

# Strip the `.sh` extension from the task for building error logs
# TODO: we should probably strip all extensions
task = args.task
task_base = re.sub('\\.sh$', '', os.path.basename(task))
task_full = os.path.abspath(os.path.expanduser(task))

# Job function for mapping with multiprocessing


def run_single(subj_dir):
    subject = os.path.basename(subj_dir)
    log_file = os.path.join(path_log, '{}_{}.log'.format(task_base, subject))
    err_file = os.path.join(path_log, 'err.{}_{}.log'.format(task_base, subject))

    print('Running {}. See log file {}'.format(subject, log_file))

    # A full copy of the environment is needed otherwise sct programs won't necessarily be found
    envir = os.environ.copy()
    # Add the script relevant environment variables
    envir.update({
        'PATH_SEGMANUAL': args.path_segmanual,
        'PATH_DATA': path_data,
        'PATH_RESULTS': path_results,
        'PATH_LOG': path_log,
        'PATH_QC': path_qc,
        'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS': str(args.itk_threads),
        'SCT_PROGRESS_BAR': 'off'
    })

    # Ship the job out, merging stdout/stderr and piping to log file
    res = subprocess.run([task_full, subj_dir] + args.task_args.split(' '),
                         env=envir,
                         stdout=open(log_file, 'w'),
                         stderr=subprocess.STDOUT)

    # Check the return code, if it failed rename the log file to indicate
    # an error
    if res.returncode != 0:
        os.rename(log_file, err_file)

    # Assert that the command needs to have run successfully
    assert res.returncode == 0, 'Processing of subject {} failed'.format(subject)


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

print('Finished :-)\n'
      'Started: {}\n'
      'Ended: {}\n'
      ''.format(start, end)
      )

open_cmd = 'open' if sys.platform == 'darwin' else 'xdg-open'

print('To open the Quality Control (QC) report on a web-browser, run the following:\n'
      '{} {}/index.html'.format(open_cmd, path_qc)
      )
