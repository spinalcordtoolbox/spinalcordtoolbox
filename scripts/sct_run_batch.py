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
import functools
import json
import warnings
import yaml

from getpass import getpass
from spinalcordtoolbox.utils import Metavar, SmartFormatter, Tee, send_email
from spinalcordtoolbox import __version__
from textwrap import dedent
from types import SimpleNamespace
import sct_utils as sct


def get_parser():
    parser = argparse.ArgumentParser(
        description='Wrapper to processing scripts, which loops across subjects. Subjects '
        'should be organized as folders within a single directory. We recommend '
        'following the BIDS convention (https://bids.neuroimaging.io/). '
        'The processing script (task) should accept a subject directory as its only argument. '
        'Additional information is passed via environment variables and the arguments '
        'passed via `-task-args`',
        formatter_class=SmartFormatter,
        prog=os.path.basename(__file__).strip('.py'))

    parser.add_argument('-config', '-c',
                        help='R|'
                        'A json (.json) or yaml (.yml|.yaml) file with arguments. All arguments to the configuration '
                        'file are the same as the command line arguments, except all dashes (-) are replaced with '
                        'underscores (_). Using command line flags can be used to override arguments provided in '
                        'the configuration file, but this is discouraged.\n' + dedent(
                            """
                            Example YAML configuration:
                            path_data   : ~/sct_data
                            path_output : ~/pipeline_results
                            task        : nature_paper_analysis.sh\n
                            Example JSON configuration:
                            {
                            "path_data"   : "~/sct_data"
                            "path_output" : "~/pipeline_results"
                            "task"        : "nature_paper_analysis.sh"
                            }\n
                            """))
    parser.add_argument('-jobs', type=int, default=1,
                        help='The number of jobs to run in parallel. '
                        'Either an integer greater than or equal to one '
                        'specifying the number of cores, 0 or a negative integer '
                        'specifying number of cores minus that number. For example '
                        '\'-jobs -1\' indicates run ncores - 1 jobs in parallel. Set \'-jobs 0\''
                        'to use all available cores.',
                        metavar=Metavar.int)
    parser.add_argument('-path-data', help='R|Setting for environment variable: PATH_DATA\n'
                        'Path containing subject directories in a consistent format')
    parser.add_argument('-subject-prefix', default='sub-',
                        help='Subject prefix, defaults to "sub-" which is the prefix used for BIDS directories. '
                        'If the subject directories do not share a common prefix, an empty string can be '
                        'passed here.')
    parser.add_argument('-path-output', default='./',
                        help='R|Base directory for environment variables:\n'
                        'PATH_RESULTS=' + os.path.join('<path-output>', 'results') + '\n'
                        'PATH_QC=' + os.path.join('<path-output>', 'QC') + '\n'
                        'PATH_LOG=' + os.path.join('<path-output>', 'log') + '\n'
                        'Which are respectively output paths for results, QC and logs')
    parser.add_argument('-batch-log', default='sct_run_batch_log.txt',
                        help='A log file for all terminal output produced by this script (not '
                        'necessarily including the individual job outputs. File will be relative '
                        'to "<path-output>/log".')
    parser.add_argument('-include',
                        help='Optional regex used to filter the list of subject directories. Only process '
                        'a subject if they match the regex. Inclusions are processed before exclusions. '
                        'Cannot be used with `include-list`.')
    parser.add_argument('-include-list',
                        help='Optional space separated list of subjects to include. Only process '
                        'a subject if they are on this list. Inclusions are processed before exclusions. '
                        'Cannot be used with `include`.', nargs='+')
    parser.add_argument('-exclude',
                        help='Optional regex used to filter the list of subject directories. Only process '
                        'a subject if they do not match the regex. Exclusions are processed '
                        'after inclusions. Cannot be used with `exclude-list`')
    parser.add_argument('-exclude-list',
                        help='Optional space separated list of subjects to exclude. Only process '
                        'a subject if they are not on this list. Inclusions are processed before exclusions. '
                        'Cannot be used with either `exclude`.', nargs='+')
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
    parser.add_argument('-email-to',
                        help='Optional email address where sct_run_batch can send an alert on completion of the '
                        'batch processing.')
    parser.add_argument('-email-from',
                        help='Optional alternative email to use to send the email. Defaults to the same address as '
                             '`-email-to`')
    parser.add_argument('-email-host', default='smtp.gmail.com:587',
                        help='Optional smtp server and port to use to send the email. Defaults to gmail\'s server')
    parser.add_argument('-continue-on-error', type=int, default=1, choices=(0, 1),
                        help='Whether the batch processing should continue if a subject fails.')
    parser.add_argument('-task',
                        help='Shell script used to process the data.')

    return parser


def run_single(subj_dir, task, task_args, path_segmanual, path_data, path_results, path_log, path_qc, itk_threads, continue_on_error=False):
    'Job function for mapping with multiprocessing'

    # Strip the `.sh` extension from the task for building error logs
    # TODO: we should probably strip all extensions
    task_base = re.sub('\\.sh$', '', os.path.basename(task))
    task_full = os.path.abspath(os.path.expanduser(task))

    subject = os.path.basename(subj_dir)
    log_file = os.path.join(path_log, '{}_{}.log'.format(task_base, subject))
    err_file = os.path.join(path_log, 'err.{}_{}.log'.format(task_base, subject))

    print('Running {}. See log file {}'.format(subject, log_file), flush=True)

    # A full copy of the environment is needed otherwise sct programs won't necessarily be found
    envir = os.environ.copy()
    # Add the script relevant environment variables
    envir.update({
        'PATH_SEGMANUAL': path_segmanual,
        'PATH_DATA': path_data,
        'PATH_RESULTS': path_results,
        'PATH_LOG': path_log,
        'PATH_QC': path_qc,
        'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS': str(itk_threads),
        'SCT_PROGRESS_BAR': 'off'
    })

    # Ship the job out, merging stdout/stderr and piping to log file
    try:
        res = subprocess.run([task_full, subj_dir] + task_args.split(' '),
                             env=envir,
                             stdout=open(log_file, 'w'),
                             stderr=subprocess.STDOUT)

        assert res.returncode == 0, 'Processing of subject {} failed'.format(subject)
    except Exception as e:
        process_completed = 'res' in locals()
        res = res if process_completed else SimpleNamespace(returncode=-1)
        process_suceeded = res.returncode == 0

        if not process_suceeded and os.path.exists(log_file):
            # If the process didn't complete or succeed rename the log file to indicate
            # the error
            os.rename(log_file, err_file)

        if process_suceeded or continue_on_error:
            return res
        else:
            raise e

    return res


def main(argv):
    # Print the sct startup info
    sct.init_sct()

    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args(argv if argv else ['--help'])

    # See if there's a configuration file and import those options
    if args.config is not None:
        print('configuring')
        with open(args.config, 'r') as conf:
            _, ext = os.path.splitext(args.config)
            if ext == '.json':
                config = json.load(conf)
            if ext == '.yml' or ext == '.yaml':
                config = yaml.load(conf, Loader=yaml.Loader)

        # Warn people if they're overriding their config file
        if len(vars(args)) > 1:
            warnings.warn(UserWarning('Using the `-config|-c` flag with additional arguments is discouraged'))

        # Check for unsupported arguments
        orig_keys = set(vars(args).keys())
        config_keys = set(config.keys())
        if orig_keys != config_keys:
            for k in config_keys.difference(orig_keys):
                del config[k]  # Remove the unknown key
                warnings.warn(UserWarning(
                    'Unknown key "{}" found in your configuration file, ignoring.'.format(k)))

        # Update the default to match the config
        parser.set_defaults(**config)

        # Reparse the arguments
        args = parser.parse_args(argv)

    # Set up email notifications if desired
    do_email = args.email_to is not None
    if do_email:
        email_to = args.email_to
        if args.email_from is not None:
            email_from = args.email_from
        else:
            email_from = args.email_to

        smtp_host, smtp_port = args.email_host.split(":")
        smtp_port = int(smtp_port)
        email_pass = getpass('Please input your email password:\n')

        def send_notification(subject, message):
            send_email(email_to, email_from,
                       subject=subject,
                       message=message,
                       passwd=email_pass,
                       smtp_host=smtp_host,
                       smtp_port=smtp_port)

        while True:
            send_test = input('Would you like to send a test email to validate your settings? [Y/n]:\n')
            if send_test.lower() in ['', 'y', 'n']:
                break
            else:
                print('Please input y or n')

        if send_test.lower() in ['', 'y']:
            send_notification('sct_run_batch: test notification', 'Looks good')

    # Set up output directories and create them if they don't already exist
    path_output = os.path.abspath(os.path.expanduser(args.path_output))
    path_results = os.path.join(path_output, 'results')
    path_log = os.path.join(path_output, 'log')
    path_qc = os.path.join(path_output, 'qc')
    path_segmanual = os.path.abspath(os.path.expanduser(args.path_segmanual))
    task = os.path.abspath(os.path.expanduser(args.task))

    for pth in [path_output, path_results, path_log, path_qc]:
        if not os.path.exists(pth):
            os.mkdir(pth)

    # Check that the task can be found
    if not os.path.exists(task):
        raise FileNotFoundError('Couldn\'t find the task script at {}'.format(task))

    # Setup overall log
    batch_log = open(os.path.join(path_log, args.batch_log), 'w')

    # Duplicate init_sct message to batch_log
    print('\n--\nSpinal Cord Toolbox ({})\n'.format(__version__), file=batch_log, flush=True)

    # Tee IO to batch_log and std(out/err)
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    sys.stdout = Tee(batch_log, orig_stdout)
    sys.stderr = Tee(batch_log, orig_stderr)

    def reset_streams():
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr

    # Log the current arguments (in yaml because it's cleaner)
    print('sct_run_batch arguments were:')
    print(yaml.dump(vars(args)))

    # Find subjects and process inclusion/exclusions
    path_data = os.path.abspath(os.path.expanduser(args.path_data))
    subject_dirs = [f for f in os.listdir(path_data) if f.startswith(args.subject_prefix)]

    # Handle inclusion lists
    assert not ((args.include is not None) and (args.include_list is not None)),\
        'Only one of `include` and `include-list` can be used'

    if args.include is not None:
        subject_dirs = [f for f in subject_dirs if re.search(args.include, f) is not None]

    if args.include_list is not None:
        # TODO decide if we should warn users if one of their inclusions isn't around
        subject_dirs = [f for f in subject_dirs if f in args.include_list]

    # Handle exclusions
    assert not ((args.exclude is not None) and (args.exclude_list is not None)),\
        'Only one of `exclude` and `exclude-list` can be used'

    if args.exclude is not None:
        subject_dirs = [f for f in subject_dirs if re.search(args.exclude, f) is None]

    if args.exclude_list is not None:
        subject_dirs = [f for f in subject_dirs if f not in args.exclude_list]

    # Determine the number of jobs we can run simulataneously
    if args.jobs < 1:
        jobs = multiprocessing.cpu_count() + args.jobs
    else:
        jobs = args.jobs

    # Run the jobs, recording start and end times
    start = time.strftime('%H:%M', time.localtime(time.time()))

    # Trap errors to send an email if a task fails.
    try:
        with multiprocessing.Pool(jobs) as p:
            run_single_dir = functools.partial(run_single,
                                               task=task,
                                               task_args=args.task_args,
                                               path_segmanual=path_segmanual,
                                               path_data=path_data,
                                               path_results=path_results,
                                               path_log=path_log,
                                               path_qc=path_qc,
                                               itk_threads=args.itk_threads,
                                               continue_on_error=args.continue_on_error)
            results = p.map(run_single_dir, subject_dirs)
            end = time.strftime('%H:%M', time.localtime(time.time()))
    except Exception as e:
        if do_email is not None:
            message = ('Oh no there has been the following error in your pipeline:\n\n'
                       '{}'.format(e))
            try:
                # I consider the multiprocessing error more significant than a potential email error, this
                # ensures that the multiprocessing error is signalled.
                send_notification('sct_run_batch errored', message)
            except Exception:
                raise e

            raise e
        else:
            raise e

    end = time.strftime('%H:%M', time.localtime(time.time()))

    # Check for failed subjects
    fails = [sd for (sd, ret) in zip(subject_dirs, results) if ret.returncode != 0]

    smiley_or_newline = ':-)\n' if len(fails) == 0 else '\n'
    completed_message = ('Finished {}'
                         'Started: {}\n'
                         'Ended: {}\n'
                         ''.format(smiley_or_newline, start, end))

    if len(fails) == 0:
        status_message = 'Hooray your batch completed successfully\n'
    else:
        status_message = ('Your batch completed but some subjects may have not completed '
                          'successfully, please consult the logs for:\n'
                          '{}\n'.format('\n'.join(fails)))

    print(status_message + completed_message)

    if do_email:
        send_notification('sct_run_batch: Run completed',
                          status_message + completed_message)

    open_cmd = 'open' if sys.platform == 'darwin' else 'xdg-open'

    print('To open the Quality Control (QC) report on a web-browser, run the following:\n'
          '{} {}/index.html'.format(open_cmd, path_qc))

    reset_streams()
    batch_log.close()


if __name__ == '__main__':
    main(sys.argv[1:])
