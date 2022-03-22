#!/usr/bin/env python
##############################################################################
#
# Wrapper to processing scripts, which loops across subjects. Data should be
# organized according to the BIDS structure and the processing script should
# make use of the some of the environment variables passed here. More details
# at: https://spine-generic.readthedocs.io/en/latest/
#
# ----------------------------------------------------------------------------
# Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad, Chris Hammill
#
# About the license: see the file LICENSE.TXT
##############################################################################

import os
import sys
from getpass import getpass
import multiprocessing
import subprocess
import platform
import re
import time
import datetime
import functools
import json
import tempfile
import warnings
import shutil
from types import SimpleNamespace
from textwrap import dedent

import yaml
import psutil

from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar
from spinalcordtoolbox.utils.sys import send_email, init_sct, __get_commit, __get_git_origin, __version__, __sct_dir__, set_loglevel
from spinalcordtoolbox.utils.fs import Tee

from stat import S_IEXEC


def get_parser():
    parser = SCTArgumentParser(
        description='Wrapper to processing scripts, which loops across subjects. Subjects should be organized as '
                    'folders within a single directory. We recommend following the BIDS convention '
                    '(https://bids.neuroimaging.io/). The processing script should accept a subject directory '
                    'as its only argument. Additional information is passed via environment variables and the '
                    'arguments passed via `-script-args`. If the script or the input data are located within a '
                    'git repository, the git commit is displayed. If the script or data have changed since the latest '
                    'commit, the symbol "*" is added after the git commit number. If no git repository is found, the '
                    'git commit version displays "?!?". The script is copied on the output folder (-path-out).'
    )

    parser.add_argument('-config', '-c',
                        help=''
                        'A json (.json) or yaml (.yml|.yaml) file with arguments. All arguments to the configuration '
                        'file are the same as the command line arguments, except all dashes (-) are replaced with '
                        'underscores (_). Using command line flags can be used to override arguments provided in '
                        'the configuration file, but this is discouraged. Please note that while quotes are optional '
                        'for strings in YAML omitting them may cause parse errors.\n' + dedent(
                            """
                            Example YAML configuration:
                            path_data   : "~/sct_data"
                            path_output : "~/pipeline_results"
                            script      : "nature_paper_analysis.sh"
                            jobs        : -1\n
                            Example JSON configuration:
                            {
                            "path_data"   : "~/sct_data"
                            "path_output" : "~/pipeline_results"
                            "script"      : "nature_paper_analysis.sh"
                            "jobs"        : -1
                            }\n
                            """))
    parser.add_argument('-jobs', type=int, default=1,
                        help='The number of jobs to run in parallel. Either an integer greater than or equal to one '
                        'specifying the number of cores, 0 or a negative integer specifying number of cores minus '
                        'that number. For example \'-jobs -1\' will run with all the available cores minus one job in '
                        'parallel. Set \'-jobs 0\' to use all available cores.\n'
                        'This argument enables process-based parallelism, while \'-itk-threads\' enables thread-based '
                        'parallelism. You may need to tweak both to find a balance that works best for your system.',
                        metavar=Metavar.int)
    parser.add_argument('-itk-threads', type=int, default=1,
                        help='Sets the environment variable "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS".\n'
                        'Number of threads to use for ITK based programs including ANTs. Increasing this can '
                        'provide a performance boost for high-performance (multi-core) computing environments. '
                        'However, increasing the number of threads may also result in a large increase in memory.\n'
                        'This argument enables thread-based parallelism, while \'-jobs\' enables process-based '
                        'parallelism. You may need to tweak both to find a balance that works best for your system.',
                        metavar=Metavar.int)
    parser.add_argument('-path-data', help='Setting for environment variable: PATH_DATA\n'
                        'Path containing subject directories in a consistent format')
    parser.add_argument('-subject-prefix', default='sub-',
                        help='Subject prefix, defaults to "sub-" which is the prefix used for BIDS directories. '
                        'If the subject directories do not share a common prefix, an empty string can be '
                        'passed here.')
    parser.add_argument('-path-output', default='./',
                        help='Base directory for environment variables:\n'
                        'PATH_DATA_PROCESSED=' + os.path.join('<path-output>', 'data_processed') + '\n'
                        'PATH_RESULTS=' + os.path.join('<path-output>', 'results') + '\n'
                        'PATH_QC=' + os.path.join('<path-output>', 'qc') + '\n'
                        'PATH_LOG=' + os.path.join('<path-output>', 'log') + '\n'
                        'Which are respectively output paths for the processed data, results, quality control (QC) '
                        'and logs')
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
                        help='Setting for environment variable: PATH_SEGMANUAL\n'
                        'A path containing manual segmentations to be used by the script program.')
    parser.add_argument('-script-args', default='',
                        help='A quoted string with extra flags and arguments to pass to the script. '
                        'For example \'sct_run_batch -path-data data/ -script-args "-foo bar -baz /qux" process_data.sh \'')
    parser.add_argument('-email-to',
                        help='Optional email address where sct_run_batch can send an alert on completion of the '
                        'batch processing.')
    parser.add_argument('-email-from',
                        help='Optional alternative email to use to send the email. Defaults to the same address as '
                             '`-email-to`')
    parser.add_argument('-email-host', default='smtp.gmail.com:587',
                        help='Optional smtp server and port to use to send the email. Defaults to gmail\'s server. Note'
                             ' that gmail server requires "Less secure apps access" to be turned on, which can be done '
                             ' at https://myaccount.google.com/security')
    parser.add_argument('-continue-on-error', type=int, default=1, choices=(0, 1),
                        help='Whether the batch processing should continue if a subject fails.')
    parser.add_argument('-script',
                        help='Shell script used to process the data.')
    parser.add_argument('-zip',
                        action='store_true',
                        help='Create zip archive of output folders log/, qc/ and results/.')
    parser.add_argument('-v', metavar=Metavar.int, type=int, choices=[0, 1, 2], default=1,
                        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
                        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode")
    parser.add_argument('-h', "--help", action="help", help="show this help message and exit")

    return parser


def _find_nonsys32_bash_exe():
    """
    Check the PATH to see if there is a non-System32 copy of bash.exe.

    On newer copies of Windows. a bash.exe file is included in the System32
    folder. When called, Windows will try to launch WSL if it's installed,
    or it will give this message if WSL is not installed:

    > Windows Subsystem for Linux has no installed distributions.
    > Distributions can be installed by visiting the Microsoft Store:
    > https://aka.ms/wslstore

    Since we are supporting Windows natively, we don't want the WSL copy, and instead
    want to use the next copy of bash.exe in the PATH (for example one provided by Cygwin).
    """
    nonsys32_paths = os.pathsep.join([p for p in os.environ['PATH'].split(os.pathsep)
                                      if 'system32' not in p.lower()])
    return shutil.which('bash', path=nonsys32_paths)


def run_single(subj_dir, script, script_args, path_segmanual, path_data, path_data_processed, path_results, path_log,
               path_qc, itk_threads, continue_on_error=False):
    """
    Job function for mapping with multiprocessing
    :param subj_dir:
    :param script:
    :param script_args:
    :param path_segmanual:
    :param path_data:
    :param path_data_processed:
    :param path_results:
    :param path_log:
    :param path_qc:
    :param itk_threads:
    :param continue_on_error:
    :return:
    """

    # Strip the `.sh` extension from the script for building error logs
    # TODO: we should probably strip all extensions
    script_base = re.sub('\\.sh$', '', os.path.basename(script))
    script_full = os.path.abspath(os.path.expanduser(script))

    if os.path.sep in subj_dir:
        subject, session = subj_dir.split(os.path.sep)
        subject_session = subject + '_' + session
    else:
        subject = subj_dir
        subject_session = subject

    log_file = os.path.join(path_log, '{}_{}.log'.format(script_base, subject_session))
    err_file = os.path.join(path_log, 'err.{}_{}.log'.format(script_base, subject_session))

    print('Started at {}: {}. See log file {}'.format(time.strftime('%Hh%Mm%Ss'), subject_session, log_file), flush=True)

    # A full copy of the environment is needed otherwise sct programs won't necessarily be found
    envir = os.environ.copy()
    # Add the script relevant environment variables
    envir.update({
        'PATH_SEGMANUAL': path_segmanual,
        'PATH_DATA': path_data,
        'PATH_DATA_PROCESSED': path_data_processed,
        'PATH_RESULTS': path_results,
        'PATH_LOG': path_log,
        'PATH_QC': path_qc,
        'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS': str(itk_threads),
        'SCT_PROGRESS_BAR': 'off'
    })
    if 'SCT_DIR' not in envir:  # For native Windows installations, the install script won't add SCT_DIR
        envir['SCT_DIR'] = __sct_dir__

    cmd = [script_full, subj_dir] + script_args.split(' ')

    # Make sure that a Windows-compatible bash port is used to run shell scripts
    if sys.platform == "win32":
        with open(script_full) as f:
            shebang = f.readline()
        if script_full.endswith('.sh') or "sh" in shebang:
            bash_exe = _find_nonsys32_bash_exe()
            if bash_exe:
                cmd.insert(0, bash_exe)
            else:
                print("WARNING: No bash.exe found in PATH. Script will be run as-is.")
                print("(If you are on Windows and trying to run a `.sh` script, please "
                      "install Cygwin, then add C:/cygwin64/bin to your PATH.)")
        else:
            print("WARNING: Script passed to sct_run_batch appears to be a non-shell script.")

    # Ship the job out, merging stdout/stderr and piping to log file
    try:
        res = subprocess.run(cmd,
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


def main(argv=None):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)

    # See if there's a configuration file and import those options
    if arguments.config is not None:
        print('configuring')
        with open(arguments.config, 'r') as conf:
            _, ext = os.path.splitext(arguments.config)
            if ext == '.json':
                config = json.load(conf)
            if ext == '.yml' or ext == '.yaml':
                config = yaml.load(conf, Loader=yaml.Loader)

        # Warn people if they're overriding their config file
        if len(argv) > 2:
            warnings.warn(UserWarning('Using the `-config|-c` flag with additional arguments is discouraged'))

        # Check for unsupported arguments
        orig_keys = set(vars(arguments).keys())
        config_keys = set(config.keys())
        if orig_keys != config_keys:
            for k in config_keys.difference(orig_keys):
                del config[k]  # Remove the unknown key
                warnings.warn(UserWarning(
                    'Unknown key "{}" found in your configuration file, ignoring.'.format(k)))

        # Update the default to match the config
        parser.set_defaults(**config)

        # Reparse the arguments
        arguments = parser.parse_args(argv)

    if arguments.script is None:
        parser.error("The -script argument must be provided, either via command-line or via the -config/-c argument.")

    # Set up email notifications if desired
    do_email = arguments.email_to is not None
    if do_email:
        email_to = arguments.email_to
        if arguments.email_from is not None:
            email_from = arguments.email_from
        else:
            email_from = arguments.email_to

        smtp_host, smtp_port = arguments.email_host.split(":")
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
    path_output = os.path.abspath(os.path.expanduser(arguments.path_output))
    path_results = os.path.join(path_output, 'results')
    path_data_processed = os.path.join(path_output, 'data_processed')
    path_log = os.path.join(path_output, 'log')
    path_qc = os.path.join(path_output, 'qc')
    path_segmanual = os.path.abspath(os.path.expanduser(arguments.path_segmanual))
    script = os.path.abspath(os.path.expanduser(arguments.script))
    path_data = os.path.abspath(os.path.expanduser(arguments.path_data))

    for pth in [path_output, path_results, path_data_processed, path_log, path_qc]:
        os.makedirs(pth, exist_ok=True)

    # Check that the script can be found
    if not os.path.exists(script):
        raise FileNotFoundError('Couldn\'t find the script script at {}'.format(script))

    # Setup overall log
    batch_log = open(os.path.join(path_log, arguments.batch_log), 'w')

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

        # Display OS

    print("INFO SYSTEM")
    print("-----------")
    platform_running = sys.platform
    if platform_running.startswith('darwin'):
        os_running = 'osx'
    elif platform_running.startswith('linux'):
        os_running = 'linux'
    elif platform_running.startswith('win32'):
        os_running = 'windows'
    else:
        os_running = platform_running
    print('OS: ' + os_running + ' (' + platform.platform() + ')')

    # Display number of CPU cores
    print('CPU cores: Available: {} | Threads used by ITK Programs: {}'.format(multiprocessing.cpu_count(), arguments.itk_threads))

    # Display RAM available
    print("RAM: Total {} MB | Available {} MB | Used {} MB".format(
        int(psutil.virtual_memory().total / 1024 / 1024),
        int(psutil.virtual_memory().available / 1024 / 1024),
        int(psutil.virtual_memory().used / 1024 / 1024),
    ))

    # Log the current arguments (in yaml because it's cleaner)
    print('\nINPUT ARGUMENTS')
    print("---------------")
    print(yaml.dump(vars(arguments)))

    # Display script version info
    print("SCRIPT")
    print("------")
    print("git commit: {}".format(__get_commit(path_to_git_folder=os.path.dirname(script))))
    print("git origin: {}".format(__get_git_origin(path_to_git_folder=os.path.dirname(script))))
    print("Copying script to output folder...")
    try:
        # Copy the script and record the new location
        script_copy = os.path.abspath(shutil.copy(script, arguments.path_output))
        print("{} -> {}".format(script, script_copy))
        script = script_copy
    except shutil.SameFileError:
        print("Input and output folder are the same. Skipping copy.")
        pass
    except IsADirectoryError:
        print("Input folder is a directory (not a file). Skipping copy.")
        pass

    print("Setting execute permissions for script file {} ...".format(arguments.script))
    script_stat = os.stat(script)
    os.chmod(script, script_stat.st_mode | S_IEXEC)

    # Display data version info
    print("\nDATA")
    print("----")
    print("git commit: {}".format(__get_commit(path_to_git_folder=path_data)))
    print("git origin: {}\n".format(__get_git_origin(path_to_git_folder=path_data)))

    # Find subjects and process inclusion/exclusions
    subject_dirs = []
    subject_flat_dirs = [f for f in os.listdir(path_data) if f.startswith(arguments.subject_prefix)]
    for isub in subject_flat_dirs:
        # Only consider folders
        if os.path.isdir(os.path.join(path_data, isub)):
            session_dirs = [f for f in os.listdir(os.path.join(path_data, isub)) if f.startswith('ses-')]
            if not session_dirs:
                # There is no session folder, so we consider only sub- directory: sub-XX
                subject_dirs.append(isub)
            else:
                # There is a session folder, so we concatenate: sub-XX/ses-YY
                session_dirs.sort()
                for isess in session_dirs:
                    subject_dirs.append(os.path.join(isub, isess))

    # Handle inclusion lists
    assert not ((arguments.include is not None) and (arguments.include_list is not None)),\
        'Only one of `include` and `include-list` can be used'

    if arguments.include is not None:
        subject_dirs = [f for f in subject_dirs if re.search(arguments.include, f) is not None]

    if arguments.include_list is not None:
        # TODO decide if we should warn users if one of their inclusions isn't around
        subject_dirs = [f for f in subject_dirs if f in arguments.include_list]

    # Handle exclusions
    assert not ((arguments.exclude is not None) and (arguments.exclude_list is not None)),\
        'Only one of `exclude` and `exclude-list` can be used'

    if arguments.exclude is not None:
        subject_dirs = [f for f in subject_dirs if re.search(arguments.exclude, f) is None]

    if arguments.exclude_list is not None:
        subject_dirs = [f for f in subject_dirs if f not in arguments.exclude_list]

    # Determine the number of jobs we can run simultaneously
    if arguments.jobs < 1:
        jobs = multiprocessing.cpu_count() + arguments.jobs
    else:
        jobs = arguments.jobs

    print("RUNNING")
    print("-------")
    print("Processing {} subjects in parallel. (Worker processes used: {}).".format(len(subject_dirs), jobs))

    # Run the jobs, recording start and end times
    start = datetime.datetime.now()

    # Trap errors to send an email if a script fails.
    try:
        with multiprocessing.Pool(jobs) as p:
            run_single_dir = functools.partial(run_single,
                                               script=script,
                                               script_args=arguments.script_args,
                                               path_segmanual=path_segmanual,
                                               path_data=path_data,
                                               path_data_processed=path_data_processed,
                                               path_results=path_results,
                                               path_log=path_log,
                                               path_qc=path_qc,
                                               itk_threads=arguments.itk_threads,
                                               continue_on_error=arguments.continue_on_error)
            results = list(p.imap(run_single_dir, subject_dirs))
    except Exception as e:
        if do_email:
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

    end = datetime.datetime.now()

    # Check for failed subjects
    fails = [sd for (sd, ret) in zip(subject_dirs, results) if ret.returncode != 0]

    if len(fails) == 0:
        status_message = '\nHooray! your batch completed successfully :-)\n'
    else:
        status_message = ('\nYour batch completed but some subjects may have not completed '
                          'successfully, please consult the logs for:\n'
                          '{}\n'.format('\n'.join(fails)))
    print(status_message)

    # Display timing
    duration = end - start
    timing_message = ('Started: {} | Ended: {} | Duration: {}\n'.format(
        start.strftime('%Hh%Mm%Ss'),
        end.strftime('%Hh%Mm%Ss'),
        (datetime.datetime.utcfromtimestamp(0) + duration).strftime('%Hh%Mm%Ss')))
    print(timing_message)

    if do_email:
        send_notification('sct_run_batch: Run completed',
                          status_message + timing_message)

    open_cmd = 'open' if sys.platform == 'darwin' else 'xdg-open'

    print('To open the Quality Control (QC) report on a web-browser, run the following:\n'
          '{} {}/index.html'.format(open_cmd, path_qc))

    if arguments.zip:
        file_zip = 'sct_run_batch_{}'.format(time.strftime('%Y%m%d%H%M%S'))
        path_tmp = os.path.join(tempfile.mkdtemp(), file_zip)
        os.makedirs(os.path.join(path_tmp, file_zip))
        for folder in [path_log, path_qc, path_results]:
            shutil.copytree(folder, os.path.join(path_tmp, file_zip, os.path.split(folder)[-1]))
        shutil.make_archive(os.path.join(path_output, file_zip), 'zip', path_tmp)
        shutil.rmtree(path_tmp)
        print("\nOutput zip archive: {}.zip".format(os.path.join(path_output, file_zip)))

    reset_streams()
    batch_log.close()


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
