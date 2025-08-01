#!/usr/bin/env python
#
# Wrapper to processing scripts, which loops across subjects. Data should be
# organized according to the BIDS structure and the processing script should
# make use of the some of the environment variables passed here. More details
# at: https://spine-generic.readthedocs.io/en/latest/
#
# Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import os
import sys
import pathlib
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
from typing import Sequence
from types import SimpleNamespace
import textwrap

import yaml
import psutil

from spinalcordtoolbox.utils import csi_filter
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, display_open, printv
from spinalcordtoolbox.utils.sys import send_email, init_sct, __get_commit, __get_git_origin, __version__, __sct_dir__, set_loglevel
from spinalcordtoolbox.utils.fs import Tee

from stat import S_IEXEC

csi_filter.register_codec()


def get_parser():
    parser = SCTArgumentParser(
        description='Wrapper to processing scripts, which loops across subjects. Subjects should be organized as '
                    'folders within a single directory. We recommend following the BIDS convention '
                    '(https://bids.neuroimaging.io/). The processing script should accept a subject directory '
                    'as its only argument. Additional information is passed via environment variables and the '
                    'arguments passed via `-script-args`. If the script or the input data are located within a '
                    'git repository, the git commit is displayed. If the script or data have changed since the latest '
                    'commit, the symbol "*" is added after the git commit number. If no git repository is found, the '
                    'git commit version displays "?!?". The script is copied on the output folder (`-path-out`).'
    )

    parser.add_argument('-config', '-c',
                        help=textwrap.dedent("""
        A json (`.json`) or yaml (`.yml`/`.yaml`) file with arguments. All arguments to the configuration file are the same as the command line arguments, except all dashes (`-`) are replaced with underscores (`_`). Using command line flags can be used to override arguments provided in the configuration file, but this is discouraged. Please note that while quotes are optional for strings in YAML omitting them may cause parse errors.

        Note that for the `"exclude_list"` (or `"include_list"`) argument you can exclude/include entire subjects or individual sessions; see examples below.

        Example YAML configuration:
          ```
          path_data    : "~/sct_data"
          path_output  : "~/pipeline_results"
          script       : "nature_paper_analysis.sh"
          jobs         : -1
          exclude_list : ["sub-01/ses-01", "sub-02", "ses-03"]  # this will exclude ses-01 for sub-01, all sessions for sub-02 and ses-03 for all subjects
          ```

        Example JSON configuration:
          ```
          {
              "path_data"   : "~/sct_data",
              "path_output" : "~/pipeline_results",
              "script"      : "nature_paper_analysis.sh",
              "jobs"        : -1,
              "exclude_list" : ["sub-01/ses-01", "sub-02", "ses-03"]
          }
          ```
    """))  # noqa: E501 (line too long)
    parser.add_argument('-jobs', type=int, default=1,
                        help=textwrap.dedent("""
                            The number of jobs to run in parallel. Either an integer greater than or equal to one specifying the number of cores, 0 or a negative integer specifying number of cores minus that number. For example `-jobs -1` will run with all the available cores minus one job in parallel. Set `-jobs 0` to use all available cores.

                            This argument enables process-based parallelism, while `-itk-threads` enables thread-based parallelism. You may need to tweak both to find a balance that works best for your system."""),  # noqa: E501 (line too long)
                        metavar=Metavar.int)
    parser.add_argument('-itk-threads', type=int, default=1,
                        help=textwrap.dedent("""
                            Sets the environment variable `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`.

                            Number of threads to use for ITK based programs including ANTs. Increasing this can provide a performance boost for high-performance (multi-core) computing environments. However, increasing the number of threads may also result in a large increase in memory.

                            This argument enables thread-based parallelism, while `-jobs` enables process-based parallelism. You may need to tweak both to find a balance that works best for your system."""),  # noqa: E501 (line too long)
                        metavar=Metavar.int)
    parser.add_argument('-path-data', help=textwrap.dedent("""
                            Setting for environment variable: `PATH_DATA`

                            Path containing subject directories in a consistent format)""")),
    parser.add_argument('-subject-prefix', default='sub-',
                        help='Subject prefix, defaults to "sub-" which is the prefix used for BIDS directories. '
                        'If the subject directories do not share a common prefix, an empty string can be '
                        'passed here.')
    parser.add_argument('-path-output', default='.',
                        help=textwrap.dedent(f"""
                            Base directory for environment variables:

                              - `PATH_DATA_PROCESSED={os.path.join('<path-output>', 'data_processed')}`
                              - `PATH_RESULTS={os.path.join('<path-output>', 'results')}`
                              - `PATH_QC={os.path.join('<path-output>', 'qc')}`
                              - `PATH_LOG={os.path.join('<path-output>', 'log')}`

                            Which are respectively output paths for the processed data, results, quality control (QC) and logs"""))
    parser.add_argument('-batch-log', default='sct_run_batch_log.txt',
                        help='A log file for all terminal output produced by this script (not '
                        'necessarily including the individual job outputs. File will be relative '
                        'to "<path-output>/log".')
    parser.add_argument('-include',
                        help='Optional regex used to filter the list of subject directories. Only process '
                        'a subject if they match the regex. Inclusions are processed before exclusions. '
                        'Cannot be used with `include-list`.')
    parser.add_argument('-include-list',
                        help=textwrap.dedent("""
                            Optional space separated list of subjects or sessions to include. Only process subjects or sessions if they are on this list. Inclusions are processed before exclusions. Cannot be used with `-include`. You can combine subjects and sessions; see examples.

                            Examples: `-include-list sub-001 sub-002` or `-include-list sub-001/ses-01 ses-02`"""),  # noqa: E501 (line too long)
                        nargs='+')
    parser.add_argument('-exclude',
                        help='Optional regex used to filter the list of subject directories. Only process '
                        'a subject if they do not match the regex. Exclusions are processed '
                        'after inclusions. Cannot be used with `exclude-list`')
    parser.add_argument('-exclude-list',
                        help=textwrap.dedent("""
                            Optional space separated list of subjects or sessions to exclude. Only process subjects or sessions if they are not on this list. Inclusions are processed before exclusions. Cannot be used with `-exclude`. You can combine subjects and sessions; see examples.

                            Examples: `-exclude-list sub-003 sub-004` or `-exclude-list sub-003/ses-01 ses-02`
        """),  # noqa: E501 (line too long)
                        nargs='+')
    parser.add_argument('-ignore-ses', action='store_true',
                        help="By default, if 'ses' subfolders are present, then 'sct_run_batch' will run the script "
                             "within each individual 'ses' subfolder. Passing `-ignore-ses` will change the behavior "
                             "so that 'sct_run_batch' will not go into each 'ses' folder. Instead, it will run the "
                             "script on just the top-level subject folders.")
    parser.add_argument('-path-segmanual', default='.',
                        help=textwrap.dedent("""
                            Setting for environment variable: PATH_SEGMANUAL
                            A path containing manual segmentations to be used by the script program."""))
    parser.add_argument('-script-args', default='',
                        help=textwrap.dedent("""
                            A quoted string with extra arguments to pass to the script. For example: `sct_run_batch -path-data data/ -script process_data.sh -script-args "ARG1 ARG2"`.
    
                            The arguments are retrieved by a script as `${2}`, `${3}`, etc.

                            - Note: `${1}` is reserved for the subject folder name, which is retrieved automatically.
                            - Note: Do not use `~` in the path. Use `${HOME}` instead.)"""))  # noqa 501 (line too long)
    parser.add_argument('-email-to',
                        help='Optional email address where sct_run_batch can send an alert on completion of the '
                        'batch processing.')
    parser.add_argument('-email-from',
                        help='Optional alternative email to use to send the email. Defaults to the same address as '
                             '`-email-to`')
    parser.add_argument('-email-host', default='smtp.gmail.com:587',
                        help="Optional smtp server and port to use to send the email. Defaults to gmail's server. Note"
                             " that gmail server requires 'Less secure apps access' to be turned on, which can be done "
                             " at https://myaccount.google.com/security")
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


def _parse_dataset_directory(path_data, subject_prefix="sub-", ignore_ses=False):
    """
    Parse a dataset directory to find subject directories (and session subdirectories, if present).

    Notes:
        - The dataset is assumed to be structured in a (somewhat) BIDS-compliant way, see:
          https://bids-specification.readthedocs.io/en/stable/02-common-principles.html#file-name-structure
        - This function is a rudimentary version of the library PyBIDS: https://github.com/bids-standard/pybids.
          TODO: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3415
    """
    dirs = []
    subject_dirs = sorted([d.name for d in os.scandir(path_data)
                           if d.name.startswith(subject_prefix)
                           and d.is_dir()])
    for sub_dir in subject_dirs:
        session_dirs = sorted([d.name for d in os.scandir(os.path.join(path_data, sub_dir))
                               if d.name.startswith('ses-')
                               and d.is_dir()])
        if session_dirs and not ignore_ses:
            # There is a 'ses-' subdirectory AND arguments.ignore_ses = False, so we concatenate: e.g. sub-XX/ses-YY
            for sess_dir in session_dirs:
                dirs.append(os.path.join(sub_dir, sess_dir))
        else:
            # Otherwise, consider only 'sub-' directories and don't include 'ses-' subdirectories: e.g. sub-XX
            dirs.append(sub_dir)

    return dirs


def _filter_directories(dir_list, include=None, include_list=None, exclude=None, exclude_list=None):
    """
    Filter a list of directories using inclusion/exclusion regex patterns or explicit lists.

    NB: Only one of [include, include_list] and only one of [exclude, exclude_list] should be passed.
        (Currently, this requirement is handled at the argument-parsing level, because we use `parser.error`.)
    """
    # Handle inclusions (regex OR explicit list, but not both)
    if include is not None:
        dir_list = [f for f in dir_list if re.search(include, f)]
    elif include_list is not None:
        dir_list = [f for f in dir_list
                    # Check if include_list specified entire path (e.g. "sub-01/ses-01")
                    if f in include_list
                    # Check if include_list specified a subdirectory (e.g. just "sub-01" or just "ses-01")
                    or any(p in include_list for p in pathlib.Path(f).parts)]

    # Handle exclusions (regex OR explicit list, but not both)
    if exclude is not None:
        dir_list = [f for f in dir_list if not re.search(exclude, f)]
    elif exclude_list is not None:
        dir_list = [f for f in dir_list
                    # Check if exclude_list specified entire path (e.g. "sub-01/ses-01")
                    if f not in exclude_list
                    # Check if exclude_list specified a subdirectory (e.g. just "sub-01" or just "ses-01")
                    and all(p not in exclude_list for p in pathlib.Path(f).parts)]

    return dir_list


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
    if 'SCT_DIR' not in envir:
        envir['SCT_DIR'] = __sct_dir__

    cmd = [script_full, subj_dir] + script_args.split(' ')

    # Make sure that a Windows-compatible bash port is used to run shell scripts
    if sys.platform.startswith("win32"):
        with open(script_full) as f:
            first_line = f.readline()
        shebang_pattern = re.compile("^#!.*/.*sh\b.*")
        if script_full.endswith('.sh') or shebang_pattern.match(first_line):
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

        if res.returncode != 0:
            raise ValueError(f"Processing of subject {subject} failed")
    except Exception as e:
        process_completed = 'res' in locals()
        res = res if process_completed else SimpleNamespace(returncode=-1)
        process_suceeded = res.returncode == 0

        if not process_suceeded and os.path.exists(log_file):
            # If the process didn't complete or succeed rename the log file to indicate
            # the error
            print(f"An error occurred while processing subject '{subject_session}'. "
                  f"Renaming log file {log_file} to {err_file}.")
            os.rename(log_file, err_file)
            with open(err_file, 'a') as err_log:
                print(f"Error: {e}", file=err_log)

        if process_suceeded or continue_on_error:
            return res
        else:
            raise e

    return res


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    # See if there's a configuration file and import those options
    if arguments.config is not None:
        print('configuring')
        with open(arguments.config, 'r') as conf:
            _, ext = os.path.splitext(arguments.config)
            if ext == '.json':
                config = json.load(conf)
            elif ext == '.yml' or ext == '.yaml':
                config = yaml.load(conf, Loader=yaml.Loader)
            else:
                raise ValueError('Unrecognized configuration file type: {}'.format(ext))

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
    batch_log = open(os.path.join(path_log, arguments.batch_log), 'w', encoding='csi-filter')

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
    if os.path.isdir(script):
        print("Input folder is a directory (not a file). Skipping copy.")
    else:
        try:
            # Copy the script and record the new location
            script_copy = os.path.abspath(shutil.copy(script, path_output))
            print("{} -> {}".format(script, script_copy))
            script = script_copy
        except shutil.SameFileError:
            print("Input and output folder are the same. Skipping copy.")
            pass

    print("Setting execute permissions for script file {} ...".format(arguments.script))
    script_stat = os.stat(script)
    os.chmod(script, script_stat.st_mode | S_IEXEC)

    # Display data version info
    print("\nDATA")
    print("----")
    print("git commit: {}".format(__get_commit(path_to_git_folder=path_data)))
    print("git origin: {}\n".format(__get_git_origin(path_to_git_folder=path_data)))

    subject_dirs = _parse_dataset_directory(path_data, arguments.subject_prefix, arguments.ignore_ses)

    if (arguments.include is not None) and (arguments.include_list is not None):
        parser.error('Only one of `include` and `include-list` can be used')

    if (arguments.exclude is not None) and (arguments.exclude_list is not None):
        parser.error('Only one of `exclude` and `exclude-list` can be used')

    subject_dirs = _filter_directories(subject_dirs,
                                       include=arguments.include, include_list=arguments.include_list,
                                       exclude=arguments.exclude, exclude_list=arguments.exclude_list)

    # Determine the number of jobs we can run simultaneously
    if arguments.jobs < 1:
        jobs = multiprocessing.cpu_count() + arguments.jobs
    else:
        jobs = arguments.jobs

    print("RUNNING")
    print("-------")
    print("Processing {} subjects in total. (Number of subjects processed in parallel: {}).".format(len(subject_dirs), jobs))

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
        printv(status_message, verbose=verbose, type='info')
    else:
        status_message = ('\nYour batch completed but some subjects may have not completed '
                          'successfully, please consult the logs for:\n'
                          '{}\n'.format('\n'.join(fails)))
        printv(status_message, verbose=verbose, type='error')

    # Display timing
    duration = end - start
    timing_message = ('Started: {} | Ended: {} | Duration: {}\n'.format(
        start.strftime('%F %Hh%Mm%Ss'),
        end.strftime('%Hh%Mm%Ss'),
        (datetime.datetime.utcfromtimestamp(0) + duration).strftime('%Hh%Mm%Ss')))
    print(timing_message)

    if do_email:
        send_notification('sct_run_batch: Run completed',
                          status_message + timing_message)

    display_open(file=os.path.join(path_qc, "index.html"),
                 message="To open the Quality Control (QC) report in a web-browser")

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
