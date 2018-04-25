import io
import os
import logging
import subprocess


def __get_branch():
    """
    Fallback if for some reason the value vas no set by sct_launcher
    :return:
    """

    p = subprocess.Popen(["git", "rev-parse", "--abbrev-ref", "HEAD"], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, cwd=__sct_dir__)
    output, _ = p.communicate()
    status = p.returncode

    if status == 0:
        return output
    return 'unknown'


def __get_commit():
    """
    Fallback if for some reason the value vas no set by sct_launcher
    :return:
    """
    p = subprocess.Popen(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=__sct_dir__)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        return output
    return 'unknown'


def _git_info(commit_env='SCT_COMMIT',branch_env='SCT_BRANCH'):

    sct_commit = os.getenv(commit_env, __get_commit())
    sct_branch = os.getenv(branch_env, __get_branch())


    if sct_commit is not 'unknown':
        install_type = 'git'
    else:
        install_type = 'package'

    with io.open(os.path.join(__sct_dir__, 'version.txt'), 'r') as myfile:
        version_sct = myfile.read().replace('\n', '')

    return install_type, sct_commit, sct_branch, version_sct


# Basic sct config
__sct_dir__ = os.getenv("SCT_DIR", os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
__data_dir__ = os.getenv("SCT_DATA_DIR", os.path.join(__sct_dir__, 'data'))
__version__ = '-'.join(_git_info(commit_env='SCT_COMMIT', branch_env='SCT_BRANCH'))


# statistic report level
__report_log_level__ = logging.ERROR
__report_exception_level__ = Exception
