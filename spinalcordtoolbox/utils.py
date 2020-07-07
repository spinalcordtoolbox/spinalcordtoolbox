#!/usr/bin/env python
# -*- coding: utf-8
# Collection of useful functions


import io
import os
import re
import tempfile
import datetime
import logging
import argparse
import subprocess
import shutil
import tqdm
from enum import Enum

logger = logging.getLogger(__name__)


# TODO: add test


class ActionCreateFolder(argparse.Action):
    """
    Custom action: creates a new folder if it does not exist. If the folder
    already exists, do nothing.

    The action will strip off trailing slashes from the folder's name.
    Source: https://argparse-actions.readthedocs.io/en/latest/
    """
    @staticmethod
    def create_folder(folder_name):
        """
        Create a new directory if not exist. The action might throw
        OSError, along with other kinds of exception
        """
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        folder_name = os.path.normpath(folder_name)
        return folder_name

    def __call__(self, parser, namespace, values, option_string=None):
        if type(values) == list:
            folders = list(map(self.create_folder, values))
        else:
            folders = self.create_folder(values)

        # Add the attribute
        setattr(namespace, self.dest, folders)


class Metavar(Enum):
    """
    This class is used to display intuitive input types via the metavar field of argparse
    """
    file = "<file>"
    str = "<str>"
    folder = "<folder>"
    int = "<int>"
    list = "<list>"
    float = "<float>"

    def __str__(self):
        return self.value


class SmartFormatter(argparse.HelpFormatter):
    """
    Custom formatter that inherits from HelpFormatter, which adjusts the default width to the current Terminal size,
    and that gives the possibility to bypass argparse's default formatting by adding "R|" at the beginning of the text.
    Inspired from: https://pythonhosted.org/skaff/_modules/skaff/cli.html
    """
    def __init__(self, *args, **kw):
        self._add_defaults = None
        super(SmartFormatter, self).__init__(*args, **kw)
        # Update _width to match Terminal width
        try:
            self._width = shutil.get_terminal_size()[0]
        except (KeyError, ValueError):
            logger.warning('Not able to fetch Terminal width. Using default: %s'.format(self._width))

    # this is the RawTextHelpFormatter._fill_text
    def _fill_text(self, text, width, indent):
        # print("splot",text)
        if text.startswith('R|'):
            paragraphs = text[2:].splitlines()
            rebroken = [argparse._textwrap.wrap(tpar, width) for tpar in paragraphs]
            rebrokenstr = []
            for tlinearr in rebroken:
                if (len(tlinearr) == 0):
                    rebrokenstr.append("")
                else:
                    for tlinepiece in tlinearr:
                        rebrokenstr.append(tlinepiece)
            return '\n'.join(rebrokenstr)  # (argparse._textwrap.wrap(text[2:], width))
        return argparse.RawDescriptionHelpFormatter._fill_text(self, text, width, indent)

    # this is the RawTextHelpFormatter._split_lines
    def _split_lines(self, text, width):
        if text.startswith('R|'):
            lines = text[2:].splitlines()
            offsets = [re.match("^[ \t]*", l).group(0) for l in lines]
            wrapped = []
            for i in range(len(lines)):
                li = lines[i]
                o = offsets[i]
                ol = len(o)
                init_wrap = argparse._textwrap.fill(li, width).splitlines()
                first = init_wrap[0]
                rest = "\n".join(init_wrap[1:])
                rest_wrap = argparse._textwrap.fill(rest, width - ol).splitlines()
                offset_lines = [o + wl for wl in rest_wrap]
                wrapped = wrapped + [first] + offset_lines
            return wrapped
        return argparse.HelpFormatter._split_lines(self, text, width)


def add_suffix(fname, suffix):
    """
    Add suffix between end of file name and extension.

    :param fname: absolute or relative file name. Example: t2.nii
    :param suffix: suffix. Example: _mean
    :return: file name with suffix. Example: t2_mean.nii

    Examples:

    - add_suffix(t2.nii, _mean) -> t2_mean.nii
    - add_suffix(t2.nii.gz, a) -> t2a.nii.gz
    """
    stem, ext = splitext(fname)
    return os.path.join(stem + suffix + ext)


def check_exe(name):
    """
    Ensure that a program exists
    :type name: string
    :param name: name or path to program
    :return: path of the program or None
    """
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(name)
    if fpath and is_exe(name):
        return fpath
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, name)
            if is_exe(exe_file):
                return exe_file

    return None


def parse_num_list(str_num):
    """
    Parse numbers in string based on delimiter: , or :
    Examples:
      '' -> []
      '1,2,3' -> [1, 2, 3]
      '1:3,4' -> [1, 2, 3, 4]
      '1,1:4' -> [1, 2, 3, 4]
    :param str_num: string
    :return: list of ints
    """
    list_num = list()

    if not str_num:
        return list_num

    elements = str_num.split(",")
    for element in elements:
        m = re.match(r"^\d+$", element)
        if m is not None:
            val = int(element)
            if val not in list_num:
                list_num.append(val)
            continue
        m = re.match(r"^(?P<first>\d+):(?P<last>\d+)$", element)
        if m is not None:
            a = int(m.group("first"))
            b = int(m.group("last"))
            list_num += [x for x in range(a, b + 1) if x not in list_num]
            continue
        raise ValueError("unexpected group element {} group spec {}".format(element, str_num))

    return list_num


def parse_num_list_inv(list_int):
    """
    Take a list of numbers and output a string that reduce this list based on delimiter: ; or :
    Note: we use ; instead of , for compatibility with csv format.
    Examples:
      [] -> ''
      [1, 2, 3] --> '1:3'
      [1, 2, 3, 5] -> '1:3;5'
    :param list_int: list of ints
    :return: str_num: string
    """
    # deal with empty list
    if not list_int or list_int is None:
        return ''
    # Sort list in increasing number
    list_int = sorted(list_int)
    # initialize string
    str_num = str(list_int[0])
    colon_is_present = False
    # Loop across list elements and build string iteratively
    for i in range(1, len(list_int)):
        # if previous element is the previous integer: I(i-1) = I(i)-1
        if list_int[i] == list_int[i - 1] + 1:
            # if ":" already there, update the last chars (based on the number of digits)
            if colon_is_present:
                str_num = str_num[:-len(str(list_int[i - 1]))] + str(list_int[i])
            # if not, add it along with the new int value
            else:
                str_num += ':' + str(list_int[i])
                colon_is_present = True
        # I(i-1) != I(i)-1
        else:
            str_num += ';' + str(list_int[i])
            colon_is_present = False

    return str_num


def splitext(fname):
    """
    Split a fname (folder/file + ext) into a folder/file and extension.

    Note: for .nii.gz the extension is understandably .nii.gz, not .gz
    (``os.path.splitext()`` would want to do the latter, hence the special case).
    """
    dir, filename = os.path.split(fname)
    for special_ext in ['.nii.gz', '.tar.gz']:
        if filename.endswith(special_ext):
            stem, ext = filename[:-len(special_ext)], special_ext
            return os.path.join(dir, stem), ext
    # If no special case, behaves like the regular splitext
    stem, ext = os.path.splitext(filename)
    return os.path.join(dir, stem), ext


def tmp_create(basename=None):
    """Create temporary folder and return its path
    """
    prefix = "sct-%s-" % datetime.datetime.now().strftime("%Y%m%d%H%M%S.%f")
    if basename:
        prefix += "%s-" % basename
    tmpdir = tempfile.mkdtemp(prefix=prefix)
    logger.info("Creating temporary folder (%s)" % tmpdir)
    return tmpdir


def send_email(addr_to, addr_from, passwd, subject, message='', filename=None, html=False, smtp_host=None, smtp_port=None, login=None):
    if smtp_host is None:
        smtp_host = os.environ.get("SCT_SMTP_SERVER", "smtp.gmail.com")
    if smtp_port is None:
        smtp_port = int(os.environ.get("SCT_SMTP_PORT", 587))
    if login is None:
        login = addr_from

    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    if html:
        msg = MIMEMultipart("alternative")
    else:
        msg = MIMEMultipart()

    msg['From'] = addr_from
    msg['To'] = addr_to
    msg['Subject'] = subject

    body = message
    if not isinstance(body, bytes):
        body = body.encode("utf-8")

    body_html = """
<html><pre style="font: monospace"><body>
{}
</body></pre></html>""".format(body).encode()

    if html:
        msg.attach(MIMEText(body_html, 'html', "utf-8"))

    msg.attach(MIMEText(body, 'plain', "utf-8"))

    # filename = "NAME OF THE FILE WITH ITS EXTENSION"
    if filename:
        attachment = open(filename, "rb")
        part = MIMEBase('application', 'octet-stream')
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
        msg.attach(part)

    # send email
    server = smtplib.SMTP(smtp_host, smtp_port)
    server.starttls()
    server.login(login, passwd)
    text = msg.as_string()
    server.sendmail(addr_from, addr_to, text)
    server.quit()


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
        return output.decode().strip()


def __get_commit():
    """
    :return: git commit ID, with trailing '*' if modified
    """
    p = subprocess.Popen(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=__sct_dir__)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        commit = output.decode().strip()
    else:
        commit = "?!?"

    p = subprocess.Popen(["git", "status", "--porcelain"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=__sct_dir__)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        unclean = True
        for line in output.decode().strip().splitlines():
            line = line.rstrip()
            if line.startswith("??"):  # ignore ignored files, they can't hurt
                continue
            break
        else:
            unclean = False
        if unclean:
            commit += "*"

    return commit


def _git_info(commit_env='SCT_COMMIT', branch_env='SCT_BRANCH'):

    sct_commit = os.getenv(commit_env, "unknown")
    sct_branch = os.getenv(branch_env, "unknown")
    if check_exe("git") and os.path.isdir(os.path.join(__sct_dir__, ".git")):
        sct_commit = __get_commit() or sct_commit
        sct_branch = __get_branch() or sct_branch

    if sct_commit != 'unknown':
        install_type = 'git'
    else:
        install_type = 'package'

    with io.open(os.path.join(__sct_dir__, 'version.txt'), 'r') as f:
        version_sct = f.read().rstrip()

    return install_type, sct_commit, sct_branch, version_sct


def _version_string():
    install_type, sct_commit, sct_branch, version_sct = _git_info()
    if install_type == "package":
        return version_sct
    else:
        return "{install_type}-{sct_branch}-{sct_commit}".format(**locals())


def sct_progress_bar(*args, **kwargs):
    """Thin wrapper around `tqdm.tqdm` which checks `SCT_PROGRESS_BAR` muffling the progress
       bar if the user sets it to `no`, `off`, or `false` (case insensitive)."""
    do_pb = os.environ.get('SCT_PROGRESS_BAR', 'yes')
    if do_pb.lower() in ['off', 'no', 'false']:
        kwargs['disable'] = True

    return tqdm.tqdm(*args, **kwargs)


__sct_dir__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
__version__ = _version_string()
__data_dir__ = os.path.join(__sct_dir__, 'data')
__deepseg_dir__ = os.path.join(__data_dir__, 'deepseg_models')
