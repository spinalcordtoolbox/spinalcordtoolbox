# System related utilities

import io
import sys
import os
import shutil
import logging
import subprocess
import time
import shlex
import atexit
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import inspect

import tqdm

logger = logging.getLogger(__name__)


def stylize(string, styles):
    """Helper function that mimics colored.stylize to reduce boilerplate when coloring text."""
    if not isinstance(styles, list):
        styles = [styles]
    style_codes = "".join([getattr(ANSIColors16, style, ANSIColors16.ResetAll) for style in styles])
    return style_codes + string + ANSIColors16.ResetAll


class ANSIColors16(object):
    """This class defines the ANSI color escape codes for terminals that support 16 colors.

    Notes:
        - Most terminals support 8 colors (base color set) or 16 colors (base colors + light colors).
        - We use these codes instead of dedicated color packages (colored, colorama) because those packages
          are meant for 256-color coloring, which only some terminals support. (Notably, the Windows Command Prompt
          does not support 256 colors.)
        - Further reading: https://www.lihaoyi.com/post/BuildyourownCommandLinewithANSIescapecodes.html#rich-text
        - Source for codes: https://pkg.go.dev/github.com/whitedevops/colors
    """
    ResetAll = "\033[0m"

    Bold = "\033[1m"
    Dim = "\033[2m"
    Underlined = "\033[4m"
    Blink = "\033[5m"
    Reverse = "\033[7m"
    Hidden = "\033[8m"

    ResetBold = "\033[21m"
    ResetDim = "\033[22m"
    ResetUnderlined = "\033[24m"
    ResetBlink = "\033[25m"
    ResetReverse = "\033[27m"
    ResetHidden = "\033[28m"

    Default = "\033[39m"
    Black = "\033[30m"
    Red = "\033[31m"
    Green = "\033[32m"
    Yellow = "\033[33m"
    Blue = "\033[34m"
    Magenta = "\033[35m"
    Cyan = "\033[36m"
    LightGray = "\033[37m"
    DarkGray = "\033[90m"
    LightRed = "\033[91m"
    LightGreen = "\033[92m"
    LightYellow = "\033[93m"
    LightBlue = "\033[94m"
    LightMagenta = "\033[95m"
    LightCyan = "\033[96m"
    White = "\033[97m"

    BackgroundDefault = "\033[49m"
    BackgroundBlack = "\033[40m"
    BackgroundRed = "\033[41m"
    BackgroundGreen = "\033[42m"
    BackgroundYellow = "\033[43m"
    BackgroundBlue = "\033[44m"
    BackgroundMagenta = "\033[45m"
    BackgroundCyan = "\033[46m"
    BackgroundLightGray = "\033[47m"
    BackgroundDarkGray = "\033[100m"
    BackgroundLightRed = "\033[101m"
    BackgroundLightGreen = "\033[102m"
    BackgroundLightYellow = "\033[103m"
    BackgroundLightBlue = "\033[104m"
    BackgroundLightMagenta = "\033[105m"
    BackgroundLightCyan = "\033[106m"
    BackgroundWhite = "\033[107m"


if os.getenv('SENTRY_DSN', None):
    # do no import if Sentry is not set (i.e., if variable SENTRY_DSN is not defined)
    import raven


def get_caller_module():
    """Return the first non-`utils.sys` module in the stack (to see where `utils.sys` is being called from)."""
    for frame in inspect.stack():
        mod = inspect.getmodule(frame[0])
        if mod.__file__ != __file__:
            break
    return mod


def set_loglevel(verbose):
    """
    Use SCT's verbosity values to set the logging level.

    The behavior of this function changes depending on how the caller module was invoked:
       1. If the module was invoked via the command line, the root (global) logger will be used.
       2. If the module was invoked via code (e.g. sct_script.main()), then only its local logger will be used.

    :verbosity: Verbosity value, typically from argparse (args.v). Values must adhere
    one of two schemes:
       - [0, 1, 2], which corresponds to [WARNING, INFO, DEBUG]. (Older scheme)
       - [False, True], which corresponds to [INFO, DEBUG].      (Newer scheme)
    """
    dict_log_levels = {
        '0': 'WARNING', '1': 'INFO', '2': 'DEBUG',  # Older scheme
        'False': 'INFO', 'True': 'DEBUG',           # Newer scheme (See issue #2676)
    }

    if str(verbose) not in dict_log_levels.keys():
        raise ValueError(f"Invalid verbosity level '{verbose}' does not map to a log level, so cannot set.")

    log_level = dict_log_levels[str(verbose)]

    # Set logging level for this file
    logger.setLevel(getattr(logging, log_level))

    # Set logging level for the file that called this function
    caller_module_name = get_caller_module().__name__
    caller_logger = logging.getLogger(caller_module_name)
    caller_logger.setLevel(getattr(logging, log_level))

    # Set the logging level globally, but only when scripts are directly invoked (e.g. from the command line)
    if caller_module_name == "__main__":
        logging.root.setLevel(getattr(logging, log_level))
    else:
        # NB: Nothing will be set if we're calling a CLI script in-code, i.e. <sct_cli_script>.main(). This keeps
        # the loglevel changes from leaking: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3341
        pass


def removesuffix(self: str, suffix: str) -> str:
    """
    Source: https://www.python.org/dev/peps/pep-0616/

    TODO: Replace with built-in str.removesuffix method after upgrading to Python 3.9
    """
    # suffix='' should not call self[:-0].
    if suffix and self.endswith(suffix):
        return self[:-len(suffix)]
    else:
        return self[:]


# TODO: add test
def init_sct():
    """
    Initialize SCT for typical terminal usage, including logging initialization, Sentry
    configuration, as well as a status message with the SCT version and the command run.
    """

    def _format_wrap(old_format):
        def _format(record):
            res = old_format(record)
            if record.levelno >= logging.ERROR:
                res = "\x1B[31;1m{}\x1B[0m".format(res)
            elif record.levelno >= logging.WARNING:
                res = "\x1B[33m{}\x1B[0m".format(res)
            else:
                pass
            return res
        return _format

    # Initialize logging
    set_loglevel(verbose=False)  # False => "INFO". For "DEBUG", must be called again with verbose=True.
    hdlr = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter()
    fmt.format = _format_wrap(fmt.format)
    hdlr.setFormatter(fmt)
    logging.root.addHandler(hdlr)

    # Sentry config
    init_error_client()
    if os.environ.get("SCT_TIMER", None) is not None:
        add_elapsed_time_counter()

    # Display SCT version
    logger.info('\n--\nSpinal Cord Toolbox ({})\n'.format(__version__))

    # Display command (Only if called from CLI: check for .py in first arg)
    # Use next(iter()) to not fail on empty list (vs. sys.argv[0])
    if '.py' in next(iter(sys.argv), None):
        script = removesuffix(os.path.basename(sys.argv[0]), ".py")
        arguments = ' '.join(sys.argv[1:])
        logger.info(f"{script} {arguments}\n"
                    f"--\n")


def add_elapsed_time_counter():
    class Timer():
        def __init__(self):
            self._t0 = time.time()

        def atexit(self):
            print("Elapsed time: %.3f seconds" % (time.time() - self._t0))
    t = Timer()
    atexit.register(t.atexit)


def traceback_to_server(client):
    """
    Send all traceback children of Exception to sentry
    """

    def excepthook(exctype, value, traceback):
        if issubclass(exctype, Exception):
            client.captureException(exc_info=(exctype, value, traceback))
        sys.__excepthook__(exctype, value, traceback)

    sys.excepthook = excepthook


def init_error_client():
    if os.getenv('SENTRY_DSN'):
        logger.debug('Configuring sentry report')
        try:
            client = raven.Client(
                release=__version__,
                processors=(
                    'raven.processors.RemoveStackLocalsProcessor',
                    'raven.processors.SanitizePasswordsProcessor'),
            )
            server_log_handler(client)
            traceback_to_server(client)
            old_exitfunc = sys.exitfunc

            def exitfunc():
                sent_something = False
                try:
                    # implementation-specific
                    for handler, args, kw in atexit._exithandlers:
                        if handler.__module__.startswith("raven."):
                            sent_something = True
                except:
                    pass
                old_exitfunc()
                if sent_something:
                    print("Note: you can opt out of Sentry reporting by editing the file ${SCT_DIR}/bin/sct_launcher and delete the line starting with \"export SENTRY_DSN\"")
            sys.exitfunc = exitfunc
        except raven.exceptions.InvalidDsn:
            # This could happen if sct staff change the dsn
            logger.debug('Sentry DSN not valid anymore, not reporting errors')


def server_log_handler(client):
    """ Adds sentry log handler to the logger

    :return: the sentry handler
    """
    from raven.handlers.logging import SentryHandler

    sh = SentryHandler(client=client, level=logging.ERROR)

    # Don't send Sentry events for command-line usage errors
    old_emit = sh.emit

    def emit(self, record):
        if not record.message.startswith("Command-line usage error:"):
            return old_emit(record)

    sh.emit = lambda x: emit(sh, x)

    fmt = ("[%(asctime)s][%(levelname)s] %(filename)s: %(lineno)d | "
           "%(message)s")
    formatter = logging.Formatter(fmt=fmt, datefmt="%H:%M:%S")
    formatter.converter = time.gmtime
    sh.setFormatter(formatter)

    logger.addHandler(sh)
    return sh


def send_email(addr_to, addr_from, subject, message='', passwd=None, filename=None, html=False, smtp_host=None, smtp_port=None, login=None):
    if smtp_host is None:
        smtp_host = os.environ.get("SCT_SMTP_SERVER", "smtp.gmail.com")
    if smtp_port is None:
        smtp_port = int(os.environ.get("SCT_SMTP_PORT", 587))
    if login is None:
        login = addr_from

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
    if passwd is not None:
        server.login(login, passwd)
    text = msg.as_string()
    server.sendmail(addr_from, addr_to, text)
    server.quit()


def sct_progress_bar(*args, **kwargs):
    """Thin wrapper around `tqdm.tqdm` which checks `SCT_PROGRESS_BAR` muffling the progress
       bar if the user sets it to `no`, `off`, or `false` (case insensitive)."""
    do_pb = os.environ.get('SCT_PROGRESS_BAR', 'yes')
    if do_pb.lower() in ['off', 'no', 'false']:
        kwargs['disable'] = True

    return tqdm.tqdm(*args, **kwargs)


def _which_sct_binaries():
    """
    :return name of the sct binaries to use on this platform
    """

    if sys.platform.startswith("darwin"):
        return "binaries_osx"
    elif sys.platform.startswith("win32"):
        return "binaries_win"
    else:
        return "binaries_linux"


def list2cmdline(lst):
    return " ".join(shlex.quote(x) for x in lst)


def run_proc(cmd, verbose=1, raise_exception=True, cwd=None, env=None, is_sct_binary=False):
    if cwd is None:
        cwd = os.getcwd()

    if env is None:
        env = os.environ

    if is_sct_binary:
        if not os.path.isdir(__bin_dir__):
            run_proc(["sct_download_data", "-d", _which_sct_binaries(), "-k"])

        name = cmd[0] if isinstance(cmd, list) else cmd.split(" ", 1)[0]
        path = os.path.join(__bin_dir__, name)

        if isinstance(cmd, list):
            cmd[0] = path
        elif isinstance(cmd, str):
            rem = cmd.split(" ", 1)[1:]
            cmd = path if len(rem) == 0 else "{} {}".format(path, rem[0])

    if isinstance(cmd, str):
        cmdline = cmd
    else:
        cmdline = list2cmdline(cmd)

    if verbose:
        printv("%s # in %s" % (cmdline, cwd), 1, 'code')

    shell = isinstance(cmd, str)

    process = subprocess.Popen(cmd, shell=shell, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    output_final = ''
    while True:
        # Watch out for deadlock!!!
        output = process.stdout.readline().decode("utf-8")
        if output == '' and process.poll() is not None:
            break
        if output:
            if verbose == 2:
                logger.debug(f"output => {output.strip()}")
            output_final += output.strip() + '\n'

    status = process.returncode
    output = output_final.rstrip()

    if status != 0 and raise_exception:
        raise RuntimeError(output)

    return status, output


def printv(string, verbose=1, type='normal', file=None):
    """
    Enables to print color-coded messages, depending on verbose status. Only use in command-line programs (e.g.,
    sct_propseg).
    """
    colors = {'normal': ANSIColors16.ResetAll, 'info': ANSIColors16.LightGreen,
              'warning': ANSIColors16.LightYellow + ANSIColors16.Bold,
              'error': ANSIColors16.LightRed + ANSIColors16.Bold,
              'code': ANSIColors16.LightBlue, 'bold': ANSIColors16.Bold, 'process': ANSIColors16.LightMagenta}

    if file is None:
        # replicate the logic from print()
        # so that we can check file.isatty()
        file = sys.stdout

    if verbose:
        # The try/except is there in case file does not have isatty field (it did happen to me)
        try:
            # Print color only if the output is the terminal
            if file.isatty():
                color = colors.get(type, ANSIColors16.ResetAll)
                print(color + string + ANSIColors16.ResetAll, file=file)
            else:
                print(string, file=file)
        except Exception:
            print(string)

    if type == 'error':
        sys.exit(1)


def sct_dir_local_path(*args):
    """Construct a directory path relative to __sct_dir__"""
    return os.path.join(__sct_dir__, *args)


def sct_test_path(*args):
    """Construct a directory path relative to the sct testing data. Consults
    the SCT_TESTING_DATA environment variable.
    If unset, use the default dataset location from sct_download_data."""

    test_path = os.environ.get('SCT_TESTING_DATA', '')
    if test_path:
        return os.path.join(test_path, 'sct_testing_data', *args)
    else:
        # NB: The default path written below is actually determined inside
        #     sct_download_data. But, trying to import the path from the script
        #     causes a circular dependency. So, we duplicate the path here.
        #     This could cause bugs if the default location ever changes.
        # TODO: Consider moving sct_test_path() inside testing/conftest.py,
        #       since it's a testing-specific function. This would mitigate the
        #       circular dependency, since we would no longer be importing
        #       from sct_download_data inside sys.py.
        return sct_dir_local_path('data', 'sct_testing_data', *args)


def check_exe(name):
    """
    Ensure that a program exists and can be executed

    :param name: str: name of program or path to program
    :return: boolean
    """
    _, filename = os.path.split(name)
    # Case 1: Check full filepath directly (which may point to a location not on the PATH)
    if os.path.isfile(name) and os.access(name, os.X_OK):
        return True
    # Case 2: Check filename only via the PATH
    elif shutil.which(filename) and os.access(shutil.which(filename), os.X_OK):
        return True
    else:
        return False


def _version_string():
    install_type, sct_commit, sct_branch, version_sct = _git_info()
    if install_type == "package":
        return version_sct
    return "{install_type}-{sct_branch}-{sct_commit}".format(**locals())


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

    with io.open(os.path.join(__sct_dir__, 'spinalcordtoolbox', 'version.txt'), 'r') as f:
        version_sct = f.read().rstrip()

    return install_type, sct_commit, sct_branch, version_sct


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


def __get_commit(path_to_git_folder=None):
    """
    :return: git commit ID, with trailing '*' if modified
    """
    if path_to_git_folder is None:
        path_to_git_folder = __sct_dir__
    else:
        path_to_git_folder = os.path.abspath(os.path.expanduser(path_to_git_folder))

    p = subprocess.Popen(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=path_to_git_folder)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        commit = output.decode().strip()
    else:
        commit = "?!?"

    p = subprocess.Popen(["git", "status", "--porcelain"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=path_to_git_folder)
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


def __get_git_origin(path_to_git_folder=None):
    """
    :return: git origin url if available
    """
    if path_to_git_folder is None:
        path_to_git_folder = __sct_dir__
    else:
        path_to_git_folder = os.path.abspath(os.path.expanduser(path_to_git_folder))

    p = subprocess.Popen(["git", "remote", "get-url", "origin"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=path_to_git_folder)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        origin = output.decode().strip()
    else:
        origin = "?!?"

    return origin


__sct_dir__ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
__version__ = _version_string()
__data_dir__ = os.path.join(__sct_dir__, 'data')
__bin_dir__ = os.path.join(__sct_dir__, 'bin')
__deepseg_dir__ = os.path.join(__data_dir__, 'deepseg_models')
