#!/usr/bin/env python
# -*- coding: utf-8
# System related utilities

import io
import sys
import os
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

import tqdm

from .fs import __version__, sct_dir_local_path

logger = logging.getLogger(__name__)

if os.getenv('SENTRY_DSN', None):
    # do no import if Sentry is not set (i.e., if variable SENTRY_DSN is not defined)
    import raven


# TODO: add test
def init_sct(log_level=1, update=False):
    """
    Initialize the sct for typical terminal usage
    :param log_level: int: 0: warning, 1: info, 2: debug.
    :param update: Bool: If True, only update logging log level. Otherwise, set logging + Sentry.
    :return:
    """
    dict_log_levels = {0: 'WARNING', 1: 'INFO', 2: 'DEBUG'}

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

    # Set logging level for logger and increase level for global config (to avoid logging when calling child functions)
    logger.setLevel(getattr(logging, dict_log_levels[log_level]))
    logging.root.setLevel(getattr(logging, dict_log_levels[log_level]))

    if not update:
        # Initialize logging
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

    if sys.platform.startswith("linux"):
        return "binaries_linux"
    else:
        return "binaries_osx"


def list2cmdline(lst):
    return " ".join(shlex.quote(x) for x in lst)


def run_proc(cmd, verbose=1, raise_exception=True, cwd=None, env=None, is_sct_binary=False):
    if cwd is None:
        cwd = os.getcwd()

    if env is None:
        env = os.environ

    if is_sct_binary:
        name = cmd[0] if isinstance(cmd, list) else cmd.split(" ", 1)[0]
        path = None
        binaries_location_default = sct_dir_local_path("bin")
        for directory in (
            sct_dir_local_path("bin"),
        ):
            candidate = os.path.join(directory, name)
            if os.path.exists(candidate):
                path = candidate
        if path is None:
            run_proc(["sct_download_data", "-d", _which_sct_binaries(), "-o", binaries_location_default])
            path = os.path.join(binaries_location_default, name)

        if isinstance(cmd, list):
            cmd[0] = path
        elif isinstance(cmd, str):
            rem = cmd.split(" ", 1)[1:]
            cmd = path if len(rem) == 0 else "{} {}".format(path, rem[0])

    if isinstance(cmd, str):
        cmdline = cmd
    else:
        cmdline = list2cmdline(cmd)

    logger.debug(f"{cmdline} # in {cwd}")

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
