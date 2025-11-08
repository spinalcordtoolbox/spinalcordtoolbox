"""
Filesystem related helpers and utilities

Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""
import pathlib
import sys
import io
import os
import shutil
import tempfile
import datetime
import logging
from hashlib import md5
from pathlib import Path
from contextlib import contextmanager

import portalocker

from .sys import printv

logger = logging.getLogger(__name__)


def tmp_create(basename):
    """Create temporary folder and return its path
    """
    prefix = f"sct_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{basename}_"
    tmpdir = tempfile.mkdtemp(prefix=prefix)
    logger.info(f"Creating temporary folder ({tmpdir})")
    return tmpdir


# Modified from https://shallowsky.com/blog/programming/python-tee.html
class Tee:
    def __init__(self, _fd1, _fd2):
        self.fd1 = _fd1
        self.fd2 = _fd2

    def close(self):
        # We can't assume that we own the underlying files exclusively; maybe
        # another part of the code also has a reference to them, in which case
        # it would be rude to actually close them. But, we can give up our
        # references to them, and this should automatically close them if the
        # reference count goes to zero.
        del self.fd1, self.fd2

    def write(self, text):
        self.fd1.write(text)
        self.fd2.write(text)

    def flush(self):
        self.fd1.flush()
        self.fd2.flush()

    def isatty(self):
        # This method is needed to ensure that `printv` correctly applies color formatting when sys.stdout==Tee().
        # Use utils.csi_filter if you want to strip the color codes from only one half of the Tee
        return self.fd1.isatty() or self.fd2.isatty()


def copy_helper(src, dst, verbose=1):
    """Copy src to dst, almost like shutil.copy
    If src and dst are the same files, don't crash.
    """
    if not os.path.isfile(src):
        folder = os.path.dirname(src)
        contents = os.listdir(folder)
        raise ValueError(f"Couldn't find {os.path.basename(src)} in {folder} (contents: {contents})")

    try:
        logger.info(f"cp {src} {dst}")
        shutil.copy(src, dst)
    except Exception as e:
        if sys.hexversion < 0x03000000:
            if isinstance(e, shutil.Error) and "same file" in str(e):
                return
        else:
            if isinstance(e, shutil.SameFileError):
                return
        raise  # Must be another error


def rmtree(folder, verbose=1):
    """Recursively remove folder, almost like shutil.rmtree
    """
    printv("rm -rf %s" % (folder), verbose=verbose, type="code")
    shutil.rmtree(folder, ignore_errors=True)


def extract_fname(fpath):
    """
    Split a full path into a parent folder component, filename stem and extension.

    Note: for .nii.gz the extension is understandably .nii.gz, not .gz
    (``os.path.splitext()`` would want to do the latter, hence the special case).
    """
    parent, filename = os.path.split(fpath)
    if filename.endswith(".nii.gz"):
        stem, ext = filename[:-7], ".nii.gz"
    else:
        stem, ext = os.path.splitext(filename)
    return parent, stem, ext


def get_absolute_path(fname):
    if os.path.isfile(fname) or os.path.isdir(fname):
        return os.path.realpath(fname)
    else:
        printv('\nERROR: ' + fname + ' does not exist. Exit program.\n', 1, 'error')


def check_file_exist(fname, verbose=1):
    if fname[0] == '-':
        # fname should be a warping field that will be inverted, ignore the "-"
        fname_to_test = fname[1:]
    else:
        fname_to_test = fname
    if os.path.isfile(fname_to_test):
        if verbose:
            printv('  OK: ' + fname, verbose, 'normal')
        return True
    else:
        printv('\nERROR: The file ' + fname + ' does not exist. Exit program.\n', 1, 'error')
        return False


class TempFolder(object):
    """This class will create a temporary folder."""

    def __init__(self, basename, verbose=0):
        self.path_tmp = tmp_create(basename)
        self.previous_path = None

    def chdir(self):
        """This method will change the working directory to the temporary folder."""
        self.previous_path = os.getcwd()
        os.chdir(self.path_tmp)

    def chdir_undo(self):
        """This method will return to the previous working directory, the directory
        that was the state before calling the chdir() method."""
        if self.previous_path is not None:
            os.chdir(self.previous_path)

    def get_path(self):
        """Return the temporary folder path."""
        return self.path_tmp

    def copy_from(self, filename):
        """This method will copy a specified file to the temporary folder.

        :param filename: The filename to copy into the folder.
        """
        file_fname = os.path.basename(filename)
        copy(filename, self.path_tmp)
        return os.path.join(self.path_tmp, file_fname)

    def cleanup(self):
        """Remove the created folder and its contents."""
        rmtree(self.path_tmp)


def cache_signature(input_files=[], input_params={}):
    """
    Create a signature to be used for caching purposes

    :param input_files: paths of input files (that can influence output)
    :param input_params: input parameters (that can influence output)

    Notes:

    - Use this with cache_valid (to validate caching assumptions)
      and cache_save (to store them)

    - Using this system, the outputs are not checked, only the
      inputs and parameters (to regenerate the outputs in case
      these change).
      If the outputs have been modified directly, then they may
      be reused.
      To prevent that, the next step would be to record the outputs
      signature in the cache file, so as to also verify them prior
      to taking a shortcut.

    """
    import hashlib
    h = hashlib.md5()
    for path in input_files:
        with io.open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
    for k, v in sorted(input_params.items()):
        h.update(str(type(k)).encode('utf-8'))
        h.update(str(k).encode('utf-8'))
        h.update(str(type(v)).encode('utf-8'))
        h.update(str(v).encode('utf-8'))

    return "# Cache file generated by SCT\nDEPENDENCIES_SIG={}\n".format(h.hexdigest()).encode()


def cache_valid(cachefile, sig_expected):
    """
    Verify that the cachefile contains the right signature
    """
    if not os.path.exists(cachefile):
        return False
    with io.open(cachefile, "rb") as f:
        sig_measured = f.read()
    return sig_measured == sig_expected


def cache_save(cachefile, sig):
    """
    Save signature to cachefile

    :param cachefile: path to cache file to be saved
    :param sig: cache signature created with cache_signature()
    """
    with io.open(cachefile, "wb") as f:
        f.write(sig)


def mv(src, dst, verbose=1):
    """Move a file from src to dst (adding a logging message)."""
    printv("mv %s %s" % (src, dst), verbose=verbose, type="code")
    # NB: We specify `shutil.copyfile` to override the default of `shutil.copy2`.
    #     (`copy2` copies file metadata, but doing so fails with a PermissionError on WSL installations where the
    #     src/dest are on different devices. So, we use `copyfile` instead, which doesn't preserve file metadata.)
    #     Fixes https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3832.
    shutil.move(src, dst, copy_function=shutil.copyfile)


def copy(src, dst, verbose=1):
    """Copy src to dst, almost like shutil.copy
    If src and dst are the same files, don't crash.
    """
    if not os.path.isfile(src):
        folder = os.path.dirname(src)
        contents = os.listdir(folder)
        raise Exception("Couldn't find %s in %s (contents: %s)"
                        % (os.path.basename(src), folder, contents))
    try:
        printv("cp %s %s" % (src, dst), verbose=verbose, type="code")
        shutil.copy(src, dst)
    except Exception as e:
        if sys.hexversion < 0x03000000:
            if isinstance(e, shutil.Error) and "same file" in str(e):
                return
        else:
            if isinstance(e, shutil.SameFileError):
                return
        raise  # Must be another error


def relpath_or_abspath(child_path, parent_path):
    """
    Try to find a relative path between a child path and its parent path. If it doesn't exist,
    then the child path is not within the parent path, so return its abspath instead.
    """
    abspath = Path(child_path).resolve()
    try:
        return abspath.relative_to(parent_path)
    except ValueError:
        return abspath


class Mutex(portalocker.BoundedSemaphore):
    """
    General purpose mutex (mutually exclusive semaphore) with a tweaks to
    make it safer, more consistent, and easier to use within SCT. Namely:

    * Implicitly sanitizes the file used by the Mutex within the file system to ensure
        it won't create orphaned directories (or worse)
    * Its "maximum" is enforced to be 1, as required of a mutex.
    * Can be used as a context manager, ensuring it is released after use.
    * If provided, can log a message when it starts waiting to acquire its lock.

    Use instances of this class to ensure that processes run in parallel will not
    create race conditions when they try to access something another process is currently
    modifying (i.e. when writing to a shared log/QC directory).

    :param key: The key this Mutex will check against. If multiple mutexes share the same key,
        they will share access rights, even if run in different processes on the same machine.
    :param directory: Where the files used to track the mutex will be placed.
    :param timeout: How long (in seconds) the Mutex should wait when trying to acquire its corresponding
        lock before failing.
    :param check_interval: How often (in seconds) the Mutex should try to acquire the lock while it waits.
    :param fail_when_locked: Whether to throw an exception if the Mutex fails to acquire the lock within
        the timeout period.
    :param waiting_msg: Message to display when waiting to acquire the lock. If None, no message will print.
    """
    def __init__(
        self,
        key: str,
        directory: str = tempfile.gettempdir(),
        timeout: float | None = 5,  # 5 seconds
        check_interval: float | None = 0.25,  # Every quarter second
        fail_when_locked: bool | None = True,
        waiting_msg: str = None
    ):
        # Run a (slightly constrained) version of the super-class's constructor
        super().__init__(
            name=key,
            directory=directory,
            timeout=timeout,
            check_interval=check_interval,
            fail_when_locked=fail_when_locked,
            # ALWAYS MUTUALLY EXCLUSIVE
            maximum=1
        )

        # Message to print if the lock is not immediately acquired.
        self.waiting_msg = waiting_msg

        # Hashed version of our name to sanitize it
        self.hashed_name = md5(self.name.encode('utf-8')).hexdigest()

    def __setattr__(self, key, value):
        # Block setting the "maximum" value to anything other than 1
        if key == "maximum" and value != 1:
            raise ValueError("A mutex cannot allow more than 1 process to hold the lock!")
        # Otherwise, proceed normally
        super().__setattr__(key, value)

    def __enter__(self):
        # Notify the user if they are using a non-failing Mutex as a context manager that we
        #   are potentially overriding their intent; see rationale below.
        if not self.fail_when_locked:
            print(logger.warning(
                f"LoggingMutex '{self.name}' being run in 'fail_when_locked=True' mode to ensure "
                f"that code within the Python context will not run without lock acquisition."
            ))
        # Ensure that the context init will fail if we don't acquire the lock;
        #   otherwise the context could proceed while another process holds the
        #   lock, defeating the whole point of a mutex.
        self.acquire(fail_when_locked=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Release ourselves when leaving a context
        self.release()

    def acquire(
        self,
        timeout: float | None = None,
        check_interval: float | None = None,
        fail_when_locked: bool | None = None,
    ) -> portalocker.Lock | None:
        """
        Same as super-class implementation, but logs a message if it wasn't immediately
        able to acquire the lock.
        """
        # If we have a message on miss, do a very short acquisition attempt first
        if self.waiting_msg:
            super().acquire(0, check_interval, fail_when_locked=False)

            # If we have the lock, return here: we already have the lock!
            if self.lock is not None: return self.lock

            # Otherwise, print a message before proceeding to regular lock acquisition.
            printv(self.waiting_msg)

        # Attempt to acquire the lock as normal
        return super().acquire(timeout, check_interval, fail_when_locked)

    def get_filename(self, number: int) -> Path:
        """
        Returns a pointing to a file which uses our hashed name,
        rather than using our name in its raw form. This ensures
        path access and creation is sanitized preventing difficult
        to debug (and potentially destructive) side effects.
        """
        return Path(self.directory) / f"{self.hashed_name}_{number}.lock"
