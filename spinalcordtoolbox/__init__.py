#!/usr/bin/env python
import os
import multiprocessing

from .utils import __version__, __sct_dir__, __data_dir__, __deepseg_dir__


def configure_sct_env_variables():
    """
    Set environment variables that should be active in the scope of the running Python process.

    Note: This change will be active whenever `spinalcordtoolbox` is imported. So, try to only make light
    changes here, as to not significantly impact any downstream packages built on `spinalcordtoolbox`.
    """
    # This is used by ANTs tools (e.g. isct_antsSliceRegularization) to control multiprocessing
    if "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS" not in os.environ:
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(multiprocessing.cpu_count())

    # DISPLAY is used by the X Window System (in UNIX): https://docstore.mik.ua/orelly/unix3/upt/ch35_08.htm
    # We use this environment variable to detect whether or not we're on a headless system.
    # NB: Sometimes 'DISPLAY' can be unset for SSH sessions that aren't headless. If any users encounter issues
    # displaying plots, direct them to https://unix.stackexchange.com/q/138936
    if "DISPLAY" not in os.environ:
        # If we're on a headless system, set matplotlib's backend to 'Agg', which is a non-interactive backend.
        # This will prevent interactive plots from being shown, which keeps headless systems from hanging indefinitely.
        os.environ["MPLBACKEND"] = "Agg"
        # NB: We used to set MPLBACKEND in the user's RC file via `install_sct`. Unfortunately, this means that
        # 'export MPLBACKEND=Agg' might be hanging around in users' RC files, even if they're not on a headless system.


configure_sct_env_variables()
