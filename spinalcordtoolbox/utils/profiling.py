import atexit
import logging
import os
import time
from argparse import Action, SUPPRESS

PROFILING_TIMER = None


# Wrapper class which manages our profiling timer
class Timer:
    def __init__(self):
        self._t0 = time.time()
        atexit.register(self.stop)

    def __del__(self):
        """
        Prevent early deletion of Timers from breaking atexit
        """
        atexit.unregister(self.stop)

    def stop(self):
        time_delta = time.time() - self._t0
        logging.warning(f"PROFILER: Elapsed time: {time_delta:.3f} seconds.")


# ArgParse Action class which starts a timer when requested
class StartGlobalTimer(Action):
    def __init__(self, option_strings, dest=SUPPRESS, default=SUPPRESS, help=None):
        super(StartGlobalTimer, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help
        )

    """argparse Action, which will start a timing tracker for us"""
    def __call__(self, *args, **kwargs):
        begin_global_timer()


def begin_global_timer():
    # Fetch the PROFILING_TIMER from the global space
    global PROFILING_TIMER
    # If it hasn't already been set, start the timer
    if PROFILING_TIMER is None:
        PROFILING_TIMER = Timer()
    # Otherwise, skip and warn the user
    else:
        # If this was because the OS-level environmental variable was set
        if "SCT_TIMER" in os.environ.keys():
            logging.warning(
                "Both the '-timeit' flag and the 'SCT_TIMER' environmental variable were used, resulting in the timer "
                "being started twice; you should only do one or the other!"
            )
        else:
            logging.warning(
                "Tried to start the global profiling timer twice; you should generally leave this initialization to the "
                "CLI, rather than calling 'begin_global_timer yourself!"
            )
