import atexit
import logging
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
class StartProfilingTimer(Action):
    def __init__(self, option_strings, dest=SUPPRESS, default=SUPPRESS, help=None):
        super(StartProfilingTimer, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help
        )

    """argparse Action, which will start a timing tracker for us"""
    def __call__(self, *args, **kwargs):
        begin_global_time()


def begin_global_time():
    # Fetch the PROFILING_TIMER from the global space
    global PROFILING_TIMER
    # If it hasn't already been set, start the timer
    if PROFILING_TIMER is None:
        PROFILING_TIMER = Timer()
    # Otherwise, skip and warn the user
    else:
        logging.warning("Tried to start a profiling timer more than once! Perhaps you forgot you had set the "
                        "'SCT_TIMER' variable?")
