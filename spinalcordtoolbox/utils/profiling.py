import atexit
import logging
import time

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
        logging.info(f"Total runtime; {time_delta:.3f} seconds.")


def begin_global_timer():
    # Fetch the PROFILING_TIMER from the global space
    global PROFILING_TIMER
    # If it hasn't already been set, start the timer
    if PROFILING_TIMER is None:
        PROFILING_TIMER = Timer()
    # Otherwise, skip and warn the user
    else:
        logging.warning(
            "Tried to start the global profiling timer twice; you should leave generally leave this initialization to "
            "SCT itself, rather than calling 'begin_global_timer' explicitly!"
        )
