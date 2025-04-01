import atexit
import cProfile
import io
import logging
import pstats
import time

from pathlib import Path


PROFILING_TIMER = None
TIME_PROFILER = None


# == Time-based Profiling == #
class Timer:
    """
    Wrapper class which manages our total runtime timer
    """
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


class TimeProfilingManager:

    default_output = Path('./time_profiler_results.txt')

    def __init__(self, out_file=None):
        # Initialize and enable the profiler
        self._profiler = cProfile.Profile()
        self._profiler.enable()
        # Track the output file, if it was provided
        if out_file is not None:
            self._output_file = out_file
        else:
            self._output_file = self.default_output
        # Ensure the profiler completes its runtime at program exit
        atexit.register(self._finish_profiling)

    def __del__(self):
        # If the profiler is deleted explicitly, just clean up without reporting anything
        atexit.unregister(self._finish_profiling)
        self._profiler.disable()

    def _finish_profiling(self):
        # Finish profiling
        self._profiler.disable()

        # Grab the stats, and organize them from most to least time-consuming (cumulatively)
        io_stream = io.StringIO()
        profiling_stats = pstats.Stats(self._profiler, stream=io_stream)
        profiling_stats = profiling_stats.sort_stats(pstats.SortKey.CUMULATIVE)
        profiling_stats.print_stats()

        # Save the results to the desired file
        with open(self._output_file, 'w') as out_stream:
            out_stream.write(io_stream.getvalue())


def begin_profiling_time(out_path: Path = None):
    # Fetch the PROFILING_TIMER from the global space
    global TIME_PROFILER
    # If it hasn't already been set, initiate and set up the profiler
    if TIME_PROFILER is None:
        # Initialize the profiler
        TIME_PROFILER = TimeProfilingManager(out_path)
    # Otherwise, skip and warn the user
    else:
        logging.warning(
            "Tried to start the time profiler twice; you should leave generally leave this to be handled by the CLI, "
            "rather than calling 'begin_profiling_time' directly!"
        )
