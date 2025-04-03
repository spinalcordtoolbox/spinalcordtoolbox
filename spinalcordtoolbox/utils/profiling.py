import atexit
import cProfile
import inspect
import io
import logging
import pstats
import time
import tracemalloc

from argparse import Action
from pathlib import Path


PROFILING_TIMER = None
TIME_PROFILER = None
MEMORY_TRACER = None


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

        # Report that the file was written, and where to
        logging.info(f"Saved time profiling results to '{self._output_file.resolve()}'.")


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


class TimeProfilingAction(Action):
    """
    ArgParse action, which initialized the time-based profiler when the argument is present.

    The user can specify either the output directory or file, if they want the results saved somewhere specific.
    If not, the file will be saved in the current working directory (unless the 'default' is explicitly set otherwise)
    """

    default_fname = Path("time_profiling_results.txt")

    def __call__(self, parser, namespace, values, option_string=None):
        # If the length of the values provided is greater than 1, raise an error
        if values is not None and not isinstance(values, str) and len(values) > 1:
            raise ValueError("Only one output file can be specified for profiler outputs!")

        # If no value as provided, set the output path to be in the current directory
        if values is None:
            out_path = Path('.') / self.default_fname
        else:
            # Otherwise, try to get the path associated with this parameter, and confirm its root exists
            out_path = Path(values)

        # Check to see if the file exists and is a directory
        if out_path.exists() and out_path.is_dir():
            # If so, set the result to be saved to our default file output
            out_path /= self.default_fname

        # Finally, initiate the time-based profiler, designating it's output file to be the user specified one
        begin_profiling_time(out_path)

        setattr(namespace, self.dest, out_path)


# == Memory-based Profiling == #
class MemoryTracingManager:

    default_output = Path('memory_tracer_results.txt')

    def __init__(self, out_file=None):
        # Initialize and enable the profiler
        tracemalloc.start()

        # Track the output file, if it was provided
        if out_file is not None:
            self._output_file = out_file
        else:
            self._output_file = self.default_output

        # Reset the file if it already exists
        if out_file.exists():
            out_file.unlink()
            out_file.touch()

        # Ensure the profiler completes its runtime at program exit
        atexit.register(self._finish_tracing)

    def __del__(self):
        # If the profiler is deleted explicitly, just clean up without reporting anything
        atexit.unregister(self._finish_tracing)
        tracemalloc.stop()

    def snapshot_memory(self, label: str):
        # Snapshot the current memory state
        mem_snapshot = tracemalloc.take_snapshot()
        mem_stats = mem_snapshot.statistics('lineno', cumulative=True)

        # Generate the header for this output
        header = f"=== {label} ===\n"

        # Save the stats to file with this header
        with open(self._output_file, 'a') as fp:
            fp.write(header)
            fp.writelines([str(x) + '\n' for x in mem_stats])
            fp.write('\n')


    def _finish_tracing(self):
        # Get the peak memory consumed during the program, and save it
        _, traced_peak = tracemalloc.get_traced_memory()

        # Finish profiling after sampler (as otherwise the peak is reset to 0)
        tracemalloc.stop()

        # Save the peak results to the output file
        with open(self._output_file, 'a') as fp:
            fp.write(f"PEAK MEMORY USE; {traced_peak / 1024:.3f} KiB")

        # Report that the file was written, and where to
        logging.info(f"Saved memory tracing results to '{self._output_file.resolve()}'.")


def begin_tracing_memory(out_path: Path = None):
    # Fetch the MEMORY_PROFILER from the global space
    global MEMORY_TRACER
    # If it hasn't already been set, initiate and set up the profiler
    if MEMORY_TRACER is None:
        # Initialize the profiler
        MEMORY_TRACER = MemoryTracingManager(out_path)
    # Otherwise, skip and warn the user
    else:
        logging.warning(
            "Tried to start the memory profiler twice; you should leave generally leave this to be handled by the CLI, "
            "rather than calling 'begin_profiling_memory' directly!"
        )


def snapshot_memory():
    # Return early if memory tracing is not active
    global MEMORY_TRACER
    if MEMORY_TRACER is None:
        return

    # Get the calling function
    caller = inspect.stack()[1]
    calling_id = f"{caller.function} ({Path(caller.filename).name}, line {caller.lineno})"

    # Snapshot the current memory state, saving it to file immediately
    # noinspection PyUnresolvedReferences
    MEMORY_TRACER.snapshot_memory(calling_id)


class MemoryTracingAction(Action):
    """
    ArgParse action, which initialized the memory tracer when the argument is present.

    The user can specify either the output directory or file, if they want the results saved somewhere specific.
    If not, the file will be saved in the current working directory (unless the 'const' is explicitly set otherwise)
    """
    def __call__(self, parser, namespace, values, option_string=None):
        # If the length of the values provided is greater than 1, raise an error
        if values is not None and not isinstance(values, str) and len(values) > 1:
            raise ValueError("Only one output file can be specified for profiler outputs!")

        # If no value as provided, set the output path to be in the current directory
        if values is None:
            out_path = MemoryTracingManager.default_output
        else:
            # Otherwise, try to get the path associated with this parameter, and confirm its root exists
            out_path = Path(values)

        # Check to see if the file exists and is a directory
        if out_path.exists() and out_path.is_dir():
            # If so, set the result to be saved to our default file output
            out_path /= MemoryTracingManager.default_output

        # Finally, initiate the memory tracer, designating it's output file to be the user specified one
        begin_tracing_memory(out_path)

        setattr(namespace, self.dest, out_path)
