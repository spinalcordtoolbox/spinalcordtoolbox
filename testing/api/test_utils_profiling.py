# pytest unit tests for spinalcordtoolbox.utils.profiling

import atexit
import logging
import time
from collections import namedtuple
from typing import Callable

import pytest

from spinalcordtoolbox.utils import profiling
from spinalcordtoolbox.utils.shell import SCTArgumentParser
from spinalcordtoolbox.utils.sys import init_sct

# Named tuple to keep tabs on function arguments for us, in a `atexit` like way
fn_tuple = namedtuple('fn_tuple', ['fn', 'args', 'kwargs'])


@pytest.fixture()
def false_atexit(monkeypatch):
    """
    Proxy's a program "ending" in the eyes of atexit, allowing the user to "end" the program at will
    """
    # Define a proxy functions for atexit's registration management, allowing them to be explicitly called
    teardown_fns: dict[Callable, fn_tuple] = dict()

    def _register_proxy(fn, *args, **kwargs):
        teardown_fns[fn] = fn_tuple(fn, args, kwargs)

    def _unregister_proxy(fn):
        teardown_fns.pop(fn)

    # Monkeypatch it in
    monkeypatch.setattr(atexit, "register", _register_proxy)
    monkeypatch.setattr(atexit, "unregister", _unregister_proxy)

    # Return a method which will run each function in the teardown_fn list, simulating the program closing
    def _return_fn():
        for val in teardown_fns.values():
            val.fn(*val.args, **val.kwargs)

    return _return_fn


@pytest.fixture(autouse=True)
def cleanup_globals():
    # Do nothing initially
    yield

    # Once the fixture is done being "used", clean up the global space
    del profiling.GLOBAL_TIMER
    profiling.GLOBAL_TIMER = None

    del profiling.TIME_PROFILER
    profiling.TIME_PROFILER = None

    del profiling.MEMORY_TRACER
    profiling.MEMORY_TRACER = None


def test_timeit(false_atexit, caplog):
    """Confirm that our timer starts and runs correctly"""
    # Capture all log input explicitly, so that we can test that the total runtime was run correctly
    caplog.set_level(logging.INFO)

    init_sct()

    # Confirm the timer initialized correctly
    assert profiling.GLOBAL_TIMER is not None

    # Wait 1 second to give us some "time" to actually measure
    sleep_time = 1
    time.sleep(sleep_time)

    false_atexit()

    # At this time, the last log should be the reported time; confirm this is correct
    most_recent_log = caplog.records[-1]
    assert "Total runtime;" in most_recent_log.message

    # Confirm the reported runtime is ~1 second
    prog_runtime = float(most_recent_log.message.split("; ")[-1].split(' ')[0])
    allowed_margin = 0.30
    assert prog_runtime >= sleep_time
    assert prog_runtime < sleep_time + allowed_margin


def test_time_profiler(false_atexit, tmp_path):
    # Generate a path we want to save the results too
    out_path = tmp_path / "pytest_time_profiled.txt"

    # For sanity's sake, ensure the file does not already exist yet
    assert not out_path.exists()

    # Initiate time profiling directly
    profiling.begin_profiling_time(out_path)

    # Confirm the time profiler was initialized
    assert profiling.TIME_PROFILER is not None

    # Run some stuff, split across two loops
    no_calls = 100000
    for i in range(no_calls):
        num = i * (i+1)
        logging.info(f"{i}: {num}")

    for i in range(no_calls):
        num = i * (i-1)
        logging.info(f"{i}: {num}")

    total_calls = no_calls*2

    # "End" the program
    false_atexit()

    # Confirm the file associated with the timing profiler exists
    assert out_path.exists()

    # Confirm the contents of the output is sensible
    with open(out_path, 'r') as f:
        # Confirm that the first line reports the total number of calls and runtime
        first_line = f.readline()
        assert "function calls" in first_line
        assert "seconds" in first_line

        # Iterate through the remaining lines for our tests (as they are not guaranteed to be in the same order)
        line_vals = []
        for line in f.readlines():
            # Confirm that the `logger.info` function recorded as having been called `no_call` times
            if "logging" in line and "(info)" in line:
                assert str(total_calls) in line
            # Append the line to a tracked list for a later test
            if line != '\n':
                l_vals = [x for x in line.split(' ') if x != '']
                line_vals.append(l_vals)

    # Skipping the two header lines, confirm that the runtime is sorted in descending order
    for i in range(len(line_vals) - 3):
        cval1 = line_vals[i+2][3]
        cval2 = line_vals[i+3][3]
        assert float(cval1) >= float(cval2)


def test_cli_time_profiling(false_atexit, tmp_path):
    # Generate a path we want to save the results too
    out_path = tmp_path / "pytest_time_profiled.txt"

    # Generate a "dummy" parser with the common command-line arguments
    dummy_parser = SCTArgumentParser(
        description="A dummy parser for PyTest profiling tests"
    )
    dummy_parser.add_common_args()

    # Parse some arguments using it
    dummy_parser.parse_args(["-profile-time", str(out_path)])

    # Confirm the time profiler was initialized
    assert profiling.TIME_PROFILER is not None

    # Confirm that it writes a file on exit, but not before
    assert not out_path.exists()
    false_atexit()
    assert out_path.exists()


def test_memory_tracer(false_atexit, tmp_path):
    # Generate a path we want to save the results too
    out_path = tmp_path / "pytest_memory_traced.txt"

    # For sanity's sake, ensure the file does not already exist yet
    assert not out_path.exists()

    # Initiate memory tracing directly
    profiling.begin_tracing_memory(out_path)

    # Confirm the memory profiler was initialized
    assert profiling.MEMORY_TRACER is not None

    # Generate a gigantic list with a ton of integers (which are each 16 bits, or one byte)
    n_numbers = 1000000
    big_list = list(range(n_numbers))

    # Explicit delete the big list to ensure the reported peak was still measured correctly
    del big_list

    # Calculate the minimum amount of memory this should have required, account for pointers as well;
    #   n ints
    #   n pointers, 1 per int
    #   1 pointer to the list itself
    #   16 bits per element (Convenient perk of python; everything is a Byte!)
    min_n_bits = (2 * n_numbers + 1) * 16

    # "End" the program
    false_atexit()

    # Confirm that profiling has ceased at this point
    import tracemalloc
    assert not tracemalloc.is_tracing()

    # Confirm the output file now exists
    assert out_path.exists()

    # Confirm that the last line in the output file is the peak memory use
    with open(out_path, 'r') as fp:
        last_line = [x for x in fp.readlines()][-1]

    assert "PEAK MEMORY USE" in last_line
    recorded_mem_kib = float(last_line.split('; ')[-1].split(' ')[0])
    assert recorded_mem_kib > (min_n_bits / 1024)


def test_memory_snapshot(false_atexit, tmp_path):
    # Generate a path we want to save the results too
    out_path = tmp_path / "pytest_memory_snapshot.txt"

    # For sanity's sake, ensure the file does not already exist yet
    assert not out_path.exists()

    # Initiate memory tracing directly
    profiling.begin_tracing_memory(out_path)

    # Generate a gigantic list with a ton of integers (which are each 16 bits, or one byte)
    n_numbers = 1000000
    big_list = list(range(n_numbers))  # noqa: F841 (necessary evil to prevent garbage collection messing w/ test)

    # Snapshot the memory, which should update the output
    profiling.snapshot_memory()

    # "End" the program
    false_atexit()

    # Confirm that the memory tracer saved the snapshot correctly
    with open(out_path, 'r') as fp:
        first_line = fp.readline()
        # We don't test for the exact line, as it is prone to being changed
        assert "test_memory_snapshot (test_utils_profiling.py, line " in first_line


def test_cli_memory_tracer(false_atexit, tmp_path):
    # Generate a path we want to save the results too
    out_path = tmp_path / "pytest_memory_traced.txt"

    # Generate a "dummy" parser with the common command-line arguments
    dummy_parser = SCTArgumentParser(
        description="A dummy parser for PyTest profiling tests"
    )
    dummy_parser.add_common_args()

    # Parse some arguments using it
    dummy_parser.parse_args(["-trace-memory", str(out_path)])

    # Confirm the time profiler was initialized
    assert profiling.MEMORY_TRACER is not None

    # Confirm that it writes a file on exit, but not before
    assert not out_path.exists()
    false_atexit()
    assert out_path.exists()
