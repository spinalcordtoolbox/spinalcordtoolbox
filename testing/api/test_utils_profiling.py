# pytest unit tests for spinalcordtoolbox.utils.profiling

import atexit
import logging
import time

from collections import namedtuple
from typing import Callable

import pytest

from spinalcordtoolbox.utils import profiling, shell


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


def test_timeit_by_cli(false_atexit, caplog):
    """Confirm our profiling flags enable profiling correctly"""
    # Capture all log input explicitly, so that we can test that the total runtime was run correctly
    caplog.set_level(logging.INFO)

    # Initiate a dummy argument parser
    parser = shell.SCTArgumentParser(description="Time-based profiling test parser.")

    # Add our common argument
    parser.add_common_args()

    # Run an "analysis" with full-program time profiling
    parser.parse_args(['-timeit'])

    # Confirm the timer initialized correctly
    assert profiling.PROFILING_TIMER is not None

    # Wait 1 second to give us some "time" to actually measure
    sleep_time = 1
    time.sleep(sleep_time)

    false_atexit()

    # At this time, the last log should be the reported time; confirm this is correct
    most_recent_log = caplog.records[-1]
    assert "PROFILER:" in most_recent_log.message

    # Confirm the reported runtime is ~1 second
    prog_runtime = float(most_recent_log.message.split("; ")[-1].split(' ')[0])
    assert prog_runtime == pytest.approx(sleep_time, 0.05)
