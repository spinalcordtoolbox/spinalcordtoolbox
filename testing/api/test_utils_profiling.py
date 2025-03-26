# pytest unit tests for spinalcordtoolbox.utils.profiling

import atexit
import logging
import time

import pytest

from spinalcordtoolbox.utils import profiling, shell


@pytest.fixture()
def false_atexit(monkeypatch):
    """
    Proxy's a program "ending" in the eyes of atexit, allowing the user to "end" the program at will
    """
    # Define a proxy functions for atexit's registration management, allowing them to be explicitly called
    teardown_fns = []
    register_proxy = lambda fn: teardown_fns.append(fn)
    unregister_proxy = lambda fn: teardown_fns.remove(fn)

    # Monkeypatch it in
    monkeypatch.setattr(atexit, "register", register_proxy)
    monkeypatch.setattr(atexit, "unregister", unregister_proxy)

    # Return a method which will run each function in the teardown_fn list, simulating the program closing
    def _return_fn():
        for i, fn in enumerate(teardown_fns):
            fn()

    return _return_fn


def test_timeit_by_cli(false_atexit, caplog):
    """Confirm our profiling flags enable profiling correctly"""
    # Capture all log input explicitly, so that we can test that the total runtime was run correctrly
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
