# pytest unit tests for all cli scripts

import os
import sys
import pytest
from importlib.metadata import entry_points
import subprocess
import time

scripts = [cs.name for cs in entry_points()['console_scripts'] if cs.value.startswith("spinalcordtoolbox")]

scripts_where_no_args_is_valid = [
    'isct_test_ants',          # No args -> tests ants binaries
    'sct_check_dependencies',  # No args -> checks dependencies
    'sct_version',             # No args -> prints version
    'sct_testing'              # No args -> runs pytest in $SCT_DIR
]

scripts_to_test = [s for s in scripts if s not in scripts_where_no_args_is_valid]


@pytest.mark.parametrize("script", scripts_to_test)
def test_calling_scripts_with_no_args_shows_usage(capsys, script):
    """
    Test that SCT's scripts all return error code 2 and show usage descriptions when called with no arguments.
    Also, ensure that calling the help takes under 2.0 seconds per script.
    """
    start_time = time.time()
    completed_process = subprocess.run([script], capture_output=True)
    duration = time.time() - start_time
    assert completed_process.returncode == 2
    assert b'usage' in completed_process.stderr
    # NB: macOS GitHub Actions runners are inconsistent in their processing speed. Sometimes our requirement is met,
    #     and other times scripts take significantly longer. It's possible to just "retry" the GHA run, but to save
    #     development headache, we just skip this check if we're on a macOS GitHub Actions runner.
    if not (sys.platform.startswith("darwin") and "CI" in os.environ):
        assert duration < 2.0, f"Expected '{script} -h' to execute in under 2.0s, but took {duration}"
