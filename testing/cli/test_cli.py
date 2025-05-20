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

# Scripts which have a deprecation warning in front of them, extending their no-arg runtime by 3 seconds
deprecated_scripts = [
    'sct_deepseg_lesion',
    'sct_deepseg_sc',
]

scripts_to_test = [s for s in scripts if s not in scripts_where_no_args_is_valid]


@pytest.mark.parametrize("script", scripts_to_test)
def test_calling_scripts_with_no_args_shows_usage(capsys, script):
    """
    Test that SCT's scripts all return error code 2 and show usage descriptions when called with no arguments.
    Also, ensure that calling the help takes under 2.0 seconds (excluding deprecation warning time) per script.
    """
    start_time = time.time()
    completed_process = subprocess.run([script], capture_output=True)
    duration = time.time() - start_time
    assert completed_process.returncode == 2
    assert b'usage' in completed_process.stderr
    # NB: macOS/Windows GitHub Actions runners are inconsistent in their processing speed. Sometimes our requirement is met,
    #     and other times scripts take significantly longer. It's possible to just "retry" the GHA run, but to save
    #     development headache, we just skip this check if we're on a macOS/Windows GitHub Actions runner.
    # TODO: Revisit import profiling and see if there are new enhancements we can make to get scripts back under 2.0s
    if not ((sys.platform.startswith("darwin") or sys.platform.startswith("win32"))
            and "CI" in os.environ):
        max_duration = 2.0 if script not in deprecated_scripts else 5.0
        assert duration < max_duration, f"Expected '{script} -h' to execute in under {max_duration}s; took {duration}"
