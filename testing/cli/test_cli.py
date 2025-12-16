# pytest unit tests for all cli scripts

import pytest
from importlib.metadata import entry_points
import subprocess

scripts = [cs.name for cs in entry_points().select(group='console_scripts') if cs.value.startswith("spinalcordtoolbox")]

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
    """
    completed_process = subprocess.run([script], capture_output=True)
    assert completed_process.returncode == 2
    assert b'usage' in completed_process.stderr
