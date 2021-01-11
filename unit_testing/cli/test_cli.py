from spinalcordtoolbox.utils import __sct_dir__
import os
import pytest

scripts = []
for line in open(os.path.join(__sct_dir__, 'scripts.txt')):
    li = line.strip()
    if not li.startswith("#"):
        scripts.append(li)

# Exclude scripts where no arguments is valid usage
no_args_scripts = [s for s in scripts if s not in [
    'isct_test_ants',          # No args -> tests ants binaries
    'sct_testing',             # No args -> runs full test suite
    'sct_check_dependencies',  # No args -> checks dependencies
    'sct_version'              # No args -> prints version
]]


@pytest.mark.script_launch_mode('subprocess')
@pytest.mark.parametrize("script", no_args_scripts)
def test_argparse_no_arguments(script, script_runner):
    """Ensure all scripts return error code and show usage descriptions
    when called with no arguments."""
    ret = script_runner.run(script)
    assert ret.returncode is 2
    assert 'usage' in ret.stdout.lower() or 'usage' in ret.stderr.lower()
