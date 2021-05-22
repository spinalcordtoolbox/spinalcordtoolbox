import os
import pkg_resources
import importlib

import pytest
import matplotlib
import matplotlib.pyplot as plt

scripts = pkg_resources.get_entry_map('spinalcordtoolbox')['console_scripts'].keys()

scripts_where_no_args_is_valid = [
    'isct_test_ants',          # No args -> tests ants binaries
    'sct_testing',             # No args -> runs full test suite
    'sct_check_dependencies',  # No args -> checks dependencies
    'sct_version'              # No args -> prints version
]

scripts_to_test = [s for s in scripts if s not in scripts_where_no_args_is_valid]

scripts_without_callable_main = [
    'isct_convert_binary_to_trilinear',  # Uses 'getopt.getopt(sys.argv[1:])' instead of argparse
    'isct_minc2volume-viewer',           # Does parsing outside of main()
]

scripts_with_callable_main = [s for s in scripts_to_test if s not in scripts_without_callable_main]


@pytest.mark.parametrize("script", scripts_with_callable_main)
def test_scripts_with_no_args_as_main_func(capsys, script):
    """Test that [SCRIPTS_CALLABLE_WITH_MAIN] all return error code 2 and
    show usage descriptions when called with no arguments."""
    mod = importlib.import_module(f"spinalcordtoolbox.scripts.{script}")
    with pytest.raises(SystemExit) as system_err:
        mod.main(argv=[])
    captured = capsys.readouterr()

    assert system_err.value.code is 2
    assert 'usage' in captured.err.lower()


@pytest.mark.script_launch_mode('subprocess')
@pytest.mark.parametrize("script", scripts_without_callable_main)
def test_scripts_with_no_args_as_subprocess(script, script_runner):
    """Test that [SCRIPTS_NOT_CALLABLE_WITH_MAIN] all return error code 2 and
    show usage descriptions when called with no arguments."""
    ret = script_runner.run(script)
    assert ret.returncode is 2
    assert 'usage' in ret.stdout.lower() or 'usage' in ret.stderr.lower()


@pytest.mark.skipif('DISPLAY' in os.environ, reason="Run on headless systems only")
def test_headless_environment_assumptions_for_matplotlib():
    """Verify certain assumptions about how Matplotlib's backends work on headless systems."""
    # Assumption: MPLBACKEND is set to 'Agg', a non-GUI backend (set in launcher when $DISPLAY is not set)
    assert os.environ.get('MPLBACKEND') == 'Agg'

    # Assumption: Matplotlib backend is also set to 'Agg' due to $MPLBACKEND environment variable
    assert matplotlib.get_backend() == 'agg'

    # The two lines below test another property of non-interactive backends.
    # Context: https://github.com/matplotlib/matplotlib/issues/20281#issuecomment-846057011
    fig, ax = plt.subplots()
    assert getattr(fig.canvas, 'required_interactive_framework', False) is None
