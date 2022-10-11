import pytest
import pkg_resources
import importlib

scripts = pkg_resources.get_entry_map('spinalcordtoolbox')['console_scripts'].keys()

scripts_where_no_args_is_valid = [
    'isct_test_ants',          # No args -> tests ants binaries
    'sct_check_dependencies',  # No args -> checks dependencies
    'sct_version',             # No args -> prints version
    'sct_testing'              # No args -> runs pytest in $SCT_DIR
]

scripts_to_test = [s for s in scripts if s not in scripts_where_no_args_is_valid]


@pytest.mark.parametrize("script", scripts_to_test)
def test_calling_scripts_with_no_args_shows_usage(capsys, script):
    """Test that SCT's scripts all return error code 2 and
    show usage descriptions when called with no arguments."""
    mod = importlib.import_module(f"spinalcordtoolbox.scripts.{script}")
    with pytest.raises(SystemExit) as system_err:
        mod.main(argv=[])
    captured = capsys.readouterr()

    assert system_err.value.code is 2
    assert 'usage' in captured.err.lower()
