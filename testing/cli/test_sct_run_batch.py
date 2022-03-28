import glob
import os
import sys
import json

import pytest
from stat import S_IEXEC
from textwrap import dedent

from spinalcordtoolbox import __sct_dir__
sys.path.append(os.path.join(__sct_dir__, 'scripts'))

from spinalcordtoolbox.scripts import sct_run_batch


@pytest.fixture
def dummy_script(tmp_path):
    """Dummy executable script that displays subject."""
    path_out = str(tmp_path / "dummy_script.sh")
    script_text = """
    #!/bin/bash
    SUBJECT=$1
    echo $SUBJECT
    """
    with open(path_out, 'w') as script:
        # indexing removes beginning newline
        script.write(dedent(script_text)[1:])
    # Set "Owner has execute permission" bit to 1 to ensure script is executable
    script_stat = os.stat(path_out)
    os.chmod(path_out, script_stat.st_mode | S_IEXEC)
    # NB: This does nothing on Windows. (https://docs.python.org/3/library/os.html#os.chmod)
    # > Although Windows supports chmod(), you can only set the file’s read-only flag with it
    # > (via the stat.S_IWRITE and stat.S_IREAD constants or a corresponding integer value).
    # > All other bits are ignored.
    return path_out


def test_config_with_args_warning(tmp_path):
    data = str(tmp_path / 'data')
    out = str(tmp_path / 'out')
    config_path = str(tmp_path / 'config.json')

    with open(config_path, 'w') as f:
        config = {"jobs": 1, "path_data": data, "path_output": out}
        json.dump(config, f)

    with pytest.warns(UserWarning, match=r'-config.*discouraged'):
        # I'm not sure how to check that argparse is printing the right error here, but I trust
        with pytest.raises(FileNotFoundError):
            sct_run_batch.main(['-c', config_path, '-include', 'something', '-script', 'script'])


def test_config_extra_value_warning(tmp_path, dummy_script):
    data = tmp_path / 'data'
    data.mkdir()
    out = tmp_path / 'out'
    config_path = str(tmp_path / 'config.yml')

    with open(config_path, 'w') as config:
        cnf_text = """
        jobs: 1
        path_data: {}
        path_output: {}
        unknowable: unknown
        """.format(data, out)
        config.write(dedent(cnf_text))
        config.flush()

    with pytest.warns(UserWarning, match='unknowable'):
        sct_run_batch.main(['-c', config_path, '-script', dummy_script])


def test_only_one_include(tmp_path, dummy_script):
    data = tmp_path / 'data'
    data.mkdir()
    out = tmp_path / 'out'
    with pytest.raises(AssertionError, match='Only one'):
        sct_run_batch.main(['-include', 'arg', '-include-list', 'arg2',
                            '-path-data', str(data), '-path-out', str(out), '-script', dummy_script])


def test_non_executable_task(tmp_path, dummy_script):
    data = tmp_path / 'data'
    data.mkdir()
    out = tmp_path / 'out'

    # Set "Owner has execute permission" bit to 0 to ensure script is non-executable
    script_stat = os.stat(dummy_script)
    os.chmod(dummy_script, script_stat.st_mode & ~S_IEXEC)
    # NB: This does nothing on Windows. (https://docs.python.org/3/library/os.html#os.chmod)
    # > Although Windows supports chmod(), you can only set the file’s read-only flag with it
    # > (via the stat.S_IWRITE and stat.S_IREAD constants or a corresponding integer value).
    # > All other bits are ignored.

    # More broadly speaking, however: Don't the other tests also accomplish this same functionality?
    # i.e. Wouldn't _all_ dummy scripts we create start out as non-executable? I suspect that this test
    # could be combined with the other tests and there wouldn't be a meaningful loss of coverage.

    sct_run_batch.main(['-include', '^t.*',
                        '-subject-prefix', '',
                        '-path-data', str(data), '-path-out', str(out),
                        '-script', dummy_script,
                        '-continue-on-error', 0])


def test_no_sessions(tmp_path, dummy_script):
    # Test that sessions ('ses') can be separated so that sct_run_batch can process each session folder separately.
    data = tmp_path / 'data'
    data.mkdir()
    out = tmp_path / 'out'
    # Create dummy BIDS directory with sessions
    for sub in ['01', '02']:
        (data / f'sub-{sub}' / 'anat').mkdir(parents=True)
    sct_run_batch.main(['-path-data', str(data), '-path-out', str(out), '-script', dummy_script])
    file_log = glob.glob(os.path.join(out, 'log', '*sub-01.log'))[0]
    assert 'sub-01' in open(file_log, "r").read()


def test_separate_sessions(tmp_path, dummy_script):
    # Test that sessions ('ses') can be separated so that sct_run_batch can process each session folder separately.
    data = tmp_path / 'data'
    data.mkdir()
    out = tmp_path / 'out'
    # Create dummy BIDS directory with sessions
    sub_ses_pairs = [('01', '01'), ('01', '02'), ('01', '03'), ('02', '01'), ('02', '02')]
    for sub, ses in sub_ses_pairs:
        (data / f'sub-{sub}' / f'ses-{ses}').mkdir(parents=True)
    sct_run_batch.main(['-path-data', str(data), '-path-out', str(out), '-script', dummy_script])
    for sub, ses in sub_ses_pairs:
        file_log = glob.glob(os.path.join(out, 'log', f'*sub-{sub}_ses-{ses}.log'))[0]
        assert os.path.join(f'sub-{sub}', f'ses-{ses}') in open(file_log, "r").read()
