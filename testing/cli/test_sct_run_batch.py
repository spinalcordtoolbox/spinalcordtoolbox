import glob
import os
import sys
import json

import pytest
from textwrap import dedent

from spinalcordtoolbox import __sct_dir__
from spinalcordtoolbox.utils.sys import sct_test_path
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
        script.flush()
    return path_out


def test_config_with_args_warning(tmp_path):
    data = str(tmp_path / 'data')
    out = str(tmp_path / 'out')
    config_path = str(tmp_path / 'config.json')

    with open(config_path, 'w') as config:
        cnf_text = {"jobs": 1, "path_data": data, "path_output": out}
        json.dump(cnf_text, config)

    with pytest.warns(UserWarning, match=r'-config.*discouraged'):
        # I'm not sure how to check that argparse is printing the right error here, but I trust
        with pytest.raises(FileNotFoundError):
            sct_run_batch.main(['-c', config_path, '-include', 'something', '-script', 'script'])


def test_config_extra_value_warning(tmp_path):
    data = str(tmp_path / 'data')
    out = str(tmp_path / 'out')
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
        # I'm not sure how to check that argparse is printing the right error here, but I trust
        with pytest.raises(FileNotFoundError):
            sct_run_batch.main(['-c', config_path, '-script', 'script'])


def test_only_one_include(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    out = tmp_path / 'out'
    with pytest.raises(AssertionError, match='Only one'):
        sct_run_batch.main(['-include', 'arg', '-include-list', 'arg2',
                            '-path-data', str(data), '-path-out', str(out), '-script', str(out)])


def test_non_executable_task(tmp_path, dummy_script):
    data_path = sct_test_path()
    out = tmp_path / 'out'

    script_text = """
    #!/bin/bash
    echo $SUBJECT
    """
    with open(tmp_path / "dummy_nonexectuable_script.sh", 'w') as script:
        script.write(dedent(script_text)[1:])  # indexing removes beginning newline
        script.flush()

    # The assertion below is meant to ensure that the script is not executable at first,
    # so that we can test whether `sct_run_batch` properly changes the permissions. However,
    # checking against os.X_OK isn't compatible with Windows. So, I've commented it out:

    # assert not os.access(script.name, os.X_OK), "Script already executable"

    # More broadly speaking, however: Don't the other tests also accomplish this same functionality?
    # i.e. Wouldn't _all_ dummy scripts we create start out as non-executable? I suspect that this test
    # could be combined with the other tests and there wouldn't be a meaningful loss of coverage.

    sct_run_batch.main(['-include', '^t.*',
                        '-subject-prefix', '',
                        '-path-data', data_path, '-path-out', str(out),
                        '-script', script.name,
                        '-continue-on-error', 0])


def test_no_sessions(tmp_path, dummy_script):
    # Test that sessions ('ses') can be separated so that sct_run_batch can process each session folder separately.
    data = tmp_path / 'data'
    data.mkdir()
    out = tmp_path / 'out'
    # Create dummy BIDS directory with sessions
    os.makedirs(os.path.join(data, 'sub-01', 'anat'))
    os.makedirs(os.path.join(data, 'sub-02', 'anat'))
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
        os.makedirs(os.path.join(data, f'sub-{sub}', f'ses-{ses}'))
    sct_run_batch.main(['-path-data', str(data), '-path-out', str(out), '-script', dummy_script])
    for sub, ses in sub_ses_pairs:
        file_log = glob.glob(os.path.join(out, 'log', f'*sub-{sub}_ses-{ses}.log'))[0]
        assert os.path.join(f'sub-{sub}', f'ses-{ses}') in open(file_log, "r").read()
