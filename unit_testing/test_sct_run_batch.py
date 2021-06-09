import glob
import os
import sys
import pytest
from tempfile import NamedTemporaryFile, TemporaryDirectory
from textwrap import dedent

from spinalcordtoolbox import __sct_dir__
from spinalcordtoolbox.utils.sys import sct_test_path
sys.path.append(os.path.join(__sct_dir__, 'scripts'))

from spinalcordtoolbox.scripts import sct_run_batch


def write_dummy_script(script):
    """
    Dummy script that displays subject
    :param script: NamedTemporaryFile object. Script to write
    :return:
    """
    script_text = """
    #!/bin/bash
    SUBJECT=$1
    echo $SUBJECT
    """
    # indexing removes beginning newline
    script.write(dedent(script_text)[1:])
    script.flush()


def test_config_with_args_warning():
    # This is formatted strangely because pep8 says so.
    with \
        NamedTemporaryFile('w', suffix='.json') as config,\
            TemporaryDirectory() as data,\
            TemporaryDirectory() as out:

        cnf_text = '{{"jobs": 1, "path_data": "{}", "path_output": "{}"}}'.format(data, out)
        config.write(cnf_text)
        config.flush()

        with pytest.warns(UserWarning, match=r'-config.*discouraged'):
            # I'm not sure how to check that argparse is printing the right error here, but I trust
            with pytest.raises(FileNotFoundError):
                sct_run_batch.main(['-c', config.name, '-include', 'something', '-script', 'script'])


def test_config_extra_value_warning():
    # This is formatted strangely because pep8 says so.
    with \
        NamedTemporaryFile('w', suffix='.yml') as config,\
            TemporaryDirectory() as data,\
            TemporaryDirectory() as out:

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
                sct_run_batch.main(['-c', config.name, '-script', 'script'])


def test_only_one_include():
    # This is formatted strangely because pep8 says so.
    with \
        TemporaryDirectory() as data,\
            TemporaryDirectory() as out:
        with pytest.raises(AssertionError, match='Only one'):
            sct_run_batch.main(['-include', 'arg', '-include-list',
                                'arg2', '-path-data', data, '-path-out', out
                                , '-script', out])


def test_non_executable_task():
    data_path = sct_test_path()
    with \
        NamedTemporaryFile('w', suffix='.sh') as script,\
            TemporaryDirectory() as out:

        script_text = """
        #!/bin/bash
        echo $SUBJECT
        """
        script.write(dedent(script_text)[1:]) #indexing removes beginning newline
        script.flush()

        assert not os.access(script.name, os.X_OK), "Script already executable"

        sct_run_batch.main(['-include', '^t.*',
                            '-subject-prefix', '',
                            '-path-data', data_path, '-path-out', out,
                            '-script', script.name,
                            '-continue-on-error', 0])


def test_no_sessions():
    # Test that sessions ('ses') can be separated so that sct_run_batch can process each session folder separately.
    with TemporaryDirectory() as data,\
            TemporaryDirectory() as out,\
            NamedTemporaryFile('w', suffix='.sh') as script:
        # Create dummy BIDS directory with sessions
        os.makedirs(os.path.join(data, 'sub-01', 'anat'))
        os.makedirs(os.path.join(data, 'sub-02', 'anat'))
        write_dummy_script(script)
        sct_run_batch.main(['-path-data', data, '-path-out', out, '-script', script.name])
        file_log = glob.glob(os.path.join(out, 'log', '*sub-01.log'))[0]
        assert 'sub-01' in open(file_log, "r").read()


def test_separate_sessions():
    # Test that sessions ('ses') can be separated so that sct_run_batch can process each session folder separately.
    with TemporaryDirectory() as data,\
            TemporaryDirectory() as out,\
            NamedTemporaryFile('w', suffix='.sh') as script:
        # Create dummy BIDS directory with sessions
        sub_ses_pairs = [('01', '01'), ('01', '02'), ('01', '03'), ('02', '01'), ('02', '02')]
        for sub, ses in sub_ses_pairs:
            os.makedirs(os.path.join(data, f'sub-{sub}', f'ses-{ses}'))
        write_dummy_script(script)
        sct_run_batch.main(['-path-data', data, '-path-out', out, '-script', script.name])
        for sub, ses in sub_ses_pairs:
            file_log = glob.glob(os.path.join(out, 'log', f'*sub-{sub}_ses-{ses}.log'))[0]
            assert f'sub-{sub}/ses-{ses}' in open(file_log, "r").read()
