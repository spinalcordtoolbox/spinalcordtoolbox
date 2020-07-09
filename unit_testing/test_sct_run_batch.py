from __future__ import print_function, absolute_import

import os
import sys
import pytest
from tempfile import NamedTemporaryFile, TemporaryDirectory
from textwrap import dedent

from spinalcordtoolbox import __sct_dir__
sys.path.append(os.path.join(__sct_dir__, 'scripts'))

import sct_run_batch


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
                sct_run_batch.main(['-c', config.name, '-include', 'something', '-task', 'task'])


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
                sct_run_batch.main(['-c', config.name, '-task', 'task'])


def test_only_one_include():
    # This is formatted strangely because pep8 says so.
    with \
        TemporaryDirectory() as data,\
            TemporaryDirectory() as out:
        with pytest.raises(AssertionError, match='Only one'):
            sct_run_batch.main(['-include', 'arg', '-include-list',
                                'arg2', '-path-data', data, '-path-out', out
                                , '-task', out])
