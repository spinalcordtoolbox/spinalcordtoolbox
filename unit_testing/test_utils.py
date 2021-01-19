#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for utils

import pytest
import pytest_cases

from spinalcordtoolbox import utils

fixture = pytest_cases.fixture

def parametrize(argnames=None,
                argvalues=None,
                *args, **kwargs):
    """
    Wrap pytest_cases.parametrize (which wraps pytest.mark.parametrize)
    to avoid having to repeat fixture_ref() everywhere.

    See https://smarie.github.io/python-pytest-cases/api_reference/#parametrize
    """
    return pytest_cases.parametrize(argnames,
                                    [(pytest_cases.fixture_ref(v) if hasattr(v,'_pytestfixturefunction') else v)
                                      for v in argvalues] if argvalues else argvalues,
                                    *args,
                                    **kwargs)


def test_parse_num_list_inv():
    assert utils.parse_num_list_inv([1, 2, 3, 5, 6, 9]) == '1:3;5:6;9'
    assert utils.parse_num_list_inv([3, 2, 1, 5]) == '1:3;5'
    assert utils.parse_num_list_inv([]) == ''


def test_sct_argument_parser(capsys):
    """Test extra argparse functionality added by SCTArgumentParser subclass."""
    # Check that new defaults can still be overridden (setting add_help via args AND kwargs)
    parser1 = utils.SCTArgumentParser(None, None, None, None, [], utils.SmartFormatter, '-', None, None, 'error', True)
    assert parser1.add_help is True
    parser2 = utils.SCTArgumentParser(add_help=True)
    assert parser2.add_help is True

    # Check that new defaults are set properly
    parser3 = utils.SCTArgumentParser()
    assert parser3.prog == "test_utils"
    assert parser3.formatter_class == utils.SmartFormatter
    assert parser3.add_help is False

    # Check that error is thrown when required argument isn't passed
    parser3.add_argument('-r', '--required', required=True)
    parser3.add_argument('-h', "--help", help="show this message and exit", action="help")
    with pytest.raises(SystemExit) as e:
        parser3.parse_args()
    assert e.value.code == 2

    # Check help message is still output when above error is thrown
    captured = capsys.readouterr()
    assert "usage: test_utils" in captured.err

    # Ensure no error is thrown when help is explicitly called
    with pytest.raises(SystemExit) as e:
        parser3.parse_args(['-h'])
    assert e.value.code == 0
