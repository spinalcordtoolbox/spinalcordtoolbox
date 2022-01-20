#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for utils

import pytest

from spinalcordtoolbox import utils


def test_strip_py():
    """
    Test that utils.strip_py() correctly removes '.py' suffix if present.
    """
    # The happy path
    assert utils.strip_py("basename.py") == "basename"
    assert utils.strip_py("basename") == "basename"

    # A short string
    assert utils.strip_py("a") == "a"

    # Permutations and prefixes should not be removed
    assert utils.strip_py("ppp.yyy") == "ppp.yyy"
    assert utils.strip_py("no_dot_py") == "no_dot_py"
    assert utils.strip_py(".py_prefix") == ".py_prefix"


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
