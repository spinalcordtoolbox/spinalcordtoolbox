#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for utils

import os
import pytest
from stat import S_IEXEC

from spinalcordtoolbox import utils


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


@pytest.fixture()
def temporary_viewers(supported_viewers=utils.SUPPORTED_VIEWERS):
    """Set up and teardown viewer files to satisfy check_exe() check within the scope of the test."""
    for viewer in supported_viewers:
        open(viewer, 'a').close()
        # Set "Owner has execute permission" bit to 1 to ensure script is executable
        script_stat = os.stat(viewer)
        os.chmod(viewer, script_stat.st_mode | S_IEXEC)
    yield supported_viewers
    for viewer in supported_viewers:
        os.remove(viewer)


def test_display_viewer_syntax(temporary_viewers):
    """Test that sample input produces the required syntax string output."""
    syntax_strings = utils.display_viewer_syntax(
        files=["test_img.nii.gz", "test_img_2.nii.gz", "test_seg.nii.gz", "test_img_3.nii.gz"],
        colormaps=['gray', 'gray', 'red', 'gray'],
        minmax=['', '0,1', '0.25,0.75', ''],
        opacities=['', '0.7', '1.0', ''],
        mode="test",
        verbose=1,
    )
    for viewer in temporary_viewers:
        assert viewer in syntax_strings.keys()
        cmd_string = syntax_strings[viewer]
        cmd_opts = cmd_string.replace(f"{viewer} ", "")
        if viewer.startswith("fsleyes"):
            assert cmd_opts == ("test_img.nii.gz -cm greyscale "
                                "test_img_2.nii.gz -cm greyscale -dr 0 1 -a 70.0 "
                                "test_seg.nii.gz -cm red -dr 0.25 0.75 -a 100.0 "
                                "test_img_3.nii.gz -cm greyscale &")
        elif viewer.startswith("fslview"):
            assert cmd_opts == ("-m test "
                                "test_img.nii.gz -l Greyscale "
                                "test_img_2.nii.gz -l Greyscale -b 0,1 -t 0.7 "
                                "test_seg.nii.gz -l Red -b 0.25,0.75 -t 1.0 "
                                "test_img_3.nii.gz -l Greyscale &")
        elif viewer.startswith("itk"):
            assert cmd_opts == ("-g test_img.nii.gz "
                                "-s test_seg.nii.gz "
                                "-o test_img_2.nii.gz test_img_3.nii.gz")
