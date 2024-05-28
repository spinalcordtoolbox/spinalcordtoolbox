# pytest unit tests for spinalcordtoolbox.utils

import os
import sys
import pytest
from stat import S_IEXEC

from spinalcordtoolbox.utils import shell


def test_parse_num_list_inv():
    assert shell.parse_num_list_inv([1, 2, 3, 5, 6, 9]) == '1:3;5:6;9'
    assert shell.parse_num_list_inv([3, 2, 1, 5]) == '1:3;5'
    assert shell.parse_num_list_inv([]) == ''


def test_sct_argument_parser(capsys):
    """Test extra argparse functionality added by SCTArgumentParser subclass."""
    # Check that new defaults are set properly
    parser = shell.SCTArgumentParser(description="A test argument parser.")
    assert parser.prog == os.path.basename(sys.argv[0]).rstrip(".py")
    assert parser.formatter_class == shell.SmartFormatter
    assert parser.add_help is False

    # Check that error is thrown when required argument isn't passed
    parser.add_argument('-r', '--required', required=True)
    parser.add_argument('-h', "--help", help="show this message and exit", action="help")
    with pytest.raises(SystemExit) as e:
        parser.parse_args()
    assert e.value.code == 2

    # Check help message is still output when above error is thrown
    captured = capsys.readouterr()
    assert "usage:" in captured.err

    # Ensure no error is thrown when help is explicitly called
    with pytest.raises(SystemExit) as e:
        parser.parse_args(['-h'])
    assert e.value.code == 0


@pytest.fixture()
def temporary_viewers(supported_viewers=shell.SUPPORTED_VIEWERS):
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
    syntax_strings = shell.display_viewer_syntax(
        files=["test_img.nii.gz", "test_img_2.nii.gz", "test_seg.nii.gz", "test_img_3.nii.gz"],
        im_types=['anat', 'anat', 'seg', 'anat'],
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
                                "-o test_img_2.nii.gz test_img_3.nii.gz "
                                "-s test_seg.nii.gz")
