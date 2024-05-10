# pytest unit tests for sct_convert

import os

import pytest
import logging

from spinalcordtoolbox.scripts import sct_convert
from spinalcordtoolbox.utils.sys import sct_test_path

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
def test_sct_convert_output_file_exists():
    """Run the CLI script and verify output file exists."""
    path_out = 't2.nii'
    sct_convert.main(argv=['-i', sct_test_path('t2', 't2.nii.gz'), '-o', path_out])
    assert os.path.exists(path_out)
