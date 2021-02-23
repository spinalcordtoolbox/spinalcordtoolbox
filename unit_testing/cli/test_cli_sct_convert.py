import os

import pytest
import logging

from spinalcordtoolbox.scripts import sct_convert

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_convert_output_file_exists():
    """Run the CLI script and verify output file exists."""
    path_out = 't2.nii'
    sct_convert.main(argv=['-i', 't2/t2.nii.gz', '-o', path_out])
    os.path.exists(path_out)
