import pytest
import logging

from spinalcordtoolbox.scripts import sct_maths

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_maths_percent_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_maths.main(argv=['-i', 'mt/mtr.nii.gz', '-percent', '95', '-o', 'test.nii.gz'])


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_maths_add_integer_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_maths.main(argv=['-i', 'mt/mtr.nii.gz', '-add', '1', '-o', 'test.nii.gz'])


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_maths_add_images_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_maths.main(argv=['-i', 'mt/mtr.nii.gz', '-add', 'mt/mtr.nii.gz', 'mt/mtr.nii.gz', '-o', 'test.nii.gz'])