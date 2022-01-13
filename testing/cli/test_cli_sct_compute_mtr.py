import pytest
import logging
import numpy

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.scripts import sct_compute_mtr

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_compute_mtr_no_checks(tmp_path):
    """Run the CLI script without checking results."""
    mtr_output_path = str(tmp_path / 'mtr_output.nii.gz')

    sct_compute_mtr.main(argv=['-mt0', 'mt/mt0.nii.gz', '-mt1', 'mt/mt1.nii.gz', '-o', mtr_output_path])

    # Comparate ground truth mtr file to new generated one
    ground_truth_mtr = Image('mt/mtr.nii.gz')
    output_mtr = Image(mtr_output_path)
    assert numpy.linalg.norm(ground_truth_mtr.data - output_mtr.data) <= 0.001


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_compute_mtr_with_int16_image_type(tmp_path):
    """Run the CLI script with int_16 image type"""

    mt0_path = str(tmp_path / 'mt0_int16.nii.gz')
    mt1_path = str(tmp_path / 'mt1_int16.nii.gz')
    mtr_output_path = str(tmp_path / 'mtr_output.nii.gz')

    # Generate int16 test file based on sct_testing_data existing one
    mt0 = Image('mt/mt0_reg_slicereg_goldstandard.nii.gz')
    mt0.save(mt0_path, dtype='int16')
    mt1 = Image('mt/mt1.nii.gz')
    mt1.save(mt1_path, dtype='int16')

    sct_compute_mtr.main(argv=['-mt0', mt0_path, '-mt1', mt1_path, '-o', mtr_output_path])

    # Comparate ground truth mtr file to new generated one
    ground_truth_mtr = Image('mt/mtr.nii.gz')
    output_mtr = Image(mtr_output_path)

    # NB: numpy.isfinite is used to exclude nan/inf elements from comparison
    diff = (ground_truth_mtr.data[numpy.isfinite(ground_truth_mtr.data)] -
            output_mtr.data[numpy.isfinite(output_mtr.data)])

    assert numpy.abs(numpy.average(diff)) <= 0.5
