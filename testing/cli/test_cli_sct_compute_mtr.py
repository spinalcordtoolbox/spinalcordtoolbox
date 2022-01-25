import pytest
import logging
import numpy
import nibabel as nib

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.scripts import sct_compute_mtr

logger = logging.getLogger(__name__)


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


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_compute_mtr_with_dummy_highvalue_int16_data(tmp_path):
    """
    Test that high-valued int16 data doesn't result in clipping. See also:
    https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3638#discussion_r791972013
    """

    mt0_path = str(tmp_path / 'mt0_int16.nii.gz')
    mt1_path = str(tmp_path / 'mt1_int16.nii.gz')
    mtr_output_path = str(tmp_path / 'mtr_output.nii.gz')
    mtr_truth_path = str(tmp_path / 'mtr_dummy_truth.nii.gz')

    # Generate dummy files of inputs and output comparison
    nx, ny, nz = 9, 9, 9  # image dimension # center location

    data = numpy.zeros((nx, ny, nz), dtype=numpy.int16)
    data[4, 4, 4] = 1000
    affine = numpy.eye(4)
    mt0 = nib.nifti1.Nifti1Image(data, affine)
    nib.save(mt0, mt0_path)

    data = numpy.zeros((nx, ny, nz), dtype=numpy.int16)
    data[4, 4, 4] = 600
    affine = numpy.eye(4)
    mt1 = nib.nifti1.Nifti1Image(data, affine)
    nib.save(mt1, mt1_path)

    data = numpy.zeros((nx, ny, nz), dtype=numpy.float32)
    data[4, 4, 4] = 40
    affine = numpy.eye(4)
    mtr_truth = nib.nifti1.Nifti1Image(data, affine)
    nib.save(mtr_truth, mtr_truth_path)

    # Compute the MTR
    sct_compute_mtr.main(argv=['-mt0', mt0_path, '-mt1', mt1_path, '-o', mtr_output_path])

    # Compare the dummy output with the mtr file newly generated
    truth_mtr = Image(mtr_truth_path)
    output_mtr = Image(mtr_output_path)
    # nan_to_num is needed because MTR calculation replaces 0 with np.nan
    assert numpy.array_equal(truth_mtr.data, numpy.nan_to_num(output_mtr.data))
