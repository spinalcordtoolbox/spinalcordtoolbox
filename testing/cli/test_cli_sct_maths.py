import pytest
import logging

import numpy as np

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils import sct_test_path
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


@pytest.mark.parametrize('dim', ['0', '1', '2'])
def test_sct_maths_symmetrize(dim, tmp_path):
    """Run the CLI script, then verify that symmetrize properly flips and
    averages the image data."""
    path_in = sct_test_path('t2', 't2.nii.gz')
    path_out = str(tmp_path/f't2_sym_{dim}.nii.gz')
    sct_maths.main(argv=['-i', path_in, '-symmetrize', str(dim),
                         '-o', path_out])
    im_in = Image(path_out)
    im_out = Image(path_out)
    assert np.array_equal(im_out.data,
                          (im_in.data + np.flip(im_in.data, axis=int(dim))) / 2.0)
