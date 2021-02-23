
import pytest
import logging

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.scripts import sct_resample

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
@pytest.mark.parametrize("path_in,type_arg,dim,expected_dim", [
    # FIXME: The first test fails (and was also failing in 'sct_testing') -- Is it correct to expect '2.2'?
    ('dmri/dmri.nii.gz', '-f', '0.5x0.5x1', (20, 21, 5, 7, 1.6826923, 1.6826923, 17.5, 2.2)),
    ('t2/t2.nii.gz', '-mm', '0.97x1.14x1.2', (62, 48, 43, 1, 0.96774191, 1.1458334, 1.2093023, 1)),
    ('t2/t2.nii.gz', '-vox', '120x110x26', (120, 110, 26, 1, 0.5, 0.5, 2.0, 1)),
])
def test_sct_resample_output_has_expected_dimensions(path_in, type_arg, dim, expected_dim):
    """Run the CLI script and verify output file exists."""
    path_out = 'resampled.nii.gz'
    sct_resample.main(argv=['-i', path_in, '-o', path_out, type_arg, dim, '-v', '1'])
    actual_dim = Image(path_out).dim

    for actual_val, expected_val in zip(actual_dim, expected_dim):
        assert actual_val == pytest.approx(expected_val)
