
import pytest
import logging

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.scripts import sct_resample

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
@pytest.mark.parametrize("path_in,type_arg,dim,expected_dim", [
    ('fmri/fmri.nii.gz', '-mm', '1x1x3', (65, 65, 34, 30, 1.0, 1.0, 3.0, 1.13)),                     # 4D, mm
    ('dmri/dmri.nii.gz', '-f', '0.5x0.5x1', (20, 21, 5, 7, 1.6826923, 1.6826923, 17.5, 2.2)),        # 4D, factor
    ('t2/t2.nii.gz', '-mm', '0.97x1.14x1.2', (62, 48, 43, 1, 0.97, 1.14, 1.2, 1)),                   # 3D, mm
    ('t2/t2.nii.gz', '-vox', '120x110x26', (120, 110, 26, 1, 0.5, 0.5, 2.0, 1)),                     # 3D, vox
])
def test_sct_resample_output_has_expected_dimensions(path_in, type_arg, dim, expected_dim, tmp_path):
    """Run the CLI script and verify output file exists."""
    path_out = str(tmp_path / 'resampled.nii.gz')
    sct_resample.main(argv=['-i', path_in, '-o', path_out, type_arg, dim, '-v', '1'])
    actual_dim = Image(path_out).dim

    if type_arg == '-mm':
        # Don't expect the actual mm values to be precise, since `-mm` gets
        # converted to # of voxels (via rounding), so it's only an approx.
        abs_tol = 0.05
    else:
        abs_tol = 0.0000001  # For the others, the results should be precise

    for actual_val, expected_val in zip(actual_dim, expected_dim):
        assert actual_val == pytest.approx(expected_val, abs=abs_tol)


def test_sct_resample_no_f_mm_or_vox_arguments(tmp_path, capfd):
    """
    Test if an error is raised if none of '-f', '-mm' or '-vox' arguments are passed.
    """
    path_in = str(tmp_path / 't2/t2.nii.gz')
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        sct_resample.main(argv=['-i', path_in])
    assert pytest_wrapped_e.value.code == 2

    # Capture stdout and stderr
    out, err = capfd.readouterr()
    # Assert that the error message is printed to stderr
    assert "You need to specify one of those three arguments" in err


def test_sct_resample_both_f_and_mm_arguments(tmp_path, capfd):
    """
    Test if an error is raised if both -f and -mm arguments are passed.
    """
    path_in = str(tmp_path / 't2/t2.nii.gz')
    path_out = str(tmp_path / 'resampled.nii.gz')
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        sct_resample.main(argv=['-i', path_in, '-o', path_out, '-f', '0.5x0.5x1', '-mm', '1x1x3'])
    assert pytest_wrapped_e.value.code == 2

    # Capture stdout and stderr
    out, err = capfd.readouterr()
    # Assert that the error message is printed to stderr
    assert "You need to specify ONLY one of those three arguments" in err
