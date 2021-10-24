import pytest
import logging

from numpy import allclose, genfromtxt

from spinalcordtoolbox.scripts import sct_dmri_moco, sct_image, sct_crop_image, sct_create_mask

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_dmri_moco_check_params(tmp_path):
    """Run the CLI script and validate output moco params."""
    sct_dmri_moco.main(argv=['-i', 'dmri/dmri.nii.gz', '-bvec', 'dmri/bvecs.txt', '-g', '3', '-x', 'nn', '-r', '0',
                             '-ofolder', str(tmp_path)])

    lresults = genfromtxt(tmp_path / "moco_params.tsv", skip_header=1, delimiter='\t')[:, 0]
    lgroundtruth = [0.00047529041677414337, -1.1970542445283172e-05, -1.1970542445283172e-05, -1.1970542445283172e-05,
                    -0.1296642741802682, -0.1296642741802682, -0.1296642741802682]
    assert allclose(lresults, lgroundtruth)


@pytest.fixture
def dmri_mask(tmp_path):
    """Mask image for testing."""
    path_out = str(tmp_path / 'mask.nii')
    sct_create_mask.main(argv=['-i', 'dmri/dmri_T0000.nii.gz', '-p', 'center', '-size', '5mm', '-f', 'gaussian',
                               '-o', path_out])

    return path_out


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_dmri_moco_with_mask_check_params(tmp_path, dmri_mask):
    """Run the CLI script with '-m' option and validate output moco params."""
    sct_dmri_moco.main(argv=['-i', 'dmri/dmri.nii.gz', '-bvec', 'dmri/bvecs.txt', '-g', '3', '-r', '0',
                             '-m', dmri_mask, '-ofolder', str(tmp_path)])

    lresults = genfromtxt(tmp_path / "moco_params.tsv", skip_header=1, delimiter='\t')[:, 0]
    lgroundtruth = [0.008032332623754357, 0.0037734940916436697, 0.0037734940916436697, 0.0037734940916436697,
                    -0.01502861167728611, -0.01502861167728611, -0.01502861167728611]
    assert allclose(lresults, lgroundtruth)


@pytest.fixture
def dmri_ail_cropped(tmp_path):
    """Reorient image to sagittal for testing another orientation (and crop to save time)."""
    path_out = str(tmp_path / 'dmri_AIL_crop.nii')
    sct_image.main(argv=['-i', 'dmri/dmri.nii.gz', '-setorient', 'AIL', '-o', 'dmri/dmri_AIL.nii'])
    sct_crop_image.main(argv=['-i', 'dmri/dmri_AIL.nii', '-zmin', '19', '-zmax', '21', '-o', path_out])

    return path_out


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_dmri_moco_sagittal_no_checks(tmp_path, dmri_ail_cropped):
    """Run the CLI script, but don't check anything."""
    sct_dmri_moco.main(argv=['-i', dmri_ail_cropped, '-bvec', 'dmri/bvecs.txt', '-x', 'nn', '-r', '0',
                             '-ofolder', str(tmp_path)])
    # NB: We skip checking params because there are no output moco params for sagittal images (*_AIL)
