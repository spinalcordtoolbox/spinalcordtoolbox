# pytest unit tests for sct_fmri_moco

import pytest
import logging

from spinalcordtoolbox.scripts import sct_fmri_moco, sct_deepseg, sct_maths, sct_create_mask
from spinalcordtoolbox.utils.sys import sct_test_path

logger = logging.getLogger(__name__)


@pytest.fixture(scope='module')
def fmri_mean(tmp_path_factory, tmp_path_qc):
    """Mean segmented image for QC report generation."""
    tmp_path = tmp_path_factory.mktemp('fmri_mean')
    path_out = str(tmp_path / 'fmri_mean.nii.gz')

    sct_maths.main(argv=['-i', sct_test_path('fmri', 'fmri.nii.gz'), '-mean', 't', '-o', path_out])
    return path_out


@pytest.fixture(scope='module')
def fmri_mean_seg(tmp_path_factory, tmp_path_qc, fmri_mean):
    """Mean segmented image for QC report generation."""
    tmp_path = tmp_path_factory.mktemp('fmri_mean_seg')
    path_out = str(tmp_path / 'fmri_mean_seg.nii.gz')

    sct_deepseg.main(argv=['spinalcord', '-i', fmri_mean, '-o', path_out, '-qc', str(tmp_path_qc)])
    return path_out


@pytest.fixture(scope='module')
def fmri_mask(tmp_path_factory, fmri_mean, fmri_mean_seg):
    """Mask image for testing."""
    tmp_path = tmp_path_factory.mktemp('fmri_mask')
    path_out = str(tmp_path / 'fmri_mask.nii.gz')
    sct_create_mask.main(argv=['-i', fmri_mean, '-p', f'centerline,{fmri_mean_seg}', '-size', '35mm', '-o', path_out])
    return path_out


@pytest.mark.sct_testing
def test_sct_fmri_moco_no_checks(tmp_path_qc, fmri_mean_seg):
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_fmri_moco.main(argv=['-i', sct_test_path('fmri', 'fmri_r.nii.gz'), '-g', '4', '-x', 'nn', '-r', '0',
                             '-qc', tmp_path_qc, '-qc-seg', fmri_mean_seg])


@pytest.mark.parametrize('target_vol', ['ref', 0, 5])
def test_sct_fmri_moco_target_volume(capsys, tmp_path_qc, fmri_mean, fmri_mean_seg, target_vol):
    """Run the CLI script against a specified target volume (either `-ref` or `-param num_target`)."""
    # run `sct_fmri_moco` with a given target volume
    target_args = ['-ref', fmri_mean] if target_vol == 'ref' else ['-param', f'num_target={target_vol}']
    sct_fmri_moco.main(argv=['-i', sct_test_path('fmri', 'fmri_r.nii.gz'), '-g', '4', '-x', 'nn', '-r', '0',
                             '-qc', tmp_path_qc, '-qc-seg', fmri_mean_seg] + target_args)

    # make sure the target volume was mentioned in the logging output
    captured = capsys.readouterr()
    target_msgs = [line for line in captured.out.split("\n") if line.startswith("Target:")]
    assert len(target_msgs) > 0, "No target message found in output"
    assert str(target_vol) in target_msgs[0], f"Expected target volume {target_vol} not found in output"


@pytest.mark.parametrize("group_size", [-1, 0, 1.5, 'NaN'])
def test_sct_fmri_moco_invalid_group_values(tmp_path, tmp_path_qc, fmri_mean_seg, group_size):
    """Ensure that invalid group sizes return a parsing error."""
    with pytest.raises(SystemExit) as e:
        sct_fmri_moco.main(argv=['-i', sct_test_path('fmri', 'fmri.nii.gz'),
                                 '-g', str(group_size), '-x', 'nn', '-r', '0',
                                 '-ofolder', str(tmp_path),
                                 '-qc', tmp_path_qc, '-qc-seg', fmri_mean_seg])
    assert e.value.code == 2


@pytest.mark.sct_testing
def test_sct_fmri_moco_dl_no_checks(tmp_path_qc, fmri_mean, fmri_mask, tmp_path, fmri_mean_seg):
    """Run the CLI script with '-m' and '-ref' option and using '-dl' algorithm."""
    sct_fmri_moco.main(argv=['-i', sct_test_path('fmri', 'fmri.nii.gz'),
                             '-ref', fmri_mean, '-m', fmri_mask, '-ofolder', str(tmp_path), '-dl',
                             '-qc', tmp_path_qc, '-qc-seg', fmri_mean_seg])
    # NB: We skip checking params because there are no output moco params from DL-module
