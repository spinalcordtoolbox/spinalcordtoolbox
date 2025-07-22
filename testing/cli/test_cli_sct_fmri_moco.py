# pytest unit tests for sct_fmri_moco

import pytest
import logging

from spinalcordtoolbox.scripts import sct_fmri_moco, sct_deepseg, sct_maths
from spinalcordtoolbox.utils.sys import sct_test_path

logger = logging.getLogger(__name__)


@pytest.fixture(scope='module')
def fmri_mean_seg(tmp_path_factory):
    """Mean segmented image for QC report generation."""
    tmp_path = tmp_path_factory.mktemp('fmri_mean_seg')
    path_mean = str(tmp_path / 'fmri_mean.nii.gz')
    path_out = str(tmp_path / 'fmri_mean_seg.nii.gz')

    sct_maths.main(argv=['-i', sct_test_path('fmri', 'fmri.nii.gz'), '-mean', 't',
                         '-o', path_mean])
    sct_deepseg.main(argv=['spinalcord', '-i', path_mean, '-o', path_out, '-qc', str(tmp_path)])
    return path_out


@pytest.mark.sct_testing
def test_sct_fmri_moco_no_checks(tmp_path_qc, fmri_mean_seg):
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_fmri_moco.main(argv=['-i', sct_test_path('fmri', 'fmri_r.nii.gz'), '-g', '4', '-x', 'nn', '-r', '0',
                             '-qc', tmp_path_qc, '-qc-seg', fmri_mean_seg])
