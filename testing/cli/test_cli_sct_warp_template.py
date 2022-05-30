import pytest
import logging

from spinalcordtoolbox.scripts import sct_warp_template

logger = logging.getLogger(__name__)


@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_warp_template_warp_small_PAM50():
    """Warp the cropped, resampled version of the template from `sct_testing_data/template`."""
    sct_warp_template.main(argv=['-d', 'mt/mt1.nii.gz', '-w', 'mt/warp_template2mt.nii.gz',
                                 '-a', '0',  # -a is '1' by default, but atlas isn't present in 'template'
                                 '-t', 'template',
                                 '-qc', 'testing-qc'])


@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_warp_template_warp_full_PAM50():
    """Warp the full PAM50 template (i.e. the one that is downloaded to `data/PAM50` during installation)."""
    sct_warp_template.main(argv=['-d', 'mt/mt1.nii.gz', '-w', 'mt/warp_template2mt.nii.gz',
                                 '-a', '1', '-s', '1', '-histo', '1',
                                 '-qc', 'testing-qc'])
