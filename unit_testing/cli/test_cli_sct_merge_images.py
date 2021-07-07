import pytest
import logging

from spinalcordtoolbox.scripts import sct_merge_images

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_merge_images_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_merge_images.main(argv=['-i', 'template/template/PAM50_small_cord.nii.gz,t2/t2_seg-manual.nii.gz',
                                '-w', 'mt/warp_template2mt.nii.gz,t2/warp_template2anat.nii.gz',
                                '-d' 'mt/mt1.nii.gz'])
