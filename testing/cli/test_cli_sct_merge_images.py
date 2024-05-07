# pytest unit tests for sct_merge_images

import pytest
import logging

from spinalcordtoolbox.scripts import sct_merge_images
from spinalcordtoolbox.utils.sys import sct_test_path

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
def test_sct_merge_images_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_merge_images.main(argv=['-i',
                                sct_test_path('template', 'template', 'PAM50_small_cord.nii.gz'),
                                sct_test_path('t2', 't2_seg-manual.nii.gz'),
                                '-w',
                                sct_test_path('mt', 'warp_template2mt.nii.gz'),
                                sct_test_path('t2', 'warp_template2anat.nii.gz'),
                                '-d', sct_test_path('mt', 'mt1.nii.gz')])
