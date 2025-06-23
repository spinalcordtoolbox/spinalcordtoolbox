# pytest unit tests for sct_deepseg_sc

import logging
import os

import pytest
import numpy as np

from spinalcordtoolbox.scripts import sct_deepseg_sc
from spinalcordtoolbox.utils.sys import sct_test_path
from spinalcordtoolbox.image import Image

logger = logging.getLogger(__name__)


def test_sct_deepseg_sc_check_output_qform_sform(tmp_path):
    fname_in = sct_test_path('t2', 't2.nii.gz')
    fname_out = str(tmp_path / 'test_seg.nii.gz')
    with pytest.deprecated_call():
        sct_deepseg_sc.main(argv=['-i', fname_in, '-c', 't2', '-o', fname_out])
    # Ensure sform/qform of the segmentation matches that of the input images
    im_in = Image(fname_in)
    im_seg = Image(fname_out)
    assert np.array_equal(im_in.header.get_sform(), im_seg.header.get_sform())
    assert np.array_equal(im_in.header.get_qform(), im_seg.header.get_qform())
    assert im_in.header['sform_code'] == im_seg.header['sform_code']
    assert im_in.header['qform_code'] == im_seg.header['qform_code']


@pytest.mark.sct_testing
def test_sct_deepseg_sc_qc_report_exists():
    dir_qc = 'testing-qc'
    with pytest.deprecated_call():
        sct_deepseg_sc.main(argv=['-i', sct_test_path('t2', 't2.nii.gz'), '-c', 't2', '-qc', dir_qc])
    assert os.path.isfile(os.path.join(dir_qc, 'index.html'))
