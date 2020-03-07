
import sys, os
import pytest

import multiprocessing

from spinalcordtoolbox.utils import __sct_dir__
sys.path.append(os.path.join(__sct_dir__, 'scripts'))
from spinalcordtoolbox import resampling
import spinalcordtoolbox.reports.qc as qc
from spinalcordtoolbox.image import Image
import spinalcordtoolbox.reports.slice as qcslice



def gen_qc(i):
    t2_image = os.path.join(__sct_dir__, 'sct_testing_data', 't2', 't2.nii.gz')
    t2_seg = os.path.join(__sct_dir__, 'sct_testing_data', 't2', 't2_seg.nii.gz')

    qc.generate_qc(fname_in1 = t2_image, fname_seg = t2_seg, path_qc = "/tmp/qc",
                   process = "sct_deepseg_gm")


def test_many_qc():
    """Test many qc images can be made in parallel"""
    if multiprocessing.cpu_count() < 2:
        pytest.skip("Can't test parallel behaviour")

    # install: sct_download_data -d sct_testing_data
    with multiprocessing.Pool(6) as p:
        p.map(gen_qc, range(300))


