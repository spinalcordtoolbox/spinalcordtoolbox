
import sys, os
from tempfile import TemporaryDirectory
import pytest

import multiprocessing

from spinalcordtoolbox.utils import sct_dir_local_path
sys.path.append(sct_dir_local_path('scripts'))
from spinalcordtoolbox import resampling
import spinalcordtoolbox.reports.qc as qc
from spinalcordtoolbox.image import Image
import spinalcordtoolbox.reports.slice as qcslice


def gen_qc(args):
    i, path_qc = args

    t2_image = sct_dir_local_path('sct_testing_data', 't2', 't2.nii.gz')
    t2_seg = sct_dir_local_path('sct_testing_data', 't2', 't2_seg-manual.nii.gz')

    qc.generate_qc(fname_in1=t2_image, fname_seg=t2_seg, path_qc=path_qc, process="sct_deepseg_gm")
    return True


def test_many_qc():
    """Test many qc images can be made in parallel"""
    if multiprocessing.cpu_count() < 2:
        pytest.skip("Can't test parallel behaviour")

    with TemporaryDirectory(prefix="sct-qc-") as tmpdir:
        # install: sct_download_data -d sct_testing_data
        with multiprocessing.Pool(2) as p:
            p.map(gen_qc, ((i, tmpdir) for i in range(5)))
