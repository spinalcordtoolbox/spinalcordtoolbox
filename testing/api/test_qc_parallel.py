import sys
from tempfile import TemporaryDirectory
import pytest

import multiprocessing

from spinalcordtoolbox.utils import sct_test_path, sct_dir_local_path
sys.path.append(sct_dir_local_path('scripts'))
import spinalcordtoolbox.reports.qc as qc


def gen_qc(path_qc):
    t2_image = sct_test_path('t2', 't2.nii.gz')
    t2_seg = sct_test_path('t2', 't2_seg-manual.nii.gz')
    qc.generate_qc(fname_in1=t2_image, fname_seg=t2_seg, path_qc=path_qc, process="sct_deepseg_gm")


def test_many_qc():
    """Test many qc images can be made in parallel"""
    if multiprocessing.cpu_count() < 2:
        pytest.skip("Can't test parallel behaviour")

    p = multiprocessing.Pool(2)

    with TemporaryDirectory(prefix="sct-qc-") as tmpdir:
        # This `try, finally` pattern mitigates hanging with pytest-cov
        # See: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3661#issuecomment-1029057900
        try:
            p.map(gen_qc, [tmpdir] * 5)
        finally:
            p.close()  # Marks the pool as closed.
            p.join()   # Waits for workers to exit.
