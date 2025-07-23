# pytest unit tests for spinalcordtoolbox.reports.qc in parallel

import pytest
import logging
import sys

import multiprocessing

from spinalcordtoolbox.utils.sys import sct_test_path
import spinalcordtoolbox.reports.qc as qc


def gen_qc(path_qc):
    t2_image = sct_test_path('t2', 't2.nii.gz')
    t2_seg = sct_test_path('t2', 't2_seg-manual.nii.gz')
    qc.generate_qc(fname_in1=t2_image, fname_seg=t2_seg, path_qc=path_qc, process="sct_deepseg_gm")


def test_many_qc(tmp_path_qc):
    """Test many qc images can be made in parallel"""
    # Turn on debug logging for fs.py to check mutex acquire/release times
    # To check this output locally, run: `sct_testing -o log_cli=true testing/api/test_qc_parallel.py::test_many_qc`
    screen_handler = logging.StreamHandler(stream=sys.stderr)
    logger = logging.getLogger("portalocker.utils")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(screen_handler)

    if multiprocessing.cpu_count() < 2:
        pytest.skip("Can't test parallel behaviour")

    p = multiprocessing.Pool(multiprocessing.cpu_count())

    # This `try, finally` pattern mitigates hanging with pytest-cov
    # See: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3661#issuecomment-1029057900
    try:
        p.map(gen_qc, [tmp_path_qc] * (multiprocessing.cpu_count() * 2))
    finally:
        p.close()  # Marks the pool as closed.
        p.join()   # Waits for workers to exit.
