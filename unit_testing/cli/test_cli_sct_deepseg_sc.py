import logging
import os

import pytest

from spinalcordtoolbox.scripts import sct_deepseg_sc

logger = logging.getLogger(__name__)


def test_sct_deepseg_sc_check_output_exists(tmp_path):
    fname_out = str(tmp_path / 'test_seg.nii.gz')
    sct_deepseg_sc.main(argv=['-i', 'sct_testing_data/t2/t2.nii.gz', '-c', 't2', '-o', fname_out])
    assert os.path.isfile(fname_out)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_deepseg_sc_qc_report_exists():
    dir_qc = 'testing-qc'
    sct_deepseg_sc.main(argv=['-i', 't2/t2.nii.gz', '-c', 't2', '-qc', dir_qc])
    assert os.path.isfile(os.path.join(dir_qc, 'index.html'))
