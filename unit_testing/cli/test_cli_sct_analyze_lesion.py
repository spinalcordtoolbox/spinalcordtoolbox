import os

import pytest
import logging

from spinalcordtoolbox.scripts import sct_analyze_lesion

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_analyze_lesion_output_file_exists():
    """Run the CLI script and verify output file exists."""
    sct_analyze_lesion.main(argv=['-m', 't2/t2_seg-manual.nii.gz', '-s', 't2/t2_seg-manual.nii.gz'])
    os.path.exists('t2_seg-manual_analyzis.pkl')
