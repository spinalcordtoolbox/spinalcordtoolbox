import pytest
import logging

from spinalcordtoolbox.scripts import sct_compute_mscc

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_compute_mscc_value_against_groundtruth():
    """Run the CLI script and verify that computed mscc value is equivalent to known ground truth value."""
    di, da, db = 6.85, 7.65, 7.02
    # FIXME: The results of "sct_compute_mscc" are not actually verified. Instead, the "mscc" function is called,
    #        and THOSE results are verified instead.
    # This was copied as-is from the existing 'sct_testing' test, but should be fixed at a later date.
    sct_compute_mscc.main(argv=['-di', str(di), '-da', str(da), '-db', str(db)])
    mscc = sct_compute_mscc.mscc(di=di, da=da, db=db)
    assert mscc == pytest.approx(6.612133606, abs=1e-4)
