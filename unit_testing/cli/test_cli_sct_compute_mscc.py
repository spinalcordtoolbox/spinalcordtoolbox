import pytest
import logging

from spinalcordtoolbox.scripts import sct_compute_mscc

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_compute_mscc_value_against_groundtruth():
    """Run the CLI script and verify that computed mscc value is equivalent to known ground truth value."""
    di, da, db = 6.85, 7.65, 7.02

    # FIXME: 'sct_testing' test called both the CLI script and the function inside the script.
    #  This should be refactored so that the same functionality isn't run twice.
    sct_compute_mscc.main(argv=['-di', str(di), '-da', str(da), '-db', str(db)])  # Output not actually tested
    mscc = sct_compute_mscc.mscc(di=di, da=da, db=db)

    assert mscc == pytest.approx(6.612133606)
