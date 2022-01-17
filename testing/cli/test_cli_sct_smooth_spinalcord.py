import pytest
import logging
import os

from spinalcordtoolbox.scripts import sct_smooth_spinalcord

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_smooth_spinalcord_check_output_files(tmp_path):
    """Run the CLI script and ensure output files exist."""
    fname_out = os.path.join(str(tmp_path), "test_smooth.nii.gz")
    sct_smooth_spinalcord.main(argv=['-i', 't2/t2.nii.gz', '-s', 't2/t2_seg-manual.nii.gz', '-smooth', '0,0,5',
                                     '-o', fname_out])
    assert os.path.isfile(fname_out)

    # Currently, 3 of SCT's CLI scripts output "straightening cache files" to the working directory. There's no way
    # to output these files into a temp directory, because `sct_smooth_spinalcord` has no `-ofolder` option.
    # So, for now we have to explicitly remove the files as to not pollute the working directory.
    straightening_cache_files = ['straightening.cache', 'straight_ref.nii.gz',
                                 'warp_straight2curve.nii.gz', 'warp_curve2straight.nii.gz']
    for f in straightening_cache_files:
        assert os.path.isfile(f)
        os.unlink(f)
