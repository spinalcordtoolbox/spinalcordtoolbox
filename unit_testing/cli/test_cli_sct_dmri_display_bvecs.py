import os
import sys
import logging

import pytest

from spinalcordtoolbox.scripts import sct_dmri_display_bvecs

logger = logging.getLogger(__name__)


def test_sct_dmri_display_bvecs_png_exists():
    """Run the CLI script."""
    sct_dmri_display_bvecs.main(argv=['-bvec', 'sct_testing_data/dmri/bvecs.txt'])
    assert os.path.exists('bvecs.png')
    os.unlink('bvecs.png')


@pytest.mark.skipif(sys.platform == "darwin", reason="Script uses pyplot.show, causing macOS 10.15 CI runners to hang.")
# FIXME: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3388
def test_sct_dmri_display_bvecs_png_exists_bvec_bval_inputs():
    """Run the CLI script."""
    sct_dmri_display_bvecs.main(argv=['-bvec', 'sct_testing_data/dmri/bvecs.txt',
                                      '-bval', 'sct_testing_data/dmri/bvals.txt'])
    assert os.path.exists('bvecs.png')
    os.unlink('bvecs.png')
