import os
import sys
import logging

import pytest

from spinalcordtoolbox.scripts import sct_dmri_display_bvecs

logger = logging.getLogger(__name__)


# FIXME: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3388
def test_sct_dmri_display_bvecs_png_exists():
    """Run the CLI script."""
    sct_dmri_display_bvecs.main(argv=['-bvec', 'sct_testing_data/dmri/bvecs.txt'])
    assert os.path.exists('bvecs.png')
    os.unlink('bvecs.png')
