from pytest_console_scripts import script_runner
import pytest
import logging

import numpy as np

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.scripts import sct_label_utils

logger = logging.getLogger(__name__)


@pytest.mark.script_launch_mode('subprocess')
def test_sct_label_utils_backwards_compat(script_runner):
    ret = script_runner.run('sct_testing', '--function', 'sct_label_utils')
    logger.debug(f"{ret.stdout}")
    logger.debug(f"{ret.stderr}")
    assert ret.success
    assert ret.stderr == ''


def test_create_seg_mid(tmp_path):
    """Test the '-create-seg-mid' option in sct_label_utils."""
    input = 'sct_testing_data/t2/t2_seg-manual.nii.gz'
    output = str(tmp_path/'t2_seg_labeled.nii.gz')

    # Create a single label using the new syntax
    sct_label_utils.main(['-i', input, '-create-seg-mid', '3', '-o', output])
    output_img = Image(output)
    labels = np.argwhere(output_img.data)
    assert len(labels) == 1

    # Ensure slice coordinate of label is centered at midpoint of I-S axis
    for coord, axis, shape in zip(labels[0], output_img.orientation, Image(output).data.shape):
        if axis in ['I', 'S']:
            assert coord == round(shape/2)

    # Old syntax for this behavior should not be allowed
    with pytest.raises(DeprecationWarning):
        sct_label_utils.main(['-i', input, '-create-seg', '-1,3', '-o', output])
