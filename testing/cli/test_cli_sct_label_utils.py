# pytest unit tests for sct_label_utils

import pytest
import logging

import numpy as np

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.types import Coordinate
from spinalcordtoolbox.utils.sys import sct_test_path
from spinalcordtoolbox.scripts import sct_label_utils, sct_create_mask

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
def test_sct_label_utils_cubic_to_point():
    """Run the CLI script and verify the resulting center of mass coordinate."""
    fname_out = 'test_centerofmass.nii.gz'
    sct_label_utils.main(argv=['-i', 't2/t2_seg-manual.nii.gz', '-cubic-to-point', '-o', fname_out])
    # Note: Old, broken 'sct_testing' test used '31,28,25,1' as ground truth. Is this a regression?
    assert Image(fname_out).getNonZeroCoordinates() == [Coordinate([31, 27, 25, 1])]


@pytest.mark.sct_testing
def test_sct_label_utils_create():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_label_utils.main(argv=['-i', 't2/t2_seg-manual.nii.gz', '-create', '1,1,1,1:2,2,2,2'])


def test_create_seg_mid(tmp_path):
    """Test the '-create-seg-mid' option in sct_label_utils."""
    input = sct_test_path('t2', 't2_seg-manual.nii.gz')
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


def test_project_centerline(tmp_path):
    """Test the '-project-centerline' option in sct_label_utils"""
    # Use an image as a shape reference
    img = str(tmp_path/'t2.nii.gz')
    Image(sct_test_path('t2', 't2.nii.gz')).change_orientation('RPI').save(img)

    # Create a mask with a vertical line following the z axis with the coordinates x=20 and y=15
    mask = str(tmp_path/'t2_seg.nii.gz')
    sct_create_mask.main([
         '-i', img,
         '-o', mask,
         '-size', '1',
         '-p', 'coord,20x15',
         '-f', 'box'
         ])

    # Create an image with 3 points
    ref = str(tmp_path/'t2_seg_labeled.nii.gz')
    sct_label_utils.main([
        '-i', img,
        '-o', ref,
        '-create', '1,1,1,1:1,2,3,4:10,11,25,25'
        ])

    # Project the ref point on the previous line
    out = str(tmp_path/'t2_seg_labeled_projected.nii.gz')
    sct_label_utils.main([
        '-i', mask,
        '-o', out,
        '-project-centerline', ref
        ])

    # The coordinates of this projection should be equal to x=20, y=15, z=1
    assert Image(out).getNonZeroCoordinates() == [Coordinate([20, 15, 1, 1]), Coordinate([20, 15, 3, 4]), Coordinate([20, 15, 25, 25])]
