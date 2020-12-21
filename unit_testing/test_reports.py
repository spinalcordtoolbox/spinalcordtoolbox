#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.reports

import pytest
import numpy as np

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.reports.slice import Sagittal


def prepare_labeled_data():
    """Generate image/label pairs for various test cases of
    test_sagittal_slice_get_center_spit."""
    # Base labeled image
    im_seg = Image('sct_testing_data/t2/labels.nii.gz')

    # Create image with all but one label removed
    im_seg_one_label = im_seg.copy()
    for x, y, z in np.argwhere(im_seg_one_label.data)[1:]:
        im_seg_one_label.data[x, y, z] = 0

    # Create image with no labels
    im_seg_no_labels = im_seg.copy()
    for x, y, z in np.argwhere(im_seg_no_labels.data):
        im_seg_no_labels.data[x, y, z] = 0

    return [im_seg, im_seg_one_label, im_seg_no_labels]


def test_sagittal_slice_get_center_spit():
    """Test that get_center_split returns a valid index list."""
    im_in = Image('sct_testing_data/t2/t2.nii.gz')
    im_seg_list = prepare_labeled_data()

    for im_seg in im_seg_list:
        assert im_in.orientation == im_seg.orientation
        qcslice = Sagittal([im_in, im_seg], p_resample=None)

        # If im_seg contains no labels, get_center_spit should fail
        if np.count_nonzero(im_seg.data) == 0:
            with pytest.raises(ValueError):
                index = qcslice.get_center_spit()
        # Otherwise, if it contains labels, a valid index list should be returned
        else:
            index = qcslice.get_center_spit()

        # Index list should be n_SI long. (See issue #3087)
        for i, axis in enumerate(im_in.orientation):
            if axis in ['S', 'I']:
                assert len(index) == im_in.data.shape[i]
