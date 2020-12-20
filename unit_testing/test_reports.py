#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.reports

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.reports.slice import Sagittal


def test_sagittal_slice_get_center_spit():
    """
    This test currently only tests get_center_split when the mask only has one
    label. So, much of the function is untested, therefore:
        -TODO: Test get_center_split when mask is empty (raise error)
        -TODO: Test get_center_split when mask has more than one label
    """
    # TODO: Figure out how to replace subject data with testing data equivalent
    fname_in = '../sub-1002358_T1w_RPI_r_gradcorr.nii.gz'
    fname_seg = '../sub-1002358_T1w_RPI_r_gradcorr_labels-manual.nii.gz'
    qcslice = Sagittal([Image(fname_in), Image(fname_seg)], p_resample=None)
    index = qcslice.get_center_spit()
    # Index list should be n_SI long. RPI image -> SI axis = shape[2] (See issue #3087)
    assert len(index) == Image(fname_in).data.shape[2]
    # Index list should only contain one slice value
    assert len(set(index)) == 1
