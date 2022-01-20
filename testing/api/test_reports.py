#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.reports

import os
import logging

import pytest
import numpy as np

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.reports.slice import Sagittal
from spinalcordtoolbox.utils import sct_test_path
import spinalcordtoolbox.reports.qc as qc
import spinalcordtoolbox.reports.slice as qcslice


logger = logging.getLogger()
handle = logging.StreamHandler()
handle.setLevel(logging.DEBUG)
logger.addHandler(handle)


def labeled_data_test_params(path_in=sct_test_path('t2', 't2.nii.gz'),
                             path_seg=sct_test_path('t2', 'labels.nii.gz')):
    """Generate image/label pairs for various test cases of
    test_sagittal_slice_get_center_spit."""
    im_in = Image(path_in)            # Base anatomical image
    im_seg_labeled = Image(path_seg)  # Base labeled segmentation
    assert np.count_nonzero(im_seg_labeled.data) >= 2, "Labeled segmentation image has fewer than 2 labels"

    # Create image with all but one label removed
    im_seg_one_label = im_seg_labeled.copy()
    for x, y, z in np.argwhere(im_seg_one_label.data)[1:]:
        im_seg_one_label.data[x, y, z] = 0

    # Create image with no labels
    im_seg_no_labels = im_seg_labeled.copy()
    for x, y, z in np.argwhere(im_seg_no_labels.data):
        im_seg_no_labels.data[x, y, z] = 0

    return [pytest.param(im_in, im_seg_labeled, id='multiple_labels'),
            pytest.param(im_in, im_seg_one_label, id='one_label'),
            pytest.param(im_in, im_seg_no_labels, id='no_labels')]


@pytest.mark.parametrize('im_in,im_seg', labeled_data_test_params())
def test_sagittal_slice_get_center_spit(im_in, im_seg):
    """Test that get_center_split returns a valid index list."""
    assert im_in.orientation == im_seg.orientation, "im_in and im_seg aren't in the same orientation"
    qcslice = Sagittal([im_in, im_seg], p_resample=None)

    if np.count_nonzero(im_seg.data) == 0:
        # If im_seg contains no labels, get_center_spit should fail
        with pytest.raises(ValueError):
            qcslice.get_center_spit()
    else:
        # Otherwise, index list should be n_SI long. (See issue #3087)
        index = qcslice.get_center_spit()
        for i, axis in enumerate(im_in.orientation):
            if axis in ['S', 'I']:
                assert len(index) == im_in.data.shape[i], "Index list doesn't have expected length (n_SI)"


@pytest.fixture()
def t2_image():
    t2_file = sct_test_path('t2', 't2.nii.gz')
    return Image(t2_file)


@pytest.fixture()
def t2_seg_image():
    t2_seg_file = sct_test_path('t2', 't2_seg-manual.nii.gz')
    return Image(t2_seg_file)


@pytest.fixture()
def t2_path():
    t2_path = sct_test_path('t2')
    return t2_path


def assert_qc_assets(path):
    files = ('index.html', 'css/style.css', 'js/main.js')
    for file_name in files:
        assert os.path.exists(os.path.join(path, file_name))

    assert os.path.isdir(os.path.join(path, 'imgs'))


def test_label_vertebrae(t2_image, t2_seg_image):
    param = qc.Params(t2_image.absolutepath, 'sct_label_vertebrae', ['-a', '-b'], 'Sagittal', '/tmp')
    report = qc.QcReport(param, 'Test label vertebrae')

    @qc.QcImage(report, 'spline36', [qc.QcImage.label_vertebrae, ], process=param.command)
    def test(qslice):
        return qslice.single()

    test(qcslice.Sagittal([t2_image, t2_seg_image]))
    assert os.path.isfile(param.abs_bkg_img_path())
    assert os.path.isfile(param.abs_overlay_img_path())


def test_propseg(t2_image, t2_seg_image):
    param = qc.Params(t2_image.absolutepath, 'sct_propseg', ['-a'], 'Axial', '/tmp')
    report = qc.QcReport(param, 'Test usage')

    @qc.QcImage(report, 'none', [qc.QcImage.listed_seg, ], process=param.command)
    def test(qslice):
        return qslice.mosaic()

    test(qcslice.Axial([t2_image, t2_seg_image]))
    assert os.path.isfile(param.abs_bkg_img_path())
    assert os.path.isfile(param.abs_overlay_img_path())
    assert os.path.isfile(param.qc_results)


# FIXME: The following tests are broken, as they use outdated syntax for classes, attributes, etc.
#        For example, the Coronal slice type no longer exists, having been removed in 2018 (PR #1667)
#        These tests could could be deleted, but saving them lets them be used to model new tests after.
#
# def test_slices(t2_image, t2_seg_image):
#     sagittal = qcslice.Sagittal(t2_image, t2_seg_image)
#     coronal = qcslice.Coronal(t2_image, t2_seg_image)
#     axial = qcslice.Axial(t2_image, t2_seg_image)
#
#     for s in [sagittal, coronal, axial]:
#         assert s.get_aspect(s.image) == 1
#         s.mosaic(50, 50)
#         with pytest.raises(ValueError):
#             s.mosaic(0, 0)
#         s.mosaic(3, 5)
#         s.mosaic(2, 2)
#
#     assert sagittal.get_dim(sagittal.image) == 52
#     assert len(sagittal.get_center()) == 2
#     assert coronal.get_dim(coronal.image) == 60
#     assert len(coronal.get_center()) == 2
#     assert axial.get_dim(axial.image) == 55

# def test_template_generator():
#     param = qc.Params(['ofolder=/tmp/qc1', 'ncol=4', 'autoview=1', 'generate=0', 'thresh=11'])
#     context = {
#         'axis': 'Axial',
#         'title': 'Test Template TITLE',
#
#     }
#
#     test_report = report.ReportGenerator(param, context)
#     test_report.refresh_index_file()
#
#     with open('/tmp/qc1/index.json') as fd:
#         output = json.loads(fd.read())
#     assert os.path.exists('/tmp/qc1/index.json')
#     assert output[0]['title'] == context['title']

#
# def test_params():
#     param = qc.Params(['ofolder=/tmp/qc1', 'ncol=4', 'autoview=1', 'generate=0', 'thresh=11'])
#     template = qc.QcReport('testing', param, ['-a', '-b', '-c toto'], 'description example')
#
#     assert param.destination_folder == '/tmp/qc1'
#     assert param.nb_column == 4
#     assert param.generate_report is False
#     assert param.threshold == 11
#     assert param.show_report is True
#
#     assert template.assets_folder
#     assert template.template_folder
#     template.copy_assets()
