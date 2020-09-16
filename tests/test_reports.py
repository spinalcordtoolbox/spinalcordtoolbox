# -*- coding: utf-8 -*-
import json
import os
import logging

import pytest

import spinalcordtoolbox.reports.qc as qc
import spinalcordtoolbox.reports.slice as qcslice


logger = logging.getLogger()
handle = logging.StreamHandler()
handle.setLevel(logging.DEBUG)
logger.addHandler(handle)


@pytest.fixture()
def t2_image():
    sct_test_data = 'sct_testing_data'
    t2_file = os.path.join(sct_test_data, 't2', 't2.nii.gz')
    return t2_file


@pytest.fixture()
def t2_seg_image():
    sct_test_data = 'sct_testing_data'
    t2_seg_file = os.path.join(sct_test_data, 't2', 't2_seg-manual.nii.gz')
    return t2_seg_file


@pytest.fixture()
def t2_path():
    sct_test_data = 'sct_testing_data'
    t2_path = os.path.join(sct_test_data, 't2')
    return t2_path


def assert_qc_assets(path):
    files = ('index.html', 'css/style.css', 'js/main.js')
    for file_name in files:
        assert os.path.exists(os.path.join(path, file_name))

    assert os.path.isdir(os.path.join(path, 'imgs'))

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


def test_label_vertebrae(t2_image, t2_seg_image):
    param = qc.Params(t2_image, 'sct_label_vertebrae', ['-a', '-b'], 'Sagittal', '/tmp')
    report = qc.QcReport(param, 'Test label vertebrae')

    @qc.QcImage(report, 'spline36', [qc.QcImage.label_vertebrae, ])
    def test(qslice):
        return qslice.single()

    test(qcslice.Sagittal(t2_image, t2_seg_image))
    assert os.path.isfile(param.abs_bkg_img_path())
    assert os.path.isfile(param.abs_overlay_img_path())


def test_propseg(t2_image, t2_seg_image):
    param = qc.Params(t2_image, 'sct_propseg', ['-a'], 'Axial', '/tmp')
    report = qc.QcReport(param, 'Test usage')

    @qc.QcImage(report, 'none', [qc.QcImage.listed_seg, ])
    def test(qslice):
        return qslice.mosaic()

    test(qcslice.Axial(t2_image, t2_seg_image))
    assert os.path.isfile(param.abs_bkg_img_path())
    assert os.path.isfile(param.abs_overlay_img_path())
    assert os.path.isfile(param.qc_results)



# def test_segment_graymatter(t2_image, t2_seg_image):
#     param = qc.Params(['ofolder=/tmp/graymatter', ])
#     report = qc.QcReport('test_segment_graymatter', param, ['-e', '--wow'], 'Testing the usage')
#
#     @qc.QcImage(report, 'bicubic', [qc.QcImage.sequential_seg, qc.QcImage.colorbar])
#     def test(qslice, nb_col, thr):
#         img, seg = qslice.mosaic(nb_col)
#         seg[seg < thr] = 0
#         return img, seg
#
#     test(qcslice.Axial(t2_image, t2_seg_image), param.nb_column, param.threshold)
#     assert_qc_assets('/tmp/qc')
