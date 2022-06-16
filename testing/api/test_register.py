#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.register

import logging
import os

import pytest
import numpy as np

from spinalcordtoolbox.scripts.sct_register_to_template import Param
from spinalcordtoolbox.registration.core import register
from spinalcordtoolbox.registration.algorithms import (Paramreg, register_step_ants_registration, register_step_label,
                                                       register_step_ants_slice_regularized_registration)
from spinalcordtoolbox.utils import sct_test_path

logger = logging.getLogger(__name__)

# FIXME [AJ] fetch/compute input data from sct_testing_data/ instead of the manually copied hardcoded files below

@pytest.fixture
def step0_data():
    """
    """

    # FIXME
    src = ['register_test_data/src_label.nii']
    dest = ['register_test_data/dest_label_RPI.nii']

    step = Paramreg(
     step='0',
     type='label',
     algo='syn',
     metric='MeanSquares',
     iter='10',
     shrink='1',
     smooth='0',
     laplacian='0',
     gradStep='0.5',
     deformation='1x1x0',
     slicewise='0',
     init='',
     poly='5',
     filter_size=5,
     dof='Tx_Ty_Tz_Rx_Ry_Rz_Sz',
     smoothWarpXY='2',
     pca_eigenratio_th='1.6',
     rot_method='pca',
    )

    cli_params = Param()
    cli_params.debug = 2

    return src, dest, step, cli_params

@pytest.fixture
def step1_data():
    """
    """

    # FIXME
    src = ['register_test_data/src_seg.nii']
    dest = ['register_test_data/dest_seg_RPI.nii']

    step = Paramreg(
     step='1',
     type='seg',
     algo='centermassrot',
     metric='MeanSquares',
     iter='10',
     shrink='1',
     smooth='0',
     laplacian='0',
     gradStep='0.5',
     deformation='1x1x0',
     slicewise='0',
     init='',
     poly='5',
     filter_size=5,
     dof='Tx_Ty_Tz_Rx_Ry_Rz',
     smoothWarpXY='2',
     pca_eigenratio_th='1.6',
     rot_method='pca',
    )

    cli_params = Param()
    cli_params.debug = 2

    return src, dest, step, cli_params

@pytest.fixture
def step2_data():
    """
    """

    # FIXME
    src = ['register_test_data/src_seg_reg.nii']
    dest = ['register_test_data/dest_seg_RPI.nii']

    step = Paramreg(
     step='2',
     type='seg',
     algo='bsplinesyn',
     metric='MeanSquares',
     iter='3',
     shrink='1',
     smooth='1',
     laplacian='0',
     gradStep='0.5',
     deformation='1x1x0',
     slicewise='0',
     init='',
     poly='5',
     filter_size=5,
     dof='Tx_Ty_Tz_Rx_Ry_Rz',
     smoothWarpXY='2',
     pca_eigenratio_th='1.6',
     rot_method='pca',
    )

    cli_params = Param()
    cli_params.debug = 2

    return src, dest, step, cli_params

@pytest.fixture
def step_axial_data_in_same_space():
    """
    """
    src = sct_test_path('mt', 'mt0_seg.nii.gz')
    dest = sct_test_path('mt', 'mt1_seg.nii.gz')

    step = Paramreg(
        step='1',
        type='seg',
        algo='slicereg',
        metric='MeanSquares',
        iter='5',
    )

    cli_params = Param()
    cli_params.debug = 2

    return src, dest, step, cli_params

@pytest.mark.skip(reason="Need to fix input test data")
def test_register_step_label(step0_data):
    """
    """
    src, dest, step, cli_params = step0_data

    src = src[0]
    dest = dest[0]

    warp_forward_out, warp_inverse_out = register_step_label(src=src, dest=dest, step=step, verbose=cli_params.verbose)

@pytest.mark.skip(reason="TODO")
def test_register_step_slicewise():
     """
     """
     raise NotImplementedError()

@pytest.mark.skip(reason="TODO")
def test_register_step_slicewise_ants():
     """
     """
     raise NotImplementedError()

@pytest.mark.skip(reason="Need to fix input test data")
def test_register_step_ants_registration(step2_data):
    """
    """
    src, dest, step, cli_params = step2_data

    src = src[0]
    dest = dest[0]

    metricSize = '4'

    ants_registration_params = {'rigid': '', 'affine': '', 'compositeaffine': '', 'similarity': '', 'translation': '',
                                'bspline': ',10', 'gaussiandisplacementfield': ',3,0',
                                'bsplinedisplacementfield': ',5,10', 'syn': ',3,0', 'bsplinesyn': ',1,3'}

    warp_forward_out, warp_inverse_out = register_step_ants_registration(
     src=src,
     dest=dest,
     step=step,
     masking=[],
     ants_registration_params=ants_registration_params,
     padding=cli_params.padding,
     metricSize=metricSize,
     verbose=cli_params.verbose,
    )


def test_register_step_ants_slice_regularized_registration(step_axial_data_in_same_space):
    src, dest, step, cli_params = step_axial_data_in_same_space

    outfiles = _, _, txty_csv_out = register_step_ants_slice_regularized_registration(src, dest, step, metricSize='4')

    # Verify integrity of the output Tx Ty file
    txty_result = np.genfromtxt(txty_csv_out, skip_header=1, delimiter=',')
    txty_groundtruth = np.genfromtxt(
        sct_test_path('mt', 'step1TxTy_poly_groundtruth.csv'), skip_header=1, delimiter=',')
    assert txty_result == pytest.approx(txty_groundtruth, abs=1e-14)

    # tmp_path can't be used here because the output files are generated by isct_antsSliceRegularizedRegistration,
    # which doesn't easily allow the output filepaths to be modified. Instead, remove manually.
    for file in outfiles:
        os.unlink(file)


# higher level tests for step registration, regardless of step)
@pytest.mark.skip(reason="Need to fix input test data")
def test_register_step0(step0_data):
    """
    """
    src, dest, step, cli_params = step0_data
    warp_forward_out, warp_inverse_out = register(src=src, dest=dest, step=step, param=cli_params)

@pytest.mark.skip(reason="Need to fix input test data")
def test_register_step1(step1_data):
    """
    """
    src, dest, step, cli_params = step1_data
    warp_forward_out, warp_inverse_out = register(src=src, dest=dest, step=step, param=cli_params)

@pytest.mark.skip(reason="Need to fix input test data")
def test_register_step2(step2_data):
    """
    """
    src, dest, step, cli_params = step2_data
    warp_forward_out, warp_inverse_out = register(src=src, dest=dest, step=step, param=cli_params)

@pytest.mark.skip(reason="TODO")
def test_register2d_centermassrot():
    """
    """
    raise NotImplementedError()

@pytest.mark.skip(reason="TODO")
def test_register2d_columnwise():
    """
    """
    raise NotImplementedError()

@pytest.mark.skip(reason="TODO")
def test_register2d():
    """
    """
    raise NotImplementedError()

@pytest.mark.skip(reason="TODO")
def test_register_slicewise():
    """
    """
    raise NotImplementedError()