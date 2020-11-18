#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.register

import logging

import pytest
import csv

# FIXME should not use stuff from scripts. Ok for now
from sct_register_to_template import Param, register

from spinalcordtoolbox.registration.register import *
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
    """
    """
    src, dest, step, cli_params = step_axial_data_in_same_space

    warp_forward_out, warp_inverse_out = register_step_ants_slice_regularized_registration(
        src=src,
        dest=dest,
        step=step,
        metricSize='4')

    # Verify integrity of the output Tx Ty file.
    with open('step1TxTy_poly.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        txty_result = []
        for row in spamreader:
            txty_result.append(row)
    assert txty_result == [
        ['Tx', 'Ty'],
        ['1.00439255454324', '-0.151760183772956'],
        ['1.04645709177172', '-0.621178997815087'],
        ['0.904949339396005', '-0.862157494292993'],
        ['0.784728781515325', '-0.893095039021259'],
        ['0.890654902228898', '-0.732390997814474']]


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