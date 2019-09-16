# coding: utf-8
# This is the interface API to compute MTsat
# Code is based on QMRLab: https://github.com/neuropoly/qMRLab
# Author: Julien Cohen-Adad
# Copyright (c) 2018 Polytechnique Montreal <www.neuro.polymtl.ca>
# About the license: see the file LICENSE.TXT

from __future__ import absolute_import, division

import logging
import numpy as np

logger = logging.getLogger(__name__)


def compute_mtr(nii_mt1, nii_mt0):
    """
    Compute Magnetization Transfer Ratio in percentage.
    :param nii_mt1:
    :param nii_mt0:
    :return: nii_mtr
    """
    # Initialize Image object
    nii_mtr = nii_mt1.copy()
    nii_mtr.data = 100 * np.true_divide((nii_mt0.data - nii_mt1.data), nii_mt0.data)
    return nii_mtr


def compute_mtsat(nii_mt, nii_pd, nii_t1,
                  tr_mt, tr_pd, tr_t1,
                  fa_mt, fa_pd, fa_t1,
                  nii_b1map=None):
    """
    Compute MTsat and T1 map based on FLASH scans
    :param nii_mt:
    :param nii_pd:
    :param nii_t1:
    :param tr_mt:
    :param tr_pd:
    :param tr_t1:
    :param fa_mt:
    :param fa_pd:
    :param fa_t1:
    :param nii_b1map:
    :param verbose:
    :return:
    """
    # params
    nii_t1map = \
        None  # it would be possible in the future to input T1 map from elsewhere (e.g. MP2RAGE). Note: this T1map
    # needs to be in s unit.
    b1correctionfactor = \
        0.4  # empirically defined in https://www.frontiersin.org/articles/10.3389/fnins.2013.00095/full#h3

    # convert all TRs in s
    tr_mt *= 0.001
    tr_pd *= 0.001
    tr_t1 *= 0.001

    # Convert flip angles into radians
    fa_mt_rad = np.radians(fa_mt)
    fa_pd_rad = np.radians(fa_pd)
    fa_t1_rad = np.radians(fa_t1)

    # ignore warnings from division by zeros (will deal with that later)
    seterr_old = np.seterr(over='ignore', divide='ignore', invalid='ignore')

    # check if a T1 map was given in input; if not, compute R1
    if nii_t1map is None:
        # compute R1
        logger.info("Compute T1 map...")
        r1map = 0.5 * np.true_divide((fa_t1_rad / tr_t1) * nii_t1.data - (fa_pd_rad / tr_pd) * nii_pd.data,
                                     nii_pd.data / fa_pd_rad - nii_t1.data / fa_t1_rad)
        # remove nans and clip unrelistic values
        r1map = np.nan_to_num(r1map)
        ind_unrealistic = np.where(r1map < 0.01)  # R1=0.01 s^-1 corresponds to T1=100s which is reasonable to clip
        r1map[ind_unrealistic] = np.inf  # set to infinity so that these values will be 0 on the T1map
        # compute T1
        nii_t1map = nii_mt.copy()
        nii_t1map.data = 1. / r1map
    else:
        logger.info("Use input T1 map.")
        r1map = 1. / nii_t1map.data

    # Compute A
    logger.info("Compute A...")
    a = (tr_pd * fa_t1_rad / fa_pd_rad - tr_t1 * fa_pd_rad / fa_t1_rad) * \
        np.true_divide(np.multiply(nii_pd.data, nii_t1.data, dtype=float),
                       tr_pd * fa_t1_rad * nii_t1.data - tr_t1 * fa_pd_rad * nii_pd.data)

    # Compute MTsat
    logger.info("Compute MTsat...")
    nii_mtsat = nii_mt.copy()
    nii_mtsat.data = tr_mt * np.multiply((fa_mt_rad * np.true_divide(a, nii_mt.data) - 1),
                                         r1map, dtype=float) - (fa_mt_rad ** 2) / 2.
    # sct.printv('nii_mtsat.data[95,89,14]' + str(nii_mtsat.data[95,89,14]), type='info')
    # remove nans and clip unrelistic values
    nii_mtsat.data = np.nan_to_num(nii_mtsat.data)
    ind_unrealistic = np.where(np.abs(nii_mtsat.data) > 1)  # we expect MTsat to be on the order of 0.01
    nii_mtsat.data[ind_unrealistic] = 0
    # convert into percent unit (p.u.)
    nii_mtsat.data *= 100

    # Apply B1 correction to result
    # Weiskopf, N., Suckling, J., Williams, G., Correia, M.M., Inkster, B., Tait, R., Ooi, C., Bullmore, E.T., Lutti,
    # A., 2013. Quantitative multi-parameter mapping of R1, PD(*), MT, and R2(*) at 3T: a multi-center validation.
    # Front. Neurosci. 7, 95.
    if nii_b1map is not None:
        nii_mtsat.data = np.true_divide(nii_mtsat.data * (1 - b1correctionfactor),
                                        (1 - b1correctionfactor * nii_b1map.data))

    # set back old seterr settings
    np.seterr(**seterr_old)

    return nii_mtsat, nii_t1map

