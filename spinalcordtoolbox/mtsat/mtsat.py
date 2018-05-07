# coding: utf-8
# This is the interface API to compute MTsat
# Code is based on QMRLab: https://github.com/neuropoly/qMRLab
# Author: Julien Cohen-Adad
# Copyright (c) 2018 Polytechnique Montreal <www.neuro.polymtl.ca>
# About the license: see the file LICENSE.TXT

import os
import sct_utils as sct
from msct_image import Image
import numpy as np


def compute_mtsat(nii_mt, nii_pd, nii_t1,
                  tr_mt, tr_pd, tr_t1,
                  fa_mt, fa_pd, fa_t1,
                  nii_b1map=None, verbose=1):
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
    nii_t1map = None  # it would be possible in the future to input T1 map from elsewhere (e.g. MP2RAGE). Note: this
                      # T1map needs to be in s unit.
    b1correctionfactor = 0.4  # empirically defined in https://www.frontiersin.org/articles/10.3389/fnins.2013.00095/full#h3

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
        sct.printv('Compute T1 map...', verbose)
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
        sct.printv('Use input T1 map.', verbose)
        r1map = 1. / nii_t1map.data

    # Compute A
    sct.printv('Compute A...', verbose)
    a = (tr_pd * fa_t1_rad / fa_pd_rad - tr_t1 * fa_pd_rad / fa_t1_rad) * np.true_divide(np.multiply(nii_pd.data, nii_t1.data, dtype=float), tr_pd * fa_t1_rad * nii_t1.data - tr_t1 * fa_pd_rad * nii_pd.data)

    # Compute MTsat
    sct.printv('Compute MTsat...', verbose)
    nii_mtsat = nii_mt.copy()
    nii_mtsat.data = tr_mt * np.multiply((fa_mt_rad * np.true_divide(a, nii_mt.data) - 1), r1map, dtype=float) - (fa_mt_rad ** 2) / 2.
    # sct.printv('nii_mtsat.data[95,89,14]' + str(nii_mtsat.data[95,89,14]), type='info')
    # remove nans and clip unrelistic values
    nii_mtsat.data = np.nan_to_num(nii_mtsat.data)
    ind_unrealistic = np.where(np.abs(nii_mtsat.data) > 1)  # we expect MTsat to be on the order of 0.01
    nii_mtsat.data[ind_unrealistic] = 0
    # convert into percent unit (p.u.)
    nii_mtsat.data *= 100

    # Apply B1 correction to result
    # Weiskopf, N., Suckling, J., Williams, G., Correia, M.M., Inkster, B., Tait, R., Ooi, C., Bullmore, E.T., Lutti, A., 2013. Quantitative multi-parameter mapping of R1, PD(*), MT, and R2(*) at 3T: a multi-center validation. Front. Neurosci. 7, 95.
    if not nii_b1map is None:
        nii_mtsat.data = np.true_divide(nii_mtsat.data * (1 - b1correctionfactor), (1 - b1correctionfactor * nii_b1map.data))

    # set back old seterr settings
    np.seterr(**seterr_old)

    return nii_mtsat, nii_t1map


def compute_mtsat_from_file(fname_mt, fname_pd, fname_t1, tr_mt, tr_pd, tr_t1, fa_mt, fa_pd, fa_t1, fname_b1map=None,
                            fname_mtsat=None, fname_t1map=None, verbose=1):
    """
    Compute MTsat and T1map.
    :param fname_mt:
    :param fname_pd:
    :param fname_t1:
    :param tr_mt:
    :param tr_pd:
    :param tr_t1:
    :param fa_mt:
    :param fa_pd:
    :param fa_t1:
    :param fname_b1map:
    :param fname_mtsat:
    :param fname_t1map:
    :param verbose:
    :return: fname_mtsat: file name for MTsat map
    :return: fname_t1map: file name for T1 map
    """
    # load data
    sct.printv('Load data...', verbose)
    nii_mt = Image(fname_mt)
    nii_pd = Image(fname_pd)
    nii_t1 = Image(fname_t1)
    if fname_b1map is not None:
        nii_b1map = Image(fname_b1map)

    # compute MTsat
    nii_mtsat, nii_t1map = compute_mtsat(nii_mt, nii_pd, nii_t1, tr_mt, tr_pd, tr_t1, fa_mt, fa_pd, fa_t1,
                                         nii_b1map=nii_b1map, verbose=verbose)

    # Output MTsat and T1 maps
    # by default, output in the same directory as the input images
    sct.printv('Generate output files...', verbose)
    if fname_mtsat is None:
        fname_mtsat = os.path.join(nii_mt.path, "mtsat.nii.gz")
    nii_mtsat.setFileName(fname_mtsat)
    nii_mtsat.save()
    if fname_t1map is None:
        fname_t1map = os.path.join(nii_mt.path, "t1map.nii.gz")
    nii_t1map.setFileName(fname_t1map)
    nii_t1map.save()

    return fname_mtsat, fname_t1map
