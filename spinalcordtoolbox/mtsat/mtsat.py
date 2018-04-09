# coding: utf-8
# This is the interface API to compute MTsat
# Code is based on QMRLab: https://github.com/neuropoly/qMRLab
# Author: Julien Cohen-Adad
# Copyright (c) 2018 Polytechnique Montreal <www.neuro.polymtl.ca>
# About the license: see the file LICENSE.TXT

import sct_utils as sct
from msct_image import Image
import numpy as np
import math

def compute_mtsat(nii_mt, nii_pd, nii_t1,
                  tr_mt, tr_pd, tr_t1,
                  fa_mt, fa_pd, fa_t1,
                  nii_b1map=None, verbose=1):
    """
    Compute MTsat and T1 map based on FLASH scans
    Parameters
    ----------
    nii_mt: msct_image object
    nii_pd: msct_image object
    nii_t1: msct_image object
    tr_mt: TR in ms
    tr_pd: TR in ms
    tr_t1: TR in ms
    fa_mt: flip angle in deg
    fa_pd: flip angle in deg
    fa_t1: flip angle in deg
    nii_b1map: msct_image object of B1+ field map, used to correct the flip angle
    verbose: 0, 1
    Returns
    -------
    nii_mtsat: msct_image object
    nii_t1map: msct_image object
    """
    # params
    nii_t1map = None  # it would be possible in the future to input T1 map from elsewhere (e.g. MP2RAGE). Note: this
                      # T1map needs to be in s unit.
    b1correctionfactor = 0.4  # empirically defined in https://www.frontiersin.org/articles/10.3389/fnins.2013.00095/full#h3

    # Load TRs in seconds
    # TR_MT = 0.001*tr_mt
    # TR_PD = 0.001*tr_pd
    # TR_T1 = 0.001*tr_t1

    # Convert flip angles into radians
    fa_mt_rad = math.radians(fa_mt)
    fa_pd_rad = math.radians(fa_pd)
    fa_t1_rad = math.radians(fa_t1)

    # check if a T1 map was given in input; if not, compute R1
    if nii_t1map is None:
        # compute R1
        sct.printv('Compute T1 map...', verbose)
        r1map = 0.5 * np.divide((fa_t1_rad / tr_t1) * nii_t1.data - (fa_pd_rad / tr_pd) * nii_pd.data,
                                  nii_pd.data / fa_pd_rad - nii_t1.data / fa_t1_rad)
        # Convert R1 to s^-1
        r1map = r1map * 1000
        # compute T1
        nii_t1map = nii_mt.copy()
        nii_t1map.data = 1. / r1map
    else:
        sct.printv('Use input T1 map.', verbose)
        r1map = 1. / nii_t1map.data

    # Compute A
    sct.printv('Compute A...', verbose, 'info')
    a = (tr_pd * fa_t1_rad / fa_pd_rad - tr_t1 * fa_pd_rad / fa_t1_rad) * np.divide(np.multiply(nii_pd.data, nii_t1.data), tr_pd * fa_t1_rad * nii_t1.data - tr_t1 * fa_pd_rad * nii_pd.data)

    # Compute MTsat
    sct.printv('Compute MTsat...', verbose, 'info')
    nii_mtsat = nii_mt.copy()
    nii_mtsat.data = tr_mt * np.multiply((fa_mt_rad * np.divide(a, nii_mt.data) - 1), r1map) - (fa_mt_rad ** 2) / 2.

    # Apply B1 correction to result
    # Weiskopf, N., Suckling, J., Williams, G., Correia, M.M., Inkster, B., Tait, R., Ooi, C., Bullmore, E.T., Lutti, A., 2013. Quantitative multi-parameter mapping of R1, PD(*), MT, and R2(*) at 3T: a multi-center validation. Front. Neurosci. 7, 95.
    if not nii_b1map is None:
        nii_mtsat.data = np.divide(nii_mtsat.data * (1 - b1correctionfactor), (1 - b1correctionfactor * nii_b1map.data))

    return nii_t1map, nii_mtsat



def compute_mtsat_from_file(fname_mt, fname_pd, fname_t1, tr_mt, tr_pd, tr_t1, fa_mt, fa_pd, fa_t1, fname_b1map=None,
                            fname_mtsat=None, fname_t1map=None, verbose=1):
    """
    Compute MTsat and T1map.
    Parameters
    ----------
    fname_mt
    fname_pd
    fname_t1
    tr_mt
    tr_pd
    tr_t1
    fa_mt
    fa_pd
    fa_t1
    fname_b1map
    fname_mtsat
    fname_t1map
    verbose

    Returns
    -------
    fname_mtsat: file name for MTsat map
    fname_t1map: file name for T1 map
    """

    # load data
    sct.printv('Load data...', verbose)
    nii_mt = Image(fname_mt)
    nii_pd = Image(fname_pd)
    nii_t1 = Image(fname_t1)
    if fname_b1map is not None:
        nii_b1map = Image(fname_b1map)

    # compute MTsat
    nii_t1map, nii_mtsat = compute_mtsat(nii_mt, nii_pd, nii_t1, tr_mt, tr_pd, tr_t1, fa_mt, fa_pd, fa_t1,
                                         nii_b1map=None, verbose=verbose)

    # Output MTsat and T1 maps
    # by default, output in the same directory as the input images
    # if fname_mtsat is None:

    # MTsat.setFileName(outputs_fname[0])
    # MTsat.save()

    # save MTR file
    # nii_mtr = nii_mt1
    # nii_mtr.data = data_mtr
    # nii_mtr.setFileName('mtr.nii')
    # nii_mtr.save()
    # sct.run(fsloutput+'fslmaths -dt double mt0.nii -sub mt1.nii -mul 100 -div mt0.nii -thr 0 -uthr 100 mtr.nii', verbose)

    # Generate output files
    # sct.printv('\nGenerate output files...', verbose)
    # sct.generate_output_file(os.path.join(path_tmp, "mtr.nii"), os.path.join(path_out, file_out + ext_out))

    return fname_mtsat, fname_t1map
