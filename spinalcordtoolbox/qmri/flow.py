"""
This is the interface API to compute flow-related metrics

Copyright (c) 2023 Polytechnique Montreal <www.neuro.polymtl.ca>
Author: Julien Cohen-Adad & ChatGPTv4
License: see the file LICENSE
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def calculate_velocity(data_phase, venc):
    """
    Convert phase data to velocity. Assumes phase data is scaled between -pi and pi.
    :param data_phase: 4D numpy array of phase data. The 4th dimension should be the velocity encoding (VENC) in cm/s.
    :param venc: Maximum velocity that can be encoded in cm/s.
    :return: velocity data
    """

    # Check that phase data is scaled between -pi and pi
    if np.any(data_phase > np.pi) or np.any(data_phase < -np.pi):
        logger.warning('Phase data is not scaled between -pi and pi. Please check your data.')

    # Phase to velocity conversion
    velocity = np.true_divide(data_phase * venc, np.pi)
    return velocity


def scale_phase(data_phase):
    """
    Scale phase data between -π and π. We assume the data was originally encoded as int12 [0, 4095] (as this seems
    to be the default encoding for phase data from Siemens scanners), then scaled to [-4096, 4094] during conversion
    from DICOM to Nifti due to a slope/intercept of 2.0 and -4096.

    See also: https://github.com/rordenlab/dcm2niix/issues/406#issuecomment-654141193

    :param data_phase: numpy array of phase data
    :return: scaled phase data from
    """
    # Check that phase data is encoded as scaled 12-bit integers
    slope = 2.0
    intercept = -4096
    bounds_unscaled = [0, 4095]                                       # 4096 -> 2^12 -> 12-bit data (Siemens DICOM)
    bounds_scaled = [b * slope + intercept for b in bounds_unscaled]  # [-4096, 4094] range for NIfTI phase data
    if np.any(data_phase < bounds_scaled[0]) or np.any(data_phase > bounds_scaled[1]):
        logger.warning(f'Phase data is not within the expected range for Siemens phase data (i.e., between '
                       f'{bounds_scaled[0]} and {bounds_scaled[1]}). Please check your data.')
    data_phase = data_phase.astype(np.float32)
    # Scale the data from [-4096, 4094] to [-π, π]
    scaled_data = (data_phase - intercept) / slope      # [-4096, 4094] -> [0, 4095]
    scaled_data = scaled_data / bounds_unscaled[1] * 2  # [0, 4095] -> [0, 1] -> [0, 2]
    scaled_data = (scaled_data - 1) * np.pi             # [0, 2] -> [-1, 1] -> [-π, π]
    return scaled_data
