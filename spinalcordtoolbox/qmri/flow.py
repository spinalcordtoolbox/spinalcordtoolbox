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
    :param venc: Maximum velocity that can be encoded
    :return: velocity data
    """
    # Phase to velocity conversion
    velocity = np.true_divide(data_phase * venc, np.pi)
    return velocity

# def calculate_flow(velocity_data, pixel_area):
#     """
#     Calculate flow by integrating velocity over a given area.
#     :param velocity_data: numpy array of velocity data
#     :param pixel_area: area of each pixel in the image (in square meters)
#     :return: flow rate in the vessel (in cubic meters per second)
#     """
#     flow_rate = np.sum(velocity_data) * pixel_area
#     return flow_rate


def compute_flow(nii_phase4d, venc):
    """
    Compute velocity from VENC technique. More details at: https://mriquestions.com/what-is-venc.html
    :param nii_phase4d: 4D phase image. The 4th dimension should be the velocity encoding (VENC) in cm/s.
    :param venc: float: Maximum velocity that can be encoded
    :return: nii_flow
    """
    # Convert input to avoid numerical errors from int16 data
    # Related issue: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3636
    nii_phase4d.change_type('float32')

    # Initialize Image object
    nii_flow = nii_phase4d.copy()

    # Compute flow
    nii_flow.data = np.true_divide(nii_phase4d.data, 2 * np.pi * venc)

    return nii_flow
