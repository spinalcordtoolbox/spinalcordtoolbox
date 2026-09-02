"""
Functions to interpolate metrics (from sct_process_segmentation) into the PAM50 anatomical dimensions

Copyright (c) 2023 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import numpy as np
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.aggregate_slicewise import Metric
from spinalcordtoolbox.template import get_slices_from_vertebral_levels


def interpolate_metrics(metrics, fname_vert_levels_PAM50, fname_vert_levels):
    """
    Interpolates metrics perlevel into the PAM50 anatomical dimensions.
    :param metrics: Dict of class Metric(). Output of spinalcordtoolbox.process_seg.compute_shape.
    :param fname_vert_levels_PAM50: Path to the PAM50_levels.nii.gz (PAM50 labeled segmentation).
    :param fname_vert_levels: Path to subject's vertebral labeling file.
    :return metrics_PAM50_space: Dict of class Metric() in PAM50 anatomical dimensions.
    """
    # Load PAM50 labeled segmentation
    im_seg_labeled_PAM50 = Image(fname_vert_levels_PAM50)
    im_seg_labeled_PAM50.change_orientation('RPI')
    # Load subject's labeled segmentation
    im_seg_labeled = Image(fname_vert_levels)
    im_seg_labeled.change_orientation('RPI')

    # Get unique integer vertebral levels (but exclude 0, 49, and 50, as these aren't vertebral levels)
    levels = sorted(int(level) for level in np.unique(im_seg_labeled.data) if 0 < int(level) < 49)

    # Get slices corresponding to each level
    level_slices_PAM50 = [get_slices_from_vertebral_levels(im_seg_labeled_PAM50, level) for level in levels]
    level_slices_im = [get_slices_from_vertebral_levels(im_seg_labeled, level) for level in levels]

    # Find the mean scaling between the image and PAM50 (excluding first and last levels)
    scales = [len(slices_PAM50)/len(slices_im) for slices_PAM50, slices_im
              in zip(level_slices_PAM50[1:-1], level_slices_im[1:-1])]
    scale_mean = np.mean(scales)

    # Initialize a metrics dict filled by NaN with number of rows equal to number of slices in PAM50 template
    z = im_seg_labeled_PAM50.dim[2]  # z == number of slices
    metrics_PAM50_space_dict = {k: np.full([z], np.nan) for k in metrics.keys()}

    # Loop through slices per-level (excluding first and last levels), populating the metrics dict
    for i, (level, slices_PAM50, slices_im) in enumerate(zip(levels, level_slices_PAM50, level_slices_im)):
        is_first = (i == 0)
        is_last  = (i == len(levels) - 1)

        # Set up the necessary input parameters for the interpolation function `np.interp`: x, xp, and fp.
        #    - (xp, fp) are a set of known input->output pairs (e.g. slice1->csa1, slice2->csa2, etc.)
        #    - (x) is a set of inputs we want (e.g. slices in the PAM50 space)
        #    - However, since our goal is just to go from one linearly-spaced grid to another, all we need to know is:
        #       * A. How many points are in the level in the subject space, and
        #       * B. How many points are in the level in the PAM50 space.
        n_subj  = len(slices_im)
        n_pam50 = len(slices_PAM50)
        # However, there is a caveat:
        #    - For the first/last levels in the subject space, the cord seg could be cut off (i.e. partial levels)
        #    - So, instead of using all the points from the corresponding PAM50 level, we estimate how many points the
        #      "partial" PAM50 level would have by using the mean ratio of subj:PAM50 points and multiplying.
        if is_first or is_last:
            n_pam50 = int(scale_mean * n_subj)
        # Finally, we linearly space N points across a shared range to create both grids
        x  = np.linspace(start=0, stop=1, num=n_pam50)
        xp = np.linspace(start=0, stop=1, num=n_subj)

        # Loop through metrics
        for key, value in metrics.items():
            if key != 'length':
                metric_values_level = value.data[slices_im]
                metrics_inter = np.interp(x=x, xp=xp, fp=metric_values_level)
                # Scale interpolation of first and last levels (to account for incomplete levels)
                diff = len(metrics_inter) - len(slices_PAM50)
                if is_first:
                    # If the first level, scale from level below
                    if diff > 0:
                        metrics_inter = metrics_inter[:-diff]
                    elif diff < 0:
                        slices_PAM50 = slices_PAM50[:-abs(diff)]
                elif is_last:
                    # If the last level, scale from level above
                    if diff > 0:
                        metrics_inter = metrics_inter[diff:]
                    elif diff < 0:
                        slices_PAM50 = slices_PAM50[abs(diff):]
                metrics_PAM50_space_dict[key][slices_PAM50] = metrics_inter

    # Convert dict of ndarrays to dict of Metric() objects
    return {k: Metric(data=np.array(v), label=k) for k, v in metrics_PAM50_space_dict.items()}
