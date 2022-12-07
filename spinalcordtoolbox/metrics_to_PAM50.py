# Functions to interpolate metrics (from sct_process_segmentation) into the PAM50 anatomical dimensions
# Author: Sandrine Bédard & Jan Valosek

import numpy as np
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.aggregate_slicewise import Metric
from spinalcordtoolbox.template import get_slices_from_vertebral_levels


def get_first_and_last_levels(levels):
    """
    Gets first and last levels.
    :param levels: list: list of all available levels.
    :return min(levels): int: first level.
    :return max(levels): int: last level.
    """
    return min(levels), max(levels)


def interpolate_metrics(metrics, fname_vert_levels_PAM50, fname_vert_levels):
    """
    Interpolates metrics perlevel into the PAM50 anatomical dimensions.
    :param metrics: Dict of class Metric(). Output of spinalcordtoolbox.process_seg.compute_shape.
    :param fname_vert_levels_PAM50: Path to the PAM50_levels.nii.gz (PAM50 labeled segmentation).
    :param fname_vert_levels: Path to subject's vertebral labeling file.
    :return metrics_PAM50_agg: Dict of class Metric() in PAM50 anatomical dimensions.
    """
    # Load PAM50 labeled segmentation
    im_seg_labeled_PAM50 = Image(fname_vert_levels_PAM50)
    # Load subject's labeled segmentation
    im_seg_labeled = Image(fname_vert_levels)

    # Get number of slices in PAM50
    z = im_seg_labeled_PAM50.dim[2]

    # Create an metrics instance filled by NaN with number of rows equal to number of slices in PAM50 template
    metrics_PAM50_space = {}
    for key in metrics.keys():
        metrics_PAM50_space[key] = np.empty(z, dtype=float)
        metrics_PAM50_space[key].fill(np.nan)

    # Get unique vertebral levels
    levels = np.unique(im_seg_labeled.data)
    # Remove zero and convert levels to int
    levels = list(map(int, (levels[levels > 0])))
    
    # Get first and last level to only scale: if levels are not complete
    levels_2_skip = get_first_and_last_levels(levels)

    # Create empty list to keep the scaling between the image and PAM50
    scales = []  
    # Loop through levels
    for level in levels:
        if level not in levels_2_skip:
            # Get slices corresponding for the currently processed level
            slices_PAM50 = get_slices_from_vertebral_levels(im_seg_labeled_PAM50, level)
            slices_im = get_slices_from_vertebral_levels(im_seg_labeled, level)
            nb_slices = len(slices_PAM50)
            x_PAM50 = np.arange(0, nb_slices, 1)
            x = np.linspace(0, nb_slices - 1, len(slices_im))
            scales.append(nb_slices/len(slices_im))
            # Loop through metrics
            for key, value in metrics.items():
                metric_values_level = value.data[slices_im]
                # Interpolate in the same number of slices
                metrics_PAM50_space[key][slices_PAM50] = np.interp(x_PAM50, x, metric_values_level)
    scale_mean = np.mean(scales)
    
    # Loop through the first and the last level to scale only.
    for level in levels_2_skip:
        # Get slices corresponding for the currently processed level
        slices_PAM50 = get_slices_from_vertebral_levels(im_seg_labeled_PAM50, level)
        slices_im = get_slices_from_vertebral_levels(im_seg_labeled, level)
        # Get number of slices for the currently processed level
        nb_slices_im = len(slices_im)
        # Prepare vectors for the interpolation
        # Note: since the first and the last level can be incomplete, we use the mean scaling factor from all other levels
        x_PAM50 = np.linspace(0, scale_mean*nb_slices_im, int(scale_mean*nb_slices_im))
        x = np.linspace(0, scale_mean*nb_slices_im, nb_slices_im)
        # Loop through metrics
        for key, value in metrics.items():
            metric_values_level = value.data[slices_im]
            metrics_inter = np.interp(x_PAM50, x, metric_values_level)
            # If the first level, scale from level below
            if level == min(levels_2_skip):
                if len(metrics_inter) > len(slices_PAM50):
                    diff = len(metrics_inter) - len(slices_PAM50)
                    metrics_inter = metrics_inter[:-diff]
                elif len(metrics_inter) < len(slices_PAM50):
                    diff = len(slices_PAM50) - len(metrics_inter)
                    slices_PAM50 = slices_PAM50[:-diff]
            # If the last level, scale from level above
            else:
                if len(metrics_inter) > len(slices_PAM50):
                    diff = len(metrics_inter) - len(slices_PAM50)
                    metrics_inter = metrics_inter[diff:]
                elif len(metrics_inter) < len(slices_PAM50):
                    diff = len(slices_PAM50) - len(metrics_inter)
                    slices_PAM50 = slices_PAM50[diff:] 

            metrics_PAM50_space[key][slices_PAM50] = metrics_inter

    # Create a dict of Metric() objects
    metrics_PAM50_agg = {}
    for key in metrics.keys():
        metrics_PAM50_agg[key] = Metric(data=np.array(metrics_PAM50_space[key]), label=key)

    return metrics_PAM50_agg
    # linear interpolation


