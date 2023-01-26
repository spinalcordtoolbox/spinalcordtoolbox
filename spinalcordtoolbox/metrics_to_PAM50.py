# Functions to interpolate metrics (from sct_process_segmentation) into the PAM50 anatomical dimensions
# Author: Sandrine Bédard & Jan Valosek

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

    # Get number of slices in PAM50
    z = im_seg_labeled_PAM50.dim[2]

    # Create an metrics instance filled by NaN with number of rows equal to number of slices in PAM50 template
    metrics_PAM50_space_dict = {}
    for key in metrics.keys():
        metrics_PAM50_space_dict[key] = np.empty(z, dtype=float)
        metrics_PAM50_space_dict[key].fill(np.nan)

    # Get unique vertebral levels
    levels = np.unique(im_seg_labeled.data)
    # Remove zero and convert levels to int
    levels = list(map(int, (levels[levels > 0])))

    # Remove level 49 and 50 (not vertebral levels)
    levels = [level for level in levels if level < 49]

    # Sort levels (so min == levels[0] and max == levels[-1])
    levels = sorted(levels)

    # Get slices corresponding to each level
    level_slices_PAM50 = [get_slices_from_vertebral_levels(im_seg_labeled_PAM50, level) for level in levels]
    level_slices_im = [get_slices_from_vertebral_levels(im_seg_labeled, level) for level in levels]

    # Create empty list to keep the scaling between the image and PAM50
    scales = []
    # Loop through slices per-level (excluding first and last levels)
    for slices_PAM50, slices_im in zip(level_slices_PAM50[1:-1], level_slices_im[1:-1]):
        # Prepare vectors for the interpolation
        x_PAM50 = np.arange(0, len(slices_PAM50), 1)
        x = np.linspace(0, len(slices_PAM50) - 1, len(slices_im))
        # Compute and keep the scaling factor for the currently processed level
        scales.append(len(slices_PAM50)/len(slices_im))
        # Loop through metrics
        for key, value in metrics.items():
            if key != 'length':
                metric_values_level = value.data[slices_im]
                # Interpolate in the same number of slices
                metrics_PAM50_space_dict[key][slices_PAM50] = np.interp(x_PAM50, x, metric_values_level)
    scale_mean = np.mean(scales)

    # Loop through the slices in the first and last levels to scale only.
    for i, (slices_PAM50, slices_im) in enumerate(zip(level_slices_PAM50[::len(levels)-1],
                                                      level_slices_im[::len(levels)-1])):
        # Prepare vectors for the interpolation
        # Note: since the first and the last level can be incomplete, we use the mean scaling factor from all other levels
        x_PAM50 = np.linspace(0, scale_mean*len(slices_im), int(scale_mean*len(slices_im)))
        x = np.linspace(0, scale_mean*len(slices_im), len(slices_im))
        # Loop through metrics
        for key, value in metrics.items():
            if key != 'length':
                metric_values_level = value.data[slices_im]
                # Interpolate in the same number of slices
                metrics_inter = np.interp(x_PAM50, x, metric_values_level)
                # If the first level, scale from level below
                if i == 0:
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
                metrics_PAM50_space_dict[key][slices_PAM50] = metrics_inter

    # Create a dict of Metric() objects
    metrics_PAM50_space = {}
    # Loop through metrics
    for key, value in metrics_PAM50_space_dict.items():
        # Convert ndarray to Metric() object
        metrics_PAM50_space[key] = Metric(data=np.array(value), label=key)

    return metrics_PAM50_space
