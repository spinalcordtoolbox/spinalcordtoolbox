"""
Functions to interpolate metrics (from sct_process_segmentation) into the PAM50 anatomical dimensions

Copyright (c) 2023 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import numpy as np
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.aggregate_slicewise import Metric
from spinalcordtoolbox.template import get_slices_from_vertebral_levels, get_vertebral_level_from_slice


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

    # Compute the mean scaling factor between PAM50 and native slices
    pairs = list(zip(level_slices_PAM50, level_slices_im))
    # Exclude the first/last levels to avoid edge effects (only if there are enough levels)
    trim = 1 if len(pairs) > 2 else 0
    scale_mean = np.mean([len(s_pam) / len(s_im) for s_pam, s_im in pairs[trim:len(pairs)-trim]])

    # Initialize a metrics dict filled by NaN with number of rows equal to number of slices in PAM50 template
    z = im_seg_labeled_PAM50.dim[2]  # z == number of slices
    metrics_PAM50_space_dict = {k: np.full([z], np.nan) for k in metrics.keys()}
    # Loop through slices per-level (excluding first and last levels), populating the metrics dict
    for level, slices_PAM50, slices_im in zip(levels, level_slices_PAM50, level_slices_im):
        # Prepare vectors for the interpolation
        if level in [levels[0], levels[-1]]:
            # Note: since the first/last levels can be incomplete, we use the mean scaling factor from all other levels
            x_PAM50 = np.linspace(0, scale_mean * len(slices_im), int(scale_mean * len(slices_im)))
            x = np.linspace(0, scale_mean * len(slices_im), len(slices_im))
        else:
            x_PAM50 = np.arange(0, len(slices_PAM50), 1)
            x = np.linspace(0, len(slices_PAM50) - 1, len(slices_im))
        # Loop through metrics
        for key, value in metrics.items():
            if key != 'length':
                metric_values_level = value.data[slices_im]
                # Interpolate in the same number of slices
                metrics_inter = np.interp(x_PAM50, x, metric_values_level)
                # Scale interpolation of first and last levels (to account for incomplete levels)
                diff = len(metrics_inter) - len(slices_PAM50)
                if level == levels[0]:
                    # If the first level, scale from level below
                    if diff > 0:
                        metrics_inter = metrics_inter[:-diff]
                    elif diff < 0:
                        slices_PAM50 = slices_PAM50[:-abs(diff)]
                elif level == levels[-1]:
                    # If the last level, scale from level above
                    if diff > 0:
                        metrics_inter = metrics_inter[diff:]
                    elif diff < 0:
                        slices_PAM50 = slices_PAM50[abs(diff):]
                metrics_PAM50_space_dict[key][slices_PAM50] = metrics_inter

    # Convert dict of ndarrays to dict of Metric() objects
    return {k: Metric(data=np.array(v), label=k) for k, v in metrics_PAM50_space_dict.items()}


def build_pam50_agg_metric(agg_metric_native, nz_native, label_name, method,
                           fname_vert_level, fname_vert_level_PAM50):
    """
    Adapt `interpolate_metrics()` to the per-slice output of `extract_metric()`.
    :param agg_metric_native: per-slice native-space metrics (dict output of extract_metric() with perslice=True)
    :param nz_native: int: total z-slices in native image
    :param label_name: str: atlas label name (for 'Label' CSV column), e.g., 'white matter'
    :param method: str: extraction method ('wa', 'ml', 'map', 'bin', 'median', 'max')
    :param fname_vert_level: str: native vertebral levels file (centerline-masked)
    :param fname_vert_level_PAM50: str: PAM50 template PAM50_levels.nii.gz
    :return: dict keyed by (z,) PAM50 slice tuples, suitable for save_as_csv()
    """
    method_key_map = {
        'wa': 'WA()', 'ml': 'ML()', 'map': 'MAP()',
        'bin': 'BIN()', 'median': 'MEDIAN()', 'max': 'MAX()'
    }
    primary_key = method_key_map[method]

    # Convert metrics from extract_metric() form (one dict per slice, multiple metrics) to
    # compute_shape() form (one Metric object per metric, multiple slices), since that is the
    # form expected by interpolate_metrics()
    metric_1d = np.full(nz_native, np.nan)
    for (z,), entry in agg_metric_native.items():
        val = entry.get(primary_key)
        if val is not None:
            metric_1d[z] = val

    # Interpolate to PAM50 space; returns Dict[str, Metric] with 1D data of length z_PAM50
    metrics_pam50 = interpolate_metrics(
        {primary_key: Metric(data=metric_1d, label=primary_key)},
        fname_vert_level_PAM50,
        fname_vert_level
    )

    # Convert interpolated metrics back into the expected form, from one Metric object per metric
    # back into one dict per slice, since that is the form expected by save_as_csv()
    pam50_values = metrics_pam50[primary_key].data

    # Determine which vertebral levels are present in the native data to filter PAM50 output
    # (excluding 0 and values >=49, which are reserved/non-vertebral-level labels in the PAM50
    # convention: https://spinalcordtoolbox.com/stable/user_section/tutorials/vertebral-labeling/labeling-conventions.html)
    im_native_levels = Image(fname_vert_level).change_orientation('RPI')
    native_levels = set(
        int(v) for v in np.unique(im_native_levels.data) if 0 < int(v) < 49
    )

    # Map each PAM50 z-slice to a vertebral level and build the output agg_metric
    im_pam50_levels = Image(fname_vert_level_PAM50).change_orientation('RPI')

    agg_metric_pam50 = {}
    for z_pam50, val in enumerate(pam50_values):
        # nan means that there was no data in the native space for that slice, so we skip it in the PAM50 space as well
        if np.isnan(val):
            continue
        vert_level = get_vertebral_level_from_slice(im_pam50_levels, z_pam50)
        if vert_level is None or vert_level not in native_levels:
            continue
        entry = {
            'Label': label_name,
            'VertLevel': (vert_level,),
            'DistancePMJ': None,    # required by save_as_csv() but not relevant for PAM50 space
            primary_key: val,
        }
        agg_metric_pam50[(z_pam50,)] = entry

    return agg_metric_pam50
