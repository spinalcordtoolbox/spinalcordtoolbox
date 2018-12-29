#!/usr/bin/env python
# -*- coding: utf-8
# Functions dealing with metrics aggregation (mean, std, etc.) across slices and/or vertebral levels

from __future__ import absolute_import

import numpy as np

import sct_utils as sct
from spinalcordtoolbox.template import get_slices_from_vertebral_levels


def aggregate_per_slice_or_level(metrics, slices=None, levels=None, perslice=True, perlevel=False, im_vert_level=None,
                                 group_funcs=(('mean', np.mean),)):
    """
    It is assumed that each element of a metric's vector correspond to a slice. E.g., index #2 corresponds to slice #2.
    :param dict metrics: Dict of Class process_seg.Metric() to aggregate.
    :param slices: List[int]: Slices to aggregate metrics from
    :param levels: List[int]: Vertebral levels to aggregate metrics from
    :param Bool perslice: Aggregate per slice (True) or across slices (False)
    :param Bool perlevel: Aggregate per level (True) or across levels (False)
    :param Image im_vert_level: Image of vertebral level
    :param tuple group_funcs: Functions to apply on metrics. Example: (('mean', np.mean),))
    :return: Aggregated metrics
    """
    # TODO: move the parse_num_list to the front-end
    # Create a dictionary for the output aggregated metrics
    agg_metrics = dict((metric, dict()) for metric in metrics.keys())
    # aggregation based on levels
    if levels:
        im_vert_level.change_orientation("RPI")  # last dim should be k
        slicegroups = [tuple(get_slices_from_vertebral_levels(im_vert_level, level)) for level in levels]
        if perlevel:
            # slicegroups = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
            # vertgroups = [(2,), (3,), (4,)]
            vertgroups = [tuple([level]) for level in levels]
        # output aggregate metrics across levels
        else:
            # slicegroups = [(0, 1, 2, 3, 4, 5, 6, 7, 8)]
            slicegroups = [tuple([val for sublist in slicegroups for val in sublist])]  # flatten list_slices
            # vertgroups = [(2, 3, 4)]
            vertgroups = [tuple([level for level in levels])]
    # aggregation based on slices
    else:
        vertgroups = None
        if perslice:
            # slicegroups = [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,)]
            slicegroups = [tuple([slice]) for slice in slices]
        else:
            # slicegroups = [(0, 1, 2, 3, 4, 5, 6, 7, 8)]
            slicegroups = [tuple(slices)]
    # loop across slice group
    for slicegroup in slicegroups:
        for metric in metrics.keys():
            metric_data = []
            # make sure metric and z have same length
            if not len(metrics['z'].value) == len(metrics[metric].value):
                # TODO: raise custom exception instead of hard-coding error message
                agg_metrics[metric][slicegroup] = {'error': 'metric and z have do not have the same length'}
            else:
                for iz in slicegroup:
                    if iz in metrics['z'].value:
                        metric_data.append(metrics[metric].value[metrics['z'].value.index(iz)])
                    else:
                        sct.log.warning('z={} is not listed in the metric.'.format(iz))
                try:
                    agg_metrics[metric][slicegroup] = dict((name, func(metric_data)) for (name, func) in group_funcs)
                except Exception as e:
                    sct.log.warning(e)
                    # sct.log.warning('TypeError for metric {}'.format(metric))
                    agg_metrics[metric][slicegroup] = {'error': e.message}
                # add level info
                if levels:
                    agg_metrics[metric][slicegroup]["VertLevel"] = vertgroups[slicegroups.index(slicegroup)]
                else:
                    agg_metrics[metric][slicegroup]["VertLevel"] = None
    return agg_metrics
