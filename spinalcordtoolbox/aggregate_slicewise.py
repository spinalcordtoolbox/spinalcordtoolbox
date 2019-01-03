#!/usr/bin/env python
# -*- coding: utf-8
# Functions dealing with metrics aggregation (mean, std, etc.) across slices and/or vertebral levels

from __future__ import absolute_import

import numpy as np

import sct_utils as sct
from spinalcordtoolbox.template import get_slices_from_vertebral_levels
from spinalcordtoolbox.image import Image


def aggregate_per_slice_or_level(metrics, slices=None, levels=None, perslice=True, perlevel=False, vert_level=None,
                                 group_funcs=(('mean', np.mean),)):
    """
    It is assumed that each element of a metric's vector correspond to a slice. E.g., index #2 corresponds to slice #2.
    :param dict metrics: Dict of Class process_seg.Metric() to aggregate.
    :param slices: List[int]: Slices to aggregate metrics from. If empty, select all slices.
    :param levels: List[int]: Vertebral levels to aggregate metrics from
    :param Bool perslice: Aggregate per slice (True) or across slices (False)
    :param Bool perlevel: Aggregate per level (True) or across levels (False)
    :param vert_level: Vertebral level. Could be either an Image or a file name.
    :param tuple group_funcs: Functions to apply on metrics. Example: (('mean', np.mean),))
    :return: Aggregated metrics
    """
    # if slices is empty, select all available slices from the metrics
    if not slices:
        slices = metrics[metrics.keys()[0]].z
    # aggregation based on levels
    if levels:
        im_vert_level = Image(vert_level).change_orientation('RPI')
        # im_vert_level.change_orientation("RPI")  # last dim should be k
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
    agg_metrics = dict((slicegroup, dict()) for slicegroup in slicegroups)
    # loop across slice group
    for slicegroup in slicegroups:
        # add level info
        if levels:
            agg_metrics[slicegroup]['VertLevel'] = vertgroups[slicegroups.index(slicegroup)]
        else:
            agg_metrics[slicegroup]['VertLevel'] = None
        # create dict for each metric
        agg_metrics[slicegroup]['metrics'] = dict((metric, dict()) for metric in metrics)
        for metric in metrics.keys():
            agg_metrics[slicegroup]['metrics'][metric]['label'] = metrics[metric].label
            metric_data = []
            # make sure metric and z have same length
            if not len(metrics[metric].z) == len(metrics[metric].value):
                # TODO: raise custom exception instead of hard-coding error message
                agg_metrics[slicegroup]['metrics'][metric]['error'] = 'metric and z have do not have the same length'
            else:
                for iz in slicegroup:
                    if iz in metrics[metric].z:
                        metric_data.append(metrics[metric].value[metrics[metric].z.index(iz)])
                    else:
                        # sct.log.warning('z={} is not listed in the metric.'.format(iz))
                        agg_metrics[slicegroup]['metrics'][metric]['error'] = 'z={} is not listed in the metric.'.format(iz)
                try:
                    # Loop across functions (typically: mean, std)
                    for (name, func) in group_funcs:
                        # check if nan
                        result = func(metric_data)
                        if np.isnan(result):
                            agg_metrics[slicegroup]['metrics'][metric]['error'] = 'Contains nan'
                        else:
                            agg_metrics[slicegroup]['metrics'][metric][name] = result
                except Exception as e:
                    sct.log.warning(e)
                    # sct.log.warning('TypeError for metric {}'.format(metric))
                    agg_metrics[slicegroup]['metrics'][metric]['error'] = e.message
    return agg_metrics


def make_a_string(item):
    """Convert tuple or list or None to a string. Important: elements in tuple or list are separated with ; (not ,)
    for compatibility with csv."""
    if isinstance(item, tuple) or isinstance(item, list):
        return ';'.join([str(i) for i in item])
    elif item is None:
        return 'None'
    else:
        return item


def save_as_csv(agg_metrics, fname, append=False):
    """
    Write metric structure as csv. If field 'error' exists, it will add a specific column.
    :param metric: output of aggregate_per_slice_or_level()
    :param fname: output filename. Extention (.csv) will be added if it does not exist.
    :param append: Bool: Append results at the end of file (if exists) instead of overwrite.
    :return:
    """
    # TODO: build header based on existing func (e.g., will currently crash if no STD).
    # Create output csv file
    # If appending to existing file, no need to create header. Jump directly to appending of results.
    if append:
        file_results = open(fname, 'a')
    else:
        file_results = open(fname, 'w')
        # build header
        header = (','.join(['Slice (I->S)', 'Vertebral level']))
        for metric in agg_metrics[agg_metrics.keys()[0]]['metrics'].keys():
            header = ','.join([header, 'MEAN({})'.format(metric), 'STD({})'.format(metric)])
        file_results.write(header+'\n')
    # populate data
    for slicegroup in sorted(agg_metrics.keys()):
        line = ','.join([make_a_string(slicegroup),  # list all slices in slicegroup
                         make_a_string(agg_metrics[slicegroup]['VertLevel'])])  # list vertebral levels
        for metric in agg_metrics[slicegroup]['metrics'].keys():
            try:
                line = ','.join([line,
                                 str(agg_metrics[slicegroup]['metrics'][metric]['mean']),
                                 str(agg_metrics[slicegroup]['metrics'][metric]['std'])])
            except KeyError:
                # if mean or std field does not exist, fill value with 'nan'
                line = ','.join([line, 'nan', 'nan'])
        file_results.write(line+'\n')
    file_results.close()
