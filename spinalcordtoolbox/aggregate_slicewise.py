#!/usr/bin/env python
# -*- coding: utf-8
# Functions dealing with metrics aggregation (mean, std, etc.) across slices and/or vertebral levels


from __future__ import absolute_import

import numpy as np
import operator
import csv

import sct_utils as sct
from spinalcordtoolbox.template import get_slices_from_vertebral_levels, get_vertebral_level_from_slice
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils import parse_num_list_inv


# TODO: don't make metrics a dict anymore-- it complicates things.
# TODO: generalize this function to accept n-dim np.array instead of list in Metric().value
# TODO: maybe no need to bring Metric() class here. Just np.array, then labeling is done in parent function.
def aggregate_per_slice_or_level(metrics, mask=None, slices=None, levels=None, perslice=True, perlevel=False, vert_level=None,
                                 group_funcs=(('mean', np.mean),)):
    """
    It is assumed that each element of a metric's vector correspond to a slice. E.g., index #2 corresponds to slice #2.
    :param dict metrics: Dict of Class process_seg.Metric() to aggregate.
    :param slices: List[int]: Slices to aggregate metrics from. If empty, select all slices.
    :param levels: List[int]: Vertebral levels to aggregate metrics from. It has priority over "slices".
    :param Bool perslice: Aggregate per slice (True) or across slices (False)
    :param Bool perlevel: Aggregate per level (True) or across levels (False). Has priority over "perslice".
    :param vert_level: Vertebral level. Could be either an Image or a file name.
    :param tuple group_funcs: Functions to apply on metrics. Example: (('mean', np.mean),))
    :return: Aggregated metrics
    """
    # if slices is empty, select all available slices from the metrics
    if not slices:
        slices = range(metrics.shape[2])
        # slices = metrics[metrics.keys()[0]].z

    # aggregation based on levels
    if levels:
        im_vert_level = Image(vert_level).change_orientation('RPI')
        # slicegroups = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
        slicegroups = [tuple(get_slices_from_vertebral_levels(im_vert_level, level)) for level in levels]
        if perlevel:
            # vertgroups = [(2,), (3,), (4,)]
            vertgroups = [tuple([level]) for level in levels]
        elif perslice:
            # slicegroups = [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,)]
            slicegroups = [tuple([i]) for i in reduce(operator.concat, slicegroups)]  # reduce to individual tuple
            # vertgroups = [(2,), (2,), (2,), (3,), (3,), (3,), (4,), (4,), (4,)]
            vertgroups = [tuple([get_vertebral_level_from_slice(im_vert_level, i[0])]) for i in slicegroups]
        # output aggregate metrics across levels
        else:
            # slicegroups = [(0, 1, 2, 3, 4, 5, 6, 7, 8)]
            slicegroups = [tuple([val for sublist in slicegroups for val in sublist])]  # flatten into single tuple
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

        # add label info
        # agg_metrics[slicegroup]['label'] = metrics.label  # TODO
        # metric_data = []
        # # make sure metric and z have same length
        # if not len(metrics[metric].z) == len(metrics[metric].value):
        #     # TODO: raise custom exception instead of hard-coding error message
        #     agg_metrics[slicegroup]['metrics'][metric]['error'] = 'metric and z have do not have the same length'
        # else:
        # for iz in slicegroup:
            # if iz in metrics[metric].z:
            #     metric_data.append(metrics[metric].value[metrics[metric].z.index(iz)])
            # else:
            #     # sct.log.warning('z={} is not listed in the metric.'.format(iz))
            #     agg_metrics[slicegroup]['metrics'][metric]['error'] = 'z={} is not listed in the metric.'.format(iz)
        try:
            # Loop across functions (typically: mean, std)
            for (name, func) in group_funcs:
                data_slicegroup = metrics[:, :, slicegroup]
                if mask is not None:
                    mask_slicegroup = mask[:, :, slicegroup]
                # check if nan
                result = func(data_slicegroup, mask_slicegroup)
                if np.isnan(result):
                    # TODO: fix below
                    agg_metrics[slicegroup]['error'] = 'Contains nan'
                else:
                    agg_metrics[slicegroup][name] = result
        except Exception as e:
            sct.log.warning(e)
            # sct.log.warning('TypeError for metric {}'.format(metric))
            # TODO
            agg_metrics[slicegroup]['error'] = e.message
    return agg_metrics


def func_bin(data, mask):
    # Binarize mask
    # TODO
    # run weighted average
    return func_weighted_average(data, mask_bin)


def func_weighted_average(data, mask):
    """
    Compute weighted average
    :param data: ndarray: input data
    :param mask: ndarray: input mask to weight average
    :return: weighted_average
    """
    return np.sum(np.multiply(data, mask))


def make_a_string(item):
    """Convert tuple or list or None to a string. Important: elements in tuple or list are separated with ; (not ,)
    for compatibility with csv."""
    if isinstance(item, tuple) or isinstance(item, list):
        return ';'.join([str(i) for i in item])
    elif item is None:
        return 'None'
    else:
        return item


def save_as_csv(agg_metrics, fname_out, fname_in=None, append=False):
    """
    Write metric structure as csv. If field 'error' exists, it will add a specific column.
    :param metric: output of aggregate_per_slice_or_level()
    :param fname_out: output filename. Extention (.csv) will be added if it does not exist.
    :param fname_in: input file to be listed in the csv file (e.g., segmentation file which produced the results).
    :param append: Bool: Append results at the end of file (if exists) instead of overwrite.
    :return:
    """
    # TODO: if append=True but file does not exist yet, raise warning and set append=False
    # TODO: build header based on existing func (e.g., will currently crash if no STD).
    # write header (only if append=False)
    if not append:
        with open(fname_out, 'w') as csvfile:
            # spamwriter = csv.writer(csvfile, delimiter=',')
            header = ['Filename', 'Slice (I->S)', 'Vertebral level']
            funcs = agg_metrics[agg_metrics.keys()[0]].keys()
            funcs.remove('VertLevel')  # Remove VertLevel because already added above
            for func in funcs:
                header.append(func)  # TODO: write label instead of field name
            header.append('SCT Version')
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()

    # populate data
    with open(fname_out, 'a') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for slicegroup in sorted(agg_metrics.keys()):
            line = []
            line.append(fname_in)  # file name associated with the results
            line.append(parse_num_list_inv(slicegroup))  # list all slices in slicegroup
            line.append(parse_num_list_inv(agg_metrics[slicegroup]['VertLevel']))  # list vertebral levels
            funcs = agg_metrics[slicegroup].keys()
            funcs.remove('VertLevel')  # Remove VertLevel because already added above
            for func in funcs:
                try:
                    line.append(str(agg_metrics[slicegroup][func]))
                except KeyError:
                    line.append('')
            line.append(sct.__version__)
            spamwriter.writerow(line)
