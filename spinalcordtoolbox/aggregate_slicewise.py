#!/usr/bin/env python
# -*- coding: utf-8
# Functions dealing with metrics aggregation (mean, std, etc.) across slices and/or vertebral levels


from __future__ import absolute_import

import numpy as np
import math
import operator
import functools
import csv
import datetime

import sct_utils as sct
from spinalcordtoolbox.template import get_slices_from_vertebral_levels, get_vertebral_level_from_slice
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils import parse_num_list_inv


class Metric:
    """
    Class to include in dictionaries to associate data and label
    """
    def __init__(self, data=None, label=''):
        """
        :param data: ndarray
        :param label: str
        """
        self.data = data
        self.label = label


class LabelStruc:
    """
    Class for labels
    """
    # TODO: at some point, simplify the number of classes and variables related to labels. For example, we should be
    # able to use Metric for labels and combine data and metadata.
    def __init__(self, id, name, filename=None):
        self.id = id
        self.name = name
        self.filename = filename


# TODO: don't make metrics a dict anymore-- it complicates things.
# TODO: generalize this function to accept n-dim np.array instead of list in Metric().value
# TODO: maybe no need to bring Metric() class here. Just np.array, then labeling is done in parent function.
def aggregate_per_slice_or_level(metric, mask=None, slices=None, levels=None, perslice=None, perlevel=False, vert_level=None,
                                 group_funcs=(('MEAN', np.mean),)):
    """
    The aggregation will be performed along the last dimension of 'metric' ndarray.
    :param metric: Class Metric(): data to aggregate.
    :param mask: Class Metric(): mask to use for aggregating the data. Optional.
    :param slices: List[int]: Slices to aggregate metric from. If empty, select all slices.
    :param levels: List[int]: Vertebral levels to aggregate metric from. It has priority over "slices".
    :param Bool perslice: Aggregate per slice (True) or across slices (False)
    :param Bool perlevel: Aggregate per level (True) or across levels (False). Has priority over "perslice".
    :param vert_level: Vertebral level. Could be either an Image or a file name.
    :param tuple group_funcs: Functions to apply on metric. Example: (('mean', np.mean),))
    :return: Aggregated metric
    """
    # if slices is empty, select all available slices from the metric
    ndim = metric.data.ndim
    if not slices:
        slices = range(metric.data.shape[ndim-1])
        # slices = metric[metric.keys()[0]].z

    # If user set perlevel but did not set perslice, force perslice==False. Otherwise, set to True.
    if perslice is None:
        if perlevel:
            perslice = False
        else:
            perslice = True

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
        # output aggregate metric across levels
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
    agg_metric = dict((slicegroup, dict()) for slicegroup in slicegroups)

    # loop across slice group
    for slicegroup in slicegroups:
        # add level info
        if levels:
            agg_metric[slicegroup]['VertLevel'] = vertgroups[slicegroups.index(slicegroup)]
        else:
            agg_metric[slicegroup]['VertLevel'] = None

        # add label info
        # agg_metric[slicegroup]['label'] = metric.label  # TODO
        # metric_data = []
        # # make sure metric and z have same length
        # if not len(metric[metric].z) == len(metric[metric].value):
        #     # TODO: raise custom exception instead of hard-coding error message
        #     agg_metric[slicegroup]['metric'][metric]['error'] = 'metric and z have do not have the same length'
        # else:
        # for iz in slicegroup:
            # if iz in metric[metric].z:
            #     metric_data.append(metric[metric].value[metric[metric].z.index(iz)])
            # else:
            #     # sct.log.warning('z={} is not listed in the metric.'.format(iz))
            #     agg_metric[slicegroup]['metric'][metric]['error'] = 'z={} is not listed in the metric.'.format(iz)
        try:
            # Loop across functions (typically: mean, std)
            for (name, func) in group_funcs:
                data_slicegroup = metric.data[..., slicegroup]  # selection is done in the last dimension
                if mask is not None:
                    mask_slicegroup = mask.data[..., slicegroup, :]
                    agg_metric[slicegroup]['Label'] = mask.label
                else:
                    mask_slicegroup = np.ones(data_slicegroup.shape)
                # check if nan
                result = func(data_slicegroup, mask_slicegroup)
                if np.isnan(result):
                    # TODO: fix below
                    agg_metric[slicegroup]['error'] = 'Contains nan'
                else:
                    # here we create a field with name: FUNC(METRIC_NAME). Example: MEAN(CSA)
                    agg_metric[slicegroup]['{}({})'.format(name, metric.label)] = result
        except Exception as e:
            sct.log.warning(e)
            # sct.log.warning('TypeError for metric {}'.format(metric))
            # TODO
            agg_metric[slicegroup]['error'] = e.message
    return agg_metric


def check_labels(indiv_labels_ids, selected_labels):
    """Check the consistency of the labels asked by the user."""

    # TODO: allow selection of combined labels as "36, Ventral, 7:14,22:19"

    # convert strings to int
    list_ids_of_labels_of_interest = list(map(int, indiv_labels_ids))

    # if selected_labels:
    #     # Check if label chosen is in the right format
    #     for char in selected_labels:
    #         if not char in '0123456789,:':
    #             sct.printv(parser.usage.generate(error='\nERROR: ' + selected_labels + ' is not the correct format to select combined labels.\n Exit program.\n'))
    #
    #     if ':' in selected_labels:
    #         label_ids_range = [int(x) for x in selected_labels.split(':')]
    #         if len(label_ids_range) > 2:
    #             sct.printv(parser.usage.generate(error='\nERROR: Combined labels ID selection must be in format X:Y, with X and Y between 0 and 31.\nExit program.\n\n'))
    #         else:
    #             label_ids_range.sort()
    #             list_ids_of_labels_of_interest = [int(x) for x in range(label_ids_range[0], label_ids_range[1]+1)]
    #
    #     else:
    #         list_ids_of_labels_of_interest = [int(x) for x in selected_labels.split(',')]

    if selected_labels:
        # Remove redundant values
        list_ids_of_labels_of_interest = [i_label for n, i_label in enumerate(selected_labels) if i_label not in selected_labels[:n]]

        # Check if the selected labels are in the available labels ids
        if not set(list_ids_of_labels_of_interest).issubset(set(indiv_labels_ids)):
            sct.log.error(
                'At least one of the selected labels (' + str(list_ids_of_labels_of_interest) + ') is not available \
                according to the label list from the text file in the atlas folder.')

    return list_ids_of_labels_of_interest


def diff_between_list_or_int(l1, l2):
    """
    Return list l1 minus the elements in l2
    :param l1: a list of int
    :param l2: could be a list or an int
    :return:
    """
    if isinstance(l2, int):
        l2 = [l2]
    return [x for x in l1 if x not in l2]


def extract_metric(data, labels=None, slices=None, levels=None, perslice=True, perlevel=False,
                   vert_level=None, method=None, label_struc=None, id_label=None, indiv_labels_ids=None,
                   id_label_compl=None):
    """

    :param data: Class Metric(): Data (a.k.a. metric) of n-dimension to extract aggregated value from
    :param labels: Class Metric(): Labels of (n+1)dim. The last dim encloses the labels.
    :param slices:
    :param levels:
    :param perslice:
    :param perlevel:
    :param vert_level:
    :param method:
    :param label_struc: Label structure defined in sct_extract_metric
    :param id_label: int: ID of label to select
    :param indiv_labels_ids: list of int: IDs of labels corresponding to individual (as opposed to combined) labels for
    use with ML or MAP estimation.
    :return: aggregate_per_slice_or_level()
    """

    # If label_struc[id_label].id is a list, it means that it comes from a combined labels
    if isinstance(label_struc[id_label].id, list):
        # Sum across labels
        labels_sum = np.sum(labels[..., label_struc[id_label].id], axis=3)  # (nx, ny, nz, 1)
    else:
        # Simply extract
        labels_sum = labels[..., label_struc[id_label].id]
    # expand dim: labels_sum=(..., 1)
    ndim = labels_sum.ndim
    labels_sum = np.expand_dims(labels_sum, axis=ndim)

    # Maximum Likelihood
    if method == 'ml':
        id_label_compl = diff_between_list_or_int(indiv_labels_ids, label_struc[id_label].id)
        labels_sum = np.concatenate([labels_sum, labels[..., id_label_compl]], axis=ndim)
        mask = Metric(data=labels_sum, label=label_struc[id_label].name)
        group_funcs = (('ML', func_ml), ('STD', func_std))
    # Weighted average
    elif method == 'wa':
        mask = Metric(data=labels_sum, label=label_struc[id_label].name)
        group_funcs = (('WA', func_wa), ('STD', func_std))

    return aggregate_per_slice_or_level(data, mask=mask, slices=slices, levels=levels, perslice=perslice, perlevel=perlevel,
                                        vert_level=vert_level, group_funcs=group_funcs)

# def func_mean(data):

def func_bin(data, mask=None):
    # Binarize mask
    # TODO
    # run weighted average
    return func_wa(data, mask_bin)


def func_std(data, mask=None):
    """
    Compute standard deviation
    :param data: ndarray: input data
    :param mask: ndarray: input mask to weight average
    :return: std
    """
    # Check if mask has an additional dimension (in case it is a label). If so, squeeze matrix to match dim with data.
    if mask.ndim == data.ndim + 1:
        mask = mask.squeeze()
    average = func_wa(data, mask)
    variance = np.average((data - average) ** 2, weights=mask)
    return math.sqrt(variance)


def func_wa(data, mask=None):
    """
    Compute weighted average
    :param data: ndarray: input data
    :param mask: ndarray: input mask to weight average
    :return: weighted_average
    """
    # Check if mask has an additional dimension (in case it is a label). If so, squeeze matrix to match dim with data.
    if mask.ndim == data.ndim + 1:
        mask = mask.squeeze()
    return np.average(data, weights=mask)
    # return np.sum(np.multiply(data, mask))) / np.sum(mask)


def func_ml(data, mask):
    """
    Compute maximum likelihood (ML) for the first label of mask.
    :param data: nd-array: input data
    :param mask: (n+1)d-array: input mask. Note: this mask should include ALL labels to satisfy the necessary condition for
    ML-based estimation, i.e., at each voxel, the sum of all labels (across the last dimension) equals the probability
    to be inside the tissue. For example, for a pixel within the spinal cord, the sum of all labels should be 1.
    :return: float: beta corresponding to the first label
    """
    # reshape as 1d vector (for data) and 2d vector (for mask)
    n_vox = functools.reduce(operator.mul, data.shape, 1)
    data1d = np.reshape(data, n_vox)
    mask2d = np.reshape(mask, (n_vox, mask.shape[mask.ndim-1]))
    # ML estimation:
    #   y: measurements vector (to which weights are applied)
    #   x: linear relation between the measurements y
    #   W: weights to apply to each voxel
    #   beta = (Xt . X)-1 . Xt . y     The true metric value to be estimated
    W = np.diag(np.ones(n_vox))
    y = np.dot(W, data1d)  # [nb_vox x 1]
    x = np.dot(W, mask2d)  # [nb_vox x nb_labels]
    beta = np.dot(np.linalg.pinv(np.dot(x.T, x)), np.dot(x.T, y))
    return beta[0]


def make_a_string(item):
    """Convert tuple or list or None to a string. Important: elements in tuple or list are separated with ; (not ,)
    for compatibility with csv."""
    if isinstance(item, tuple) or isinstance(item, list):
        return ';'.join([str(i) for i in item])
    elif item is None:
        return 'None'
    else:
        return item


def save_as_csv(agg_metric, fname_out, fname_in=None, append=False):
    """
    Write metric structure as csv. If field 'error' exists, it will add a specific column.
    :param agg_metric: output of aggregate_per_slice_or_level()
    :param fname_out: output filename. Extention (.csv) will be added if it does not exist.
    :param fname_in: input file to be listed in the csv file (e.g., segmentation file which produced the results).
    :param append: Bool: Append results at the end of file (if exists) instead of overwrite.
    :return:
    """
    # Item sorted in order for display in csv output
    list_item = ['VertLevel', 'Label', 'MEAN', 'WA', 'WATH', 'BIN', 'ML', 'MAP', 'STD']
    # TODO: add timestamp + move file at the end
    # TODO: if append=True but file does not exist yet, raise warning and set append=False
    # TODO: build header and data based on existing keys, and find a way to sort them
    # write header (only if append=False)
    if not append:
        with open(fname_out, 'w') as csvfile:
            # spamwriter = csv.writer(csvfile, delimiter=',')
            header = ['Timestamp', 'SCT Version', 'Filename', 'Slice (I->S)']
            agg_metric_key = agg_metric[agg_metric.keys()[0]].keys()
            for item in list_item:
                for key in agg_metric_key:
                    if item in key:
                        header.append(key)
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()

    # populate data
    with open(fname_out, 'a') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for slicegroup in sorted(agg_metric.keys()):
            line = list()
            line.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))  # Timestamp
            line.append(sct.__version__)  # SCT Version
            line.append(fname_in)  # file name associated with the results
            line.append(parse_num_list_inv(slicegroup))  # list all slices in slicegroup
            agg_metric_key = agg_metric[agg_metric.keys()[0]].keys()
            for item in list_item:
                for key in agg_metric_key:
                    if item in key:
                        # Special case for VertLevel
                        if key == 'VertLevel':
                            line.append(
                                parse_num_list_inv(agg_metric[slicegroup]['VertLevel']))  # list vertebral levels
                        else:
                            line.append(str(agg_metric[slicegroup][key]))
            spamwriter.writerow(line)
