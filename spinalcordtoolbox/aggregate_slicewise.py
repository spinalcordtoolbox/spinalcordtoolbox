#!/usr/bin/env python
# -*- coding: utf-8
# Functions dealing with metrics aggregation (mean, std, etc.) across slices and/or vertebral levels

# TODO: when mask is empty, raise specific message instead of throwing "Weight sum to zero..."

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
    def __init__(self, id, name, filename=None, map_cluster=None):
        self.id = id
        self.name = name
        self.filename = filename
        self.map_cluster = map_cluster


def aggregate_per_slice_or_level(metric, mask=None, slices=[], levels=[], perslice=None, perlevel=False,
                                 vert_level=None, group_funcs=(('MEAN', np.mean),), map_clusters=None):
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
    :param map_clusters: list of list of int: See func_map()
    :return: Aggregated metric
    """
    # TODO: always add vertLevel if exists

    # If user neither specified slices nor levels, set perslice=True, otherwise, the output will likely contain nan
    # because in many cases the segmentation does not span the whole I-S dimension.
    if perslice is None:
        if not slices and not levels:
            perslice = True
        else:
            perslice = False

    # if slices is empty, select all available slices from the metric
    ndim = metric.data.ndim
    if not slices:
        slices = range(metric.data.shape[ndim-1])

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
        # Loop across functions (e.g.: MEAN, STD)
        for (name, func) in group_funcs:
            try:
                data_slicegroup = metric.data[..., slicegroup]  # selection is done in the last dimension
                if mask is not None:
                    mask_slicegroup = mask.data[..., slicegroup, :]
                    agg_metric[slicegroup]['Label'] = mask.label
                    # Add volume fraction
                    agg_metric[slicegroup]['Size [vox]'] = np.sum(mask_slicegroup.flatten())
                else:
                    mask_slicegroup = np.ones(data_slicegroup.shape)
                # Ignore nonfinite values
                i_nonfinite = np.where(np.isfinite(data_slicegroup) == False)
                data_slicegroup[i_nonfinite] = 0.
                # TODO: the lines below could probably be done more elegantly
                if mask_slicegroup.ndim == data_slicegroup.ndim + 1:
                    arr_tmp_concat = []
                    for i in range(mask_slicegroup.shape[-1]):
                        arr_tmp = np.reshape(mask_slicegroup[..., i], data_slicegroup.shape)
                        arr_tmp[i_nonfinite] = 0.
                        arr_tmp_concat.append(np.expand_dims(arr_tmp, axis=(mask_slicegroup.ndim-1)))
                    mask_slicegroup = np.concatenate(arr_tmp_concat, axis=(mask_slicegroup.ndim-1))
                else:
                    mask_slicegroup[i_nonfinite] = 0.
                # Run estimation
                result, _ = func(data_slicegroup, mask_slicegroup, map_clusters)
                # check if nan
                if np.isnan(result):
                    result = 'nan'
                # here we create a field with name: FUNC(METRIC_NAME). Example: MEAN(CSA)
                agg_metric[slicegroup]['{}({})'.format(name, metric.label)] = result
            except Exception as e:
                sct.log.warning(e)
                agg_metric[slicegroup]['{}({})'.format(name, metric.label)] = e.message
    return agg_metric


def check_labels(indiv_labels_ids, selected_labels):
    """Check the consistency of the labels asked by the user."""
    # convert strings to int
    list_ids_of_labels_of_interest = list(map(int, indiv_labels_ids))

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
    Examples:
      ([1, 2, 3], 1) --> [2, 3]
      ([1, 2, 3], [1, 2] --> [3]
    :param l1: a list of int
    :param l2: could be a list or an int
    :return:
    """
    if isinstance(l2, int):
        l2 = [l2]
    return [x for x in l1 if x not in l2]


def extract_metric(data, labels=None, slices=None, levels=None, perslice=True, perlevel=False,
                   vert_level=None, method=None, label_struc=None, id_label=None, indiv_labels_ids=None):
    """
    Extract metric within a data, using mask and a given method.
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
    :param map_clusters: list of list of int: See func_map()
    :return: aggregate_per_slice_or_level()
    """
    # Initializations
    map_clusters = None
    func_methods = {'ml': ('ML', func_ml), 'map': ('MAP', func_map)}
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

    # Maximum Likelihood or Maximum a Posteriori
    if method in ['ml', 'map']:
        id_label_compl = diff_between_list_or_int(indiv_labels_ids, label_struc[id_label].id)
        # Generate a list of map_clusters for each label in mask.
        # Start with the first label (the one chosed by the user)
        map_clusters = [label_struc[id_label].map_cluster]
        # Then append the remaining cluster IDs
        map_clusters += [label_struc[i].map_cluster for i in id_label_compl]
        # Concatenate labels: the one asked by the user, followed by the remaining ones
        labels_sum = np.concatenate([labels_sum, labels[..., id_label_compl]], axis=ndim)
        mask = Metric(data=labels_sum, label=label_struc[id_label].name)
        group_funcs = (func_methods[method], ('STD', func_std))
    # Weighted average
    elif method == 'wa':
        mask = Metric(data=labels_sum, label=label_struc[id_label].name)
        group_funcs = (('WA', func_wa), ('STD', func_std))
    # Binarize mask
    elif method == 'bin':
        mask = Metric(data=labels_sum, label=label_struc[id_label].name)
        group_funcs = (('BIN', func_bin), ('STD', func_std))
    # Maximum
    elif method == 'max':
        mask = Metric(data=labels_sum, label=label_struc[id_label].name)
        group_funcs = (('MAX', func_max),)

    return aggregate_per_slice_or_level(data, mask=mask, slices=slices, levels=levels, perslice=perslice,
                                        perlevel=perlevel, vert_level=vert_level, group_funcs=group_funcs,
                                        map_clusters=map_clusters)


def func_bin(data, mask, map_clusters=None):
    """
    Get the average of data after binarizing the input mask
    :param data: nd-array: input data
    :param mask: (n+1)d-array: input mask
    :param map_clusters: not used
    :return:
    """
    # Binarize mask
    mask_bin = np.where(mask >= 0.5, 1, 0)
    # run weighted average
    return func_wa(data, mask_bin)


def func_max(data, mask=None, map_clusters=None):
    """
    Get the max of an array
    :param data: nd-array: input data
    :param mask: not used
    :param map_clusters: not used
    :return:
    """
    return np.max(data), None


def func_map(data, mask, map_clusters):
    """
    Compute maximum a posteriori (MAP) for the first label of mask.
    :param data: nd-array: input data
    :param mask: (n+1)d-array: input mask. Note: this mask should include ALL labels to satisfy the necessary condition for
    ML-based estimation, i.e., at each voxel, the sum of all labels (across the last dimension) equals the probability
    to be inside the tissue. For example, for a pixel within the spinal cord, the sum of all labels should be 1.
    :param map_clusters: list of list of int: Each sublist corresponds to a cluster of labels where ML estimation will
    be performed to provide the prior beta_0 for MAP estimation.
    :return: float: beta corresponding to the first label
    """
    # Check number of labels and map_clusters
    assert mask.shape[-1] == len(map_clusters)
    # Sum across each clustered labels, then concatenate
    mask_l = []
    for i_cluster in list(set(map_clusters)):
        # Get indices for given cluster
        i_clusters = [i for i in range(len(map_clusters)) if i_cluster == map_clusters[i]]
        # Sum all labels for this cluster
        mask_l.append(np.expand_dims(np.sum(mask[..., i_clusters], axis=(mask.ndim - 1)), axis=(mask.ndim - 1)))
    mask_clusters = np.concatenate(mask_l, axis=(mask.ndim-1))
    # Run ML estimation for each clustered labels
    _, beta_cluster = func_ml(data, mask_clusters)
    # MAP estimation:
    #   y [nb_vox x 1]: measurements vector (to which weights are applied)
    #   x [nb_vox x nb_labels]: linear relation between the measurements y
    #   beta_0 [nb_labels]: A priori values estimated per cluster using ML.
    #   beta [nb_labels] = beta_0 + (Xt . X + 1)^(-1) . Xt . (y - X . beta_0) : The estimated metric value in each label
    #
    # Note: for simplicity we consider that sigma_noise = sigma_label
    n_vox = functools.reduce(operator.mul, data.shape, 1)
    y = np.reshape(data, n_vox)
    x = np.reshape(mask, (n_vox, mask.shape[mask.ndim-1]))
    beta_0 = [beta_cluster[map_clusters[i_label]] for i_label in range(mask.shape[-1])]
    beta = beta_0 + np.dot(np.linalg.pinv(np.dot(x.T, x) + np.diag(np.ones(mask.shape[-1]))),
                           np.dot(x.T,
                                  (y - np.dot(x, beta_0))))
    return beta[0], beta


def func_ml(data, mask, map_clusters=None):
    """
    Compute maximum likelihood (ML) for the first label of mask.
    :param data: nd-array: input data
    :param mask: (n+1)d-array: input mask. Note: this mask should include ALL labels to satisfy the necessary condition
    for ML-based estimation, i.e., at each voxel, the sum of all labels (across the last dimension) equals the
    probability to be inside the tissue. For example, for a pixel within the spinal cord, the sum of all labels should
    be 1.
    :return: float: beta corresponding to the first label
    """
    # TODO: support weighted least square
    # reshape as 1d vector (for data) and 2d vector (for mask)
    n_vox = functools.reduce(operator.mul, data.shape, 1)
    # ML estimation:
    #   y: measurements vector (to which weights are applied)
    #   x: linear relation between the measurements y
    #   beta [nb_labels] = (Xt . X)^(-1) . Xt . y: The estimated metric value in each label
    y = np.reshape(data, n_vox)  # [nb_vox x 1]
    x = np.reshape(mask, (n_vox, mask.shape[mask.ndim-1]))
    beta = np.dot(np.linalg.pinv(np.dot(x.T, x)), np.dot(x.T, y))
    return beta[0], beta


def func_std(data, mask=None, map_clusters=None):
    """
    Compute standard deviation
    :param data: nd-array: input data
    :param mask: (n+1)d-array: input mask
    :param map_clusters: not used
    :return:
    """
    # Check if mask has an additional dimension (in case it is a label). If so, select the first label
    if mask.ndim == data.ndim + 1:
        mask = mask[..., 0]
    average, _ = func_wa(data, np.expand_dims(mask, axis=mask.ndim))
    variance = np.average((data - average) ** 2, weights=mask)
    return math.sqrt(variance), None


def func_wa(data, mask=None, map_clusters=None):
    """
    Compute weighted average
    :param data: nd-array: input data
    :param mask: (n+1)d-array: input mask
    :param map_clusters: not used
    :return:
    """
    # Check if mask has an additional dimension (in case it is a label). If so, select the first label
    if mask.ndim == data.ndim + 1:
        mask = mask[..., 0]
    return np.average(data, weights=mask), None


def make_a_string(item):
    """Convert tuple or list or None to a string. Important: elements in tuple or list are separated with ; (not ,)
    for compatibility with csv."""
    if isinstance(item, tuple) or isinstance(item, list):
        return ';'.join([str(i) for i in item])
    elif item is None:
        return 'None'
    else:
        return item


def merge_dict(dict_in):
    """
    Merge n dictionaries that are contained at the root key
    Example:
      dict = {'key1': {'subkey1': dict1}, 'key2': {'subkey2': dict2}} --> {'subkey1': dict1+dict2, ...}
    :param dict_in:
    :return:
    """
    dict_merged = {}
    # Fetch first parent key (metric), then loop across children keys (slicegroup):
    for key_children in dict_in[dict_in.keys()[0]].keys():
        # Loop across remaining parent keys
        dict_merged[key_children] = dict_in[dict_in.keys()[0]][key_children].copy()
        for key_parent in dict_in.keys()[1:]:
            dict_merged[key_children].update(dict_in[key_parent][key_children])
    return dict_merged


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
    # list_item = ['VertLevel', 'Label', 'MEAN', 'WA', 'BIN', 'ML', 'MAP', 'STD', 'MAX']
    # TODO: The thing below is ugly and needs to be fixed, but this is the only solution I found to order the columns
    #  without refactoring the code with OrderedDict.
    list_item = ['VertLevel', 'Label', 'Size [vox]', 'MEAN(area)', 'STD(area)', 'MEAN(AP_diameter)', 'STD(AP_diameter)',
                 'MEAN(RL_diameter)', 'STD(RL_diameter)', 'MEAN(ratio_minor_major)', 'STD(ratio_minor_major)',
                 'MEAN(eccentricity)', 'STD(eccentricity)', 'MEAN(orientation)', 'STD(orientation)',
                 'MEAN(equivalent_diameter)', 'STD(equivalent_diameter)', 'MEAN(solidity)', 'STD(solidity)',
                 'MEAN(CSA', 'STD(CSA', 'MEAN(Angle', 'STD(Angle', 'WA()', 'BIN()', 'ML()', 'MAP()', 'STD()', 'MAX()']
    # TODO: if append=True but file does not exist yet, raise warning and set append=False
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
                        break
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
                        break
            spamwriter.writerow(line)
