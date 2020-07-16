#!/usr/bin/env python
# -*- coding: utf-8
# Functions dealing with metrics aggregation (mean, std, etc.) across slices and/or vertebral levels

# TODO: when mask is empty, raise specific message instead of throwing "Weight sum to zero..."

from __future__ import absolute_import

import os
import numpy as np
import math
import operator
import functools
import csv
import datetime
import logging

from spinalcordtoolbox.template import get_slices_from_vertebral_levels, get_vertebral_level_from_slice
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils import __version__, parse_num_list_inv


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
    Compute maximum a posteriori (MAP) by aggregating the last dimension of mask according to a clustering method
    defined by map_clusters

    :param data: nd-array: input data
    :param mask: (n+1)d-array: input mask. Note: this mask should include ALL labels to satisfy the necessary condition for\
    ML-based estimation, i.e., at each voxel, the sum of all labels (across the last dimension) equals the probability\
    to be inside the tissue. For example, for a pixel within the spinal cord, the sum of all labels should be 1.
    :param map_clusters: list of list of int: Each sublist corresponds to a cluster of labels where ML estimation will\
    be performed to provide the prior beta_0 for MAP estimation.
    :return: float: beta corresponding to the first label (beta[0])
    :return: nd-array: matrix of all beta
    """
    # Check number of labels and map_clusters
    assert mask.shape[-1] == len(map_clusters)

    # Iterate across all labels (excluding the first one) and generate cluster labels. Examples of input/output:
    #   [[0], [0], [0], [1], [2], [0]] --> [0, 0, 0, 1, 2, 0]
    #   [[0, 1], [0], [0], [1], [2]] --> [0, 0, 0, 0, 1]
    #   [[0, 1], [0], [1], [2], [3]] --> [0, 0, 0, 1, 2]
    possible_clusters = [map_clusters[0]]
    id_clusters = [0]  # this one corresponds to the first cluster
    for i_cluster in map_clusters[1:]:  # skip the first
        found_index = False
        for possible_cluster in possible_clusters:
            if i_cluster[0] in possible_cluster:
                id_clusters.append(possible_clusters.index(possible_cluster))
                found_index = True
        if not found_index:
            possible_clusters.append(i_cluster)
            id_clusters.append(possible_clusters.index([i_cluster[0]]))

    # Sum across each clustered labels, then concatenate to generate mask_clusters
    # mask_clusters has dimension: x, y, z, n_clustered_labels, with n_clustered_labels being equal to the number of
    # clusters that need to be estimated for ML method. Let's assume:
    # label_struc = [
    #   LabelStruc(id=0, map_cluster=0),
    #   LabelStruc(id=1, map_cluster=0),
    #   LabelStruc(id=2, map_cluster=0),
    #   LabelStruc(id=3, map_cluster=1),
    #   LabelStruc(id=4, map_cluster=2),
    # ]
    #
    # Examples of scenario below for ML estimation:
    #   labels_id_user = [0,1], mask_clusters = [np.sum(label[0:2]), label[3], label[4]]
    #   labels_id_user = [3], mask_clusters = [np.sum(label[0:2]), label[3], label[4]]
    #   labels_id_user = [0,1,2,3], mask_clusters = [np.sum(label(0:3)), label[4]]
    mask_l = []
    for i_cluster in list(set(id_clusters)):
        # Get label indices for given cluster
        id_label_cluster = [i for i in range(len(id_clusters)) if i_cluster == id_clusters[i]]
        # Sum all labels for this cluster
        mask_l.append(np.expand_dims(np.sum(mask[..., id_label_cluster], axis=(mask.ndim - 1)), axis=(mask.ndim - 1)))
    mask_clusters = np.concatenate(mask_l, axis=(mask.ndim-1))

    # Run ML estimation for each clustered labels
    _, beta_cluster = func_ml(data, mask_clusters)

    # MAP estimation:
    #   y [nb_vox x 1]: measurements vector (to which weights are applied)
    #   x [nb_vox x nb_labels]: linear relation between the measurements y
    #   beta_0 [nb_labels]: A priori values estimated per cluster using ML.
    #   beta [nb_labels] = beta_0 + (Xt . X + 1)^(-1) . Xt . (y - X . beta_0): The estimated metric value in each label
    #
    # Note: for simplicity we consider that sigma_noise = sigma_label
    n_vox = functools.reduce(operator.mul, data.shape, 1)
    y = np.reshape(data, n_vox)
    x = np.reshape(mask, (n_vox, mask.shape[mask.ndim-1]))
    beta_0 = [beta_cluster[id_clusters[i_label]] for i_label in range(mask.shape[-1])]
    beta = beta_0 + np.dot(np.linalg.pinv(np.dot(x.T, x) + np.diag(np.ones(mask.shape[-1]))),
                           np.dot(x.T,
                                  (y - np.dot(x, beta_0))))
    return beta[0], beta


def func_ml(data, mask, map_clusters=None):
    """
    Compute maximum likelihood (ML) for the first label of mask.

    :param data: nd-array: input data
    :param mask: (n+1)d-array: input mask. Note: this mask should include ALL labels to satisfy the necessary condition\
    for ML-based estimation, i.e., at each voxel, the sum of all labels (across the last dimension) equals the\
    probability to be inside the tissue. For example, for a pixel within the spinal cord, the sum of all labels should\
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


def func_sum(data, mask=None, map_clusters=None):
    """
    Compute sum

    :param data: nd-array: input data
    :param mask: not used
    :param map_clusters: not used
    :return:
    """
    # Check if mask has an additional dimension (in case it is a label). If so, select the first label
    return np.sum(data), None


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


def aggregate_per_slice_or_level(metric, mask=None, slices=[], levels=[], perslice=None, perlevel=False,
                                 vert_level=None, group_funcs=(('MEAN', func_wa),), map_clusters=None):
    """
    The aggregation will be performed along the last dimension of 'metric' ndarray.

    :param metric: Class Metric(): data to aggregate.
    :param mask: Class Metric(): mask to use for aggregating the data. Optional.
    :param slices: List[int]: Slices to aggregate metric from. If empty, select all slices.
    :param levels: List[int]: Vertebral levels to aggregate metric from. It has priority over "slices".
    :param Bool perslice: Aggregate per slice (True) or across slices (False)
    :param Bool perlevel: Aggregate per level (True) or across levels (False). Has priority over "perslice".
    :param vert_level: Vertebral level. Could be either an Image or a file name.
    :param tuple group_funcs: Name and function to apply on metric. Example: (('MEAN', func_wa),)). Note, the function
      has special requirements in terms of i/o. See the definition to func_wa and use it as a template.
    :param map_clusters: list of list of int: See func_map()
    :return: Aggregated metric
    """
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
            slicegroups = [tuple([i]) for i in functools.reduce(operator.concat, slicegroups)]  # reduce to individual tuple
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
        if vertgroups is None:
            agg_metric[slicegroup]['VertLevel'] = None
        else:
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
                # Make sure the number of pixels to extract metrics is not null
                if mask_slicegroup.sum() == 0:
                    result = None
                else:
                    # Run estimation
                    result, _ = func(data_slicegroup, mask_slicegroup, map_clusters)
                    # check if nan
                    if np.isnan(result):
                        result = None
                # here we create a field with name: FUNC(METRIC_NAME). Example: MEAN(CSA)
                agg_metric[slicegroup]['{}({})'.format(name, metric.label)] = result
            except Exception as e:
                logging.warning(e)
                agg_metric[slicegroup]['{}({})'.format(name, metric.label)] = str(e)
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
            logging.error(
                'At least one of the selected labels (' + str(list_ids_of_labels_of_interest) + ') is not available \
                according to the label list from the text file in the atlas folder.')

    return list_ids_of_labels_of_interest


def diff_between_list_or_int(l1, l2):
    """
    Return list l1 minus the elements in l2
    Examples:\
      ([1, 2, 3], 1) --> [2, 3]\
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
    :param label_struc: LabelStruc class defined above
    :param id_label: int: ID of label to select
    :param indiv_labels_ids: list of int: IDs of labels corresponding to individual (as opposed to combined) labels for\
    use with ML or MAP estimation.
    :return: aggregate_per_slice_or_level()
    """
    # Initializations
    map_clusters = None
    func_methods = {'ml': ('ML', func_ml), 'map': ('MAP', func_map)}  # TODO: complete dict with other methods
    # If label_struc[id_label].id is a list (i.e. comes from a combined labels), sum all labels
    if isinstance(label_struc[id_label].id, list):
        labels_sum = np.sum(labels[..., label_struc[id_label].id], axis=labels.ndim-1)  # (nx, ny, nz, 1)
    else:
        labels_sum = labels[..., label_struc[id_label].id]
    # expand dim: labels_sum=(..., 1)
    ndim = labels_sum.ndim
    labels_sum = np.expand_dims(labels_sum, axis=ndim)

    # Maximum Likelihood or Maximum a Posteriori
    if method in ['ml', 'map']:
        # Get the complementary list of labels (the ones not asked by the user)
        id_label_compl = diff_between_list_or_int(indiv_labels_ids, label_struc[id_label].id)
        # Generate a list of map_clusters for each label. Start with the first label (the one chosen by the user).
        # Note that the first label could be a combination of several labels (e.g., WM and GM).
        if isinstance(label_struc[id_label].id, list):
            # in case there are several labels for this id_label
            map_clusters = [list(set([label_struc[i].map_cluster for i in label_struc[id_label].id]))]
        else:
            # in case there is only one label for this id_label
            map_clusters = [[label_struc[id_label].map_cluster]]
        # Append the cluster for each remaining labels (i.e. the ones not included in the combined labels)
        for i_cluster in [label_struc[i].map_cluster for i in id_label_compl]:
            map_clusters.append([i_cluster])
        # Concatenate labels: first, the one asked by the user, then the remaining ones.
        # Examples of scenario:
        #   labels_sum = [[0], [1:36]]
        #   labels_sum = [[3,4], [0,2,5:36]]
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


def make_a_string(item):
    """Convert tuple or list or None to a string. Important: elements in tuple or list are separated with ; (not ,)
    for compatibility with csv."""
    if isinstance(item, tuple) or isinstance(item, list):
        return ';'.join([str(i) for i in item])
    elif item is None:
        return 'None'
    else:
        return item


def _merge_dict(dict_in):
    """
    Merge n dictionaries that are contained at the root key
    Example:
      dict_in = {
          'area': {(0): {'Level': 0, 'Mean(area)': 0.5}, (1): {'Level': 1, 'Mean(area)': 0.2}}
          'angle_RL': {(0): {'Level': 0, 'Mean(angle_RL)': 15}, (1): {'Level': 1, 'Mean(angle_RL)': 12}}
      }
      dict_merged = {
          (0): {'Level': 0, 'Mean(area): 0.5, 'Mean(angle_RL): 15}
          (1): {'Level': 1, 'Mean(area): 0.2, 'Mean(angle_RL): 12}
      }

    :param dict_in:
    :return:
    """
    dict_merged = {}
    metrics = [k for i, (k, v) in enumerate(dict_in.items())]
    # Fetch first parent key (metric), then loop across children keys (slicegroup):
    dict_first_metric = dict_in[metrics[0]]
    # Loop across children keys: slicegroup = [(0), (1), ...]
    for slicegroup in [k for i, (k, v) in enumerate(dict_first_metric.items())]:
        # Initialize dict with information from first metric
        dict_merged[slicegroup] = dict_first_metric[slicegroup]
        # Loop across remaining metrics
        for metric in metrics:
            dict_merged[slicegroup].update(dict_in[metric][slicegroup])
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
    list_item = ['Label', 'Size [vox]', 'MEAN(area)', 'STD(area)', 'MEAN(angle_AP)', 'STD(angle_AP)', 'MEAN(angle_RL)',
                 'STD(angle_RL)', 'MEAN(diameter_AP)', 'STD(diameter_AP)', 'MEAN(diameter_RL)', 'STD(diameter_RL)',
                 'MEAN(eccentricity)', 'STD(eccentricity)', 'MEAN(orientation)', 'STD(orientation)',
                 'MEAN(solidity)', 'STD(solidity)', 'SUM(length)', 'WA()', 'BIN()', 'ML()', 'MAP()', 'STD()', 'MAX()']
    # TODO: if append=True but file does not exist yet, raise warning and set append=False
    # write header (only if append=False)
    if not append or not os.path.isfile(fname_out):
        with open(fname_out, 'w') as csvfile:
            # spamwriter = csv.writer(csvfile, delimiter=',')
            header = ['Timestamp', 'SCT Version', 'Filename', 'Slice (I->S)', 'VertLevel']
            agg_metric_key = [v for i, (k, v) in enumerate(agg_metric.items())][0]
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
            line.append(__version__)  # SCT Version
            line.append(fname_in)  # file name associated with the results
            line.append(parse_num_list_inv(slicegroup))  # list all slices in slicegroup
            line.append(parse_num_list_inv(agg_metric[slicegroup]['VertLevel']))  # list vertebral levels
            agg_metric_key = [v for i, (k, v) in enumerate(agg_metric.items())][0]
            for item in list_item:
                for key in agg_metric_key:
                    if item in key:
                        line.append(str(agg_metric[slicegroup][key]))
                        break
            spamwriter.writerow(line)
