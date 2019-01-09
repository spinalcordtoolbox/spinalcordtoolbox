#!/usr/bin/env python
# -*- coding: utf-8
# Functions to extract metrics using segmentations and atlases

from __future__ import division, absolute_import

import numpy as np

import sct_utils as sct


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


def estimate_metric_within_tract(data, labels, method, verbose, clustered_labels=[], matching_cluster_labels=[], adv_param=[], im_weight=None):
    """Extract metric within labels.
    :data: (nx,ny,nz) numpy array
    :labels: nlabel tuple of (nx,ny,nz) array
    """

    nb_labels = len(labels)  # number of labels

    # if user asks for binary regions, binarize atlas
    if method == 'bin' or method == 'max':
        for i in range(0, nb_labels):
            labels[i][labels[i] < 0.5] = 0
            labels[i][labels[i] >= 0.5] = 1

    # if user asks for thresholded weighted-average, threshold atlas
    if method == 'wath':
        for i in range(0, nb_labels):
            labels[i][labels[i] < 0.5] = 0

    #  Select non-zero values in the union of all labels
    labels_sum = np.sum(labels)
    ind_positive_labels = labels_sum > np.finfo(float).eps
    # ind_positive_data = data > -9999999999  # data > 0
    ind_positive = ind_positive_labels  # & ind_positive_data
    data1d = data[ind_positive]
    nb_vox = len(data1d)
    labels2d = np.empty([nb_labels, nb_vox], dtype=float)
    for i in range(0, nb_labels):
        labels2d[i] = labels[i][ind_positive]

    if method == 'map' or 'ml':
        # if specified (flag -mask-weighted), define a matrix to weight voxels. If not, this matrix is set to identity.
        if im_weight:
            data_weight_1d = im_weight.data[ind_positive]
        else:
            data_weight_1d = np.ones(nb_vox)
        W = np.diag(data_weight_1d)  # weight matrix

    # Display number of non-zero values
    sct.log.info('Number of non-null voxels: ' + str(nb_vox))

    # initialization
    metric_mean = np.empty([nb_labels], dtype=object)
    metric_std = np.empty([nb_labels], dtype=object)

    # Estimation with maximum a posteriori (map)
    if method == 'map':

        # ML estimation in the defined clusters to get a priori
        # -----------------------------------------------------

        sct.log.info('Maximum likelihood estimation within the selected clusters to get a priori for the MAP '
                     'estimation...')

        nb_clusters = len(clustered_labels)

        #  Select non-zero values in the union of the clustered labels
        clustered_labels_sum = np.sum(clustered_labels)
        ind_positive_clustered_labels = clustered_labels_sum > np.finfo(float).eps  # labels_sum > epsilon (very small)

        # define the problem to apply the maximum likelihood to clustered labels
        y_apriori = data[ind_positive_clustered_labels]  # [nb_vox x 1]

        # create matrix X to use ML and estimate beta_0
        x_apriori = np.zeros([len(y_apriori), nb_clusters])
        for i_cluster in range(nb_clusters):
            x_apriori[:, i_cluster] = clustered_labels[i_cluster][ind_positive_clustered_labels]

        # remove unused voxels from the weighting matrix W
        if im_weight:
            data_weight_1d_apriori = im_weight.data[ind_positive_clustered_labels]
        else:
            data_weight_1d_apriori = np.ones(np.sum(ind_positive_clustered_labels))
        W_apriori = np.diag(data_weight_1d_apriori)  # weight matrix

        # apply the weighting matrix
        y_apriori = np.dot(W_apriori, y_apriori)
        x_apriori = np.dot(W_apriori, x_apriori)

        # estimate values using ML for each cluster
        beta = np.dot(np.linalg.pinv(np.dot(x_apriori.T, x_apriori)), np.dot(x_apriori.T, y_apriori))  # beta = (Xt . X)-1 . Xt . y
        # display results
        sct.log.info('Estimated beta0 per cluster: ' + str(beta))

        # MAP estimations within the selected labels
        # ------------------------------------------

        # perc_var_label = int(adv_param[0])^2  # variance within label, in percentage of the mean (mean is estimated using cluster-based ML)
        var_label = int(adv_param[0]) ^ 2  # variance within label
        var_noise = int(adv_param[1]) ^ 2  # variance of the noise (assumed Gaussian)

        # define the problem: y is the measurements vector (to which weights are applied, to each voxel) and x is the linear relation between the measurements y and the true metric value to be estimated beta
        y = np.dot(W, data1d)  # [nb_vox x 1]
        x = np.dot(W, labels2d.T)  # [nb_vox x nb_labels]
        # construct beta0
        beta0 = np.zeros(nb_labels)
        for i_cluster in range(nb_clusters):
            beta0[np.where(np.asarray(matching_cluster_labels) == i_cluster)[0]] = beta[i_cluster]
        # construct covariance matrix (variance between tracts). For simplicity, we set it to be the identity.
        Rlabel = np.diag(np.ones(nb_labels))
        A = np.linalg.pinv(np.dot(x.T, x) + np.linalg.pinv(Rlabel) * var_noise / var_label)
        B = x.T
        C = y - np.dot(x, beta0)
        beta = beta0 + np.dot(A, np.dot(B, C))
        for i_label in range(0, nb_labels):
            metric_mean[i_label] = beta[i_label]
            metric_std[i_label] = 0  # need to assign a value for writing output file

    # clear memory
    del data, labels

    # Estimation with maximum likelihood
    if method == 'ml':
        # define the problem: y is the measurements vector (to which weights are applied, to each voxel) and x is the linear relation between the measurements y and the true metric value to be estimated beta
        y = np.dot(W, data1d)  # [nb_vox x 1]
        x = np.dot(W, labels2d.T)  # [nb_vox x nb_labels]
        beta = np.dot(np.linalg.pinv(np.dot(x.T, x)), np.dot(x.T, y))  # beta = (Xt . X)-1 . Xt . y
        #beta, residuals, rank, singular_value = np.linalg.lstsq(np.dot(x.T, x), np.dot(x.T, y), rcond=-1)
        #beta, residuals, rank, singular_value = np.linalg.lstsq(x, y)
        # sct.printv(beta, residuals, rank, singular_value)
        for i_label in range(0, nb_labels):
            metric_mean[i_label] = beta[i_label]
            metric_std[i_label] = 0  # need to assign a value for writing output file

    # Estimation with weighted average (also works for binary)
    if method == 'wa' or method == 'bin' or method == 'wath' or method == 'max':
        for i_label in range(0, nb_labels):
            # check if all labels are equal to zero
            if sum(labels2d[i_label, :]) == 0:
                sct.log.warning('labels #' + str(i_label) + ' contains only null voxels. Mean and std are set to 0.')
                metric_mean[i_label] = 0
                metric_std[i_label] = 0
            else:
                if method == 'max':
                    # just take the max within the mask
                    metric_mean[i_label] = max(data1d * labels2d[i_label, :])
                    metric_std[i_label] = 0  # set to 0, although this value is irrelevant here
                else:
                    # estimate the weighted average
                    metric_mean[i_label] = sum(data1d * labels2d[i_label, :]) / sum(labels2d[i_label, :])
                    # estimate the biased weighted standard deviation
                    metric_std[i_label] = np.sqrt(
                        sum(labels2d[i_label, :] * (data1d - metric_mean[i_label]) ** 2) / sum(labels2d[i_label, :]))

    return metric_mean, metric_std


def extract_metric(method, data, labels, indiv_labels_ids, clusters_labels='', adv_param='', normalizing_label=[], normalization_method='', im_weight='', combined_labels_id_group='', verbose=0):
    """Extract metric in the labels specified by the file info_label.txt in the atlas folder."""

    # Initialization to default values
    clustered_labels, matching_cluster_labels = [], []

    nb_labels_total = len(indiv_labels_ids)

    # check consistency of label input parameter (* LOI=Labels of Interest)
    list_ids_LOI = check_labels(indiv_labels_ids, combined_labels_id_group)  # If 'labels_of_interest' is empty, then label_id_user' contains the index of all labels in the file info_label.txt

    if method == 'map':
        # get clustered labels
        clustered_labels, matching_cluster_labels = get_clustered_labels(clusters_labels, labels, indiv_labels_ids, list_ids_LOI, combined_labels_id_group, verbose)

    # if user wants to get unique value across labels, then combine all labels together
    if combined_labels_id_group:
        sum_combined_labels = np.sum(labels[list_ids_LOI])  # sum the labels selected by user
        if method == 'ml' or method == 'map':  # in case the maximum likelihood and the average across different labels are wanted
            # merge labels
            labels_tmp = np.empty([nb_labels_total - len(list_ids_LOI) + 1], dtype=object)
            labels = np.delete(labels, list_ids_LOI)  # remove the labels selected by user
            labels_tmp[0] = sum_combined_labels  # put the sum of the labels selected by user in first position of the tmp variable
            for i_label in range(1, len(labels_tmp)):
                labels_tmp[i_label] = labels[i_label - 1]  # fill the temporary array with the values of the non-selected labels
            labels = labels_tmp  # replace the initial labels value by the updated ones (with the summed labels)
            del labels_tmp  # delete the temporary labels

        else:  # in other cases than the maximum likelihood, we can remove other labels (not needed for estimation)
            labels = np.empty(1, dtype=object)
            labels[0] = sum_combined_labels  # we create a new label array that includes only the summed labels

    if normalizing_label:  # if the "normalization" option is wanted
        sct.log.info('Extract normalization values...')
        if normalization_method == 'sbs':  # case: the user wants to normalize slice-by-slice
            for z in range(0, data.shape[-1]):
                normalizing_label_slice = np.empty([1], dtype=object)  # in order to keep compatibility with the function
                # 'extract_metric_within_tract', define a new array for the slice z of the normalizing labels
                normalizing_label_slice[0] = normalizing_label[0][..., z]
                metric_normalizing_label = estimate_metric_within_tract(data[..., z], normalizing_label_slice, method, 0)
                # estimate the metric mean in the normalizing label for the slice z
                if metric_normalizing_label[0][0] != 0:
                    data[..., z] = data[..., z] / metric_normalizing_label[0][0]  # divide all the slice z by this value

        elif normalization_method == 'whole':  # case: the user wants to normalize after estimations in the whole labels
            metric_norm_label, metric_std_norm_label = estimate_metric_within_tract(data, normalizing_label, method, param_default.verbose)  # mean and std are lists

    # extract metrics within labels
    sct.log.info('Estimate metric within labels...')
    metric_in_labels, metric_std_in_labels = estimate_metric_within_tract(data, labels, method, verbose, clustered_labels, matching_cluster_labels, adv_param, im_weight)  # mean and std are lists

    if normalizing_label and normalization_method == 'whole':  # case: user wants to normalize after estimations in the whole labels
        metric_in_labels, metric_std_in_labels = np.divide(metric_in_labels, metric_norm_label), np.divide(metric_std_in_labels, metric_std_norm_label)

    if combined_labels_id_group:
        metric_in_labels = np.asarray([metric_in_labels[0]])
        metric_std_in_labels = np.asarray([metric_std_in_labels[0]])

    # compute fractional volume for each label
    fract_vol_per_label = np.zeros(metric_in_labels.size, dtype=float)
    for i_label in range(0, metric_in_labels.size):
        fract_vol_per_label[i_label] = np.sum(labels[i_label])

    return metric_in_labels, metric_std_in_labels, fract_vol_per_label


def get_clustered_labels(clusters_all_labels, labels, indiv_labels_ids, labels_user, averaging_flag, verbose):
    """
    Cluster labels according to selected options (labels and averaging).
    :ml_clusters: clusters in form: '0:29,30,31'
    :labels: all labels data
    :labels_user: label IDs selected by the user
    :averaging_flag: flag -a (0 or 1)
    :return: clustered_labels: labels summed by clustered
    """

    nb_clusters = len(clusters_all_labels)

    # find matching between labels and clusters in the label id list selected by the user
    matching_cluster_label_id_user = np.zeros(len(labels_user), dtype=int)
    for i_label in range(0, len(labels_user)):
        for i_cluster in range(0, nb_clusters):
            if labels_user[i_label] in clusters_all_labels[i_cluster]:
                matching_cluster_label_id_user[i_label] = i_cluster

    # reorganize the cluster according to the averaging flag chosen
    if averaging_flag:
        matching_cluster_label_id_unique = np.unique(matching_cluster_label_id_user)
        if matching_cluster_label_id_unique.size != 1:
            merged_cluster = []
            for i_cluster in matching_cluster_label_id_unique:
                merged_cluster = merged_cluster + clusters_all_labels[i_cluster]
            clusters_all_labels = list(np.delete(np.asarray(clusters_all_labels), matching_cluster_label_id_unique))
            clusters_all_labels.insert(matching_cluster_label_id_unique[0], merged_cluster)
    nb_clusters = len(clusters_all_labels)
    sct.log.info('Number of clusters: ' + str(nb_clusters))

    # sum labels within each cluster
    clustered_labels = np.empty([nb_clusters], dtype=object)  # labels(nb_labels_total, x, y, z)
    for i_cluster in range(0, nb_clusters):
        indexes_labels_cluster_i = [indiv_labels_ids.index(label_ID) for label_ID in clusters_all_labels[i_cluster]]
        clustered_labels[i_cluster] = np.sum(labels[indexes_labels_cluster_i])

    # find matching between labels and clusters in the whole label id list
    matching_cluster_label_id = np.zeros(len(labels), dtype=int)
    for i_label in range(0, len(labels)):
        for i_cluster in range(0, nb_clusters):
            if i_label in clusters_all_labels[i_cluster]:
                matching_cluster_label_id[i_label] = i_cluster
    if averaging_flag:
        cluster_averaged_labels = matching_cluster_label_id[labels_user]
        matching_cluster_label_id = list(np.delete(np.asarray(matching_cluster_label_id), labels_user))
        matching_cluster_label_id.insert(0, cluster_averaged_labels[0])  # because the average of labels will be placed in the first position

    return clustered_labels, matching_cluster_label_id


def fix_label_value(label_to_fix, data, labels, indiv_labels_ids, indiv_labels_names, ml_clusters, combined_labels_id_groups, labels_id_user):
    """
    This function updates the data and list of labels as explained in:
    https://github.com/neuropoly/spinalcordtoolbox/issues/958
    :param label_to_fix:
    :param data:
    :param labels:
    :param indiv_labels_ids:
    :param indiv_labels_names:
    :param ml_clusters:
    :param combined_labels_id_groups:
    :param labels_id_user:
    :return:
    """

    label_to_fix_ID = int(label_to_fix[0])
    label_to_fix_value = float(label_to_fix[1])

    # remove the value from the data
    label_to_fix_index = indiv_labels_ids.index(label_to_fix_ID)
    label_to_fix_fract_vol = labels[label_to_fix_index]
    data = data - label_to_fix_fract_vol * label_to_fix_value

    # remove the label to fix from the labels lists
    labels = np.delete(labels, label_to_fix_index, 0)
    del indiv_labels_ids[label_to_fix_index]
    label_to_fix_name = indiv_labels_names[label_to_fix_index]
    del indiv_labels_names[label_to_fix_index]

    # remove the label to fix from the label list specified by user
    if label_to_fix_ID in labels_id_user:
        labels_id_user.remove(label_to_fix_ID)

    # redefine the clusters
    ml_clusters = remove_label_from_group(ml_clusters, label_to_fix_ID)

    # redefine the combined labels groups
    combined_labels_id_groups = remove_label_from_group(combined_labels_id_groups, label_to_fix_ID)

    return data, labels, indiv_labels_ids, indiv_labels_names, ml_clusters, combined_labels_id_groups, labels_id_user, label_to_fix_name, label_to_fix_fract_vol


def remove_label_from_group(list_label_groups, label_ID):
    """Redefine groups of labels after removing one specific label."""

    for i_group in range(len(list_label_groups)):
        if label_ID in list_label_groups[i_group]:
            list_label_groups[i_group].remove(label_ID)

    list_label_groups = list(filter(None, list_label_groups))

    return list_label_groups
