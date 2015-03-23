__author__ = 'slevy_local'



def compute_metric_mean_and_std_slice_by_slice(np_data_metric, np_data_mask):
    """
    :param np_data_metric: Metric data loaded as numpy array (must be in RPI orientation)
    :param np_data_mask: Mask data loaded as numpy array
    :return: numpy array (number of slices mask data x 2) containing mean value (first position) and std (second
    position) of the metric in the mask slice by slice
    """

    import numpy

    nz = np_data_mask.shape[2]
    metric_mean_and_std = numpy.zeros((nz, 2))  # will contain metric mean and std within the mask

    for z in range(0, nz):
        mask_slice = np_data_mask[..., z]  # extract slice z from the mask
        ind = numpy.where(mask_slice == 1)

        metric_mean_and_std[z, 0] = numpy.mean(np_data_metric[ind[0], ind[1], z])
        metric_mean_and_std[z, 1] = numpy.std(np_data_metric[ind[0], ind[1], z])

    return metric_mean_and_std
