__author__ = 'slevy_local'



class Color:
    def __init__(self):
        self.purple = '\033[95m'
        self.cyan = '\033[96m'
        self.darkcyan = '\033[36m'
        self.blue = '\033[94m'
        self.green = '\033[92m'
        self.yellow = '\033[93m'
        self.red = '\033[91m'
        self.bold = '\033[1m'
        self.underline = '\033[4m'
        self.end = '\033[0m'

def progress3d(i, j, k, ni, nj, nk):

    import sys
    color = Color()

    progress = str( 100*((k-1)*(nj-1)*(ni-1) + (j-1)*(ni-1) + +i) / ((nk-1)*(nj-1)*(ni-1)) ) +' %'

    if [i, j, k] == [0, 0, 0]:
        sys.stdout.write(progress)
    else:
        sys.stdout.write(color.bold + '\b\b\b\b\b'+ progress + color.end)


def compute_metric_mean_and_std_slice_by_slice(np_data_metric, np_data_mask=[]):
    """
    :param np_data_metric: Metric data loaded as numpy array (must be in RPI orientation)
    :param np_data_mask: Mask data loaded as numpy array
    :return: numpy array (number of slices mask data x 2) containing mean value (first position) and std (second
    position) of the metric in the mask slice by slice
    """

    import numpy

    nz = np_data_metric.shape[2]
    metric_mean_and_std = numpy.zeros((nz, 2))  # will contain metric mean and std within the mask

    for z in range(0, nz):

        if not np_data_mask:
            metric_mean_and_std[z, 0] = numpy.mean(np_data_metric[:, :, z])
            metric_mean_and_std[z, 1] = numpy.std(np_data_metric[:, :, z])
        else:
            mask_slice = np_data_mask[..., z]  # extract slice z from the mask
            ind = numpy.where(mask_slice == 1)

            metric_mean_and_std[z, 0] = numpy.mean(np_data_metric[ind[0], ind[1], z])
            metric_mean_and_std[z, 1] = numpy.std(np_data_metric[ind[0], ind[1], z])

    return metric_mean_and_std


