from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial


# METRICS
def get_metric_fns(task, eval_params=None):
    metric_fns = [dice_score,
                  multi_class_dice_score,
                  precision_score,
                  recall_score,
                  specificity_score,
                  intersection_over_union,
                  accuracy_score]
    if eval_params:
        object_detection_metrics = eval_params['object_detection_metrics']
    else:
        object_detection_metrics = True
    if task == "segmentation" and object_detection_metrics:
        metric_fns = metric_fns + [hausdorff_score]

    return metric_fns


class MetricManager(object):
    """Computes specified metrics and stores them in a dictionary.

    Args:
        metric_fns (list): List of metric functions.

    Attributes:
        metric_fns (list): List of metric functions.
        result_dict (dict): Dictionary storing metrics.
        num_samples (int): Number of samples.
    """

    def __init__(self, metric_fns):
        self.metric_fns = metric_fns
        self.num_samples = 0
        self.result_dict = defaultdict(list)

    def __call__(self, prediction, ground_truth):
        self.num_samples += len(prediction)
        for metric_fn in self.metric_fns:
            for p, gt in zip(prediction, ground_truth):
                res = metric_fn(p, gt)
                dict_key = metric_fn.__name__
                self.result_dict[dict_key].append(res)

    def get_results(self):
        res_dict = {}
        for key, val in self.result_dict.items():
            if np.all(np.isnan(val)):  # if all values are np.nan
                res_dict[key] = None
            else:
                res_dict[key] = np.nanmean(val)
        return res_dict

    def reset(self):
        self.num_samples = 0
        self.result_dict = defaultdict(float)


def numeric_score(prediction, groundtruth):
    """Computation of statistical numerical scores:

    * FP = Soft False Positives
    * FN = Soft False Negatives
    * TP = Soft True Positives
    * TN = Soft True Negatives

    Robust to hard or soft input masks. For example::
        prediction=np.asarray([0, 0.5, 1])
        groundtruth=np.asarray([0, 1, 1])
        Leads to FP = 1.5

    Note: It assumes input values are between 0 and 1.

    Args:
        prediction (ndarray): Binary prediction.
        groundtruth (ndarray): Binary groundtruth.

    Returns:
        float, float, float, float: FP, FN, TP, TN
    """
    FP = float(np.sum(prediction * (1.0 - groundtruth)))
    FN = float(np.sum((1.0 - prediction) * groundtruth))
    TP = float(np.sum(prediction * groundtruth))
    TN = float(np.sum((1.0 - prediction) * (1.0 - groundtruth)))
    return FP, FN, TP, TN


def dice_score(im1, im2, empty_score=np.nan):
    """Computes the Dice coefficient between im1 and im2.

    Compute a soft Dice coefficient between im1 and im2, ie equals twice the sum of the two masks product, divided by
    the sum of each mask sum.
    If both images are empty, then it returns empty_score.

    Args:
        im1 (ndarray): First array.
        im2 (ndarray): Second array.
        empty_score (float): Returned value if both input array are empty.

    Returns:
        float: Dice coefficient.
    """
    im1 = np.asarray(im1)
    im2 = np.asarray(im2)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = (im1 * im2).sum()
    return (2. * intersection) / im_sum


def mse(im1, im2):
    """Compute the Mean Squared Error.

    Compute the Mean Squared Error between the two images, i.e. sum of the squared difference.

    Args:
        im1 (ndarray): First array.
        im2 (ndarray): Second array.

    Returns:
        float: Mean Squared Error.
    """
    im1 = np.asarray(im1)
    im2 = np.asarray(im2)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    err = np.sum((im1.astype("float") - im2.astype("float")) ** 2)
    err /= float(im1.shape[0] * im1.shape[1])

    return err


def hausdorff_score(prediction, groundtruth):
    """Compute the directed Hausdorff distance between two N-D arrays.

    Args:
        prediction (ndarray): First array.
        groundtruth (ndarray): Second array.

    Returns:
        float: Hausdorff distance.
    """
    if len(prediction.shape) == 4:
        n_classes, height, depth, width = prediction.shape
        # Reshape to have only 3 dimensions where prediction[:, idx, :] represents each 2D slice
        prediction = prediction.reshape((height, n_classes * depth, width))
        groundtruth = groundtruth.reshape((height, n_classes * depth, width))

    if len(prediction.shape) == 3:
        mean_hansdorff = 0
        for idx in range(prediction.shape[1]):
            pred = prediction[:, idx, :]
            gt = groundtruth[:, idx, :]
            mean_hansdorff += spatial.distance.directed_hausdorff(pred, gt)[0]
        mean_hansdorff = mean_hansdorff / prediction.shape[1]
        return mean_hansdorff

    return spatial.distance.directed_hausdorff(prediction, groundtruth)[0]


def precision_score(prediction, groundtruth, err_value=0.0):
    """Positive predictive value (PPV).

    Precision equals the number of true positive voxels divided by the sum of true and false positive voxels.
    True and false positives are computed on soft masks, see ``"numeric_score"``.

    Args:
        prediction (ndarray): First array.
        groundtruth (ndarray): Second array.
        err_value (float): Value returned in case of error.

    Returns:
        float: Precision score.
    """
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FP) <= 0.0:
        return err_value

    precision = np.divide(TP, TP + FP)
    return precision


def recall_score(prediction, groundtruth, err_value=0.0):
    """True positive rate (TPR).

    Recall equals the number of true positive voxels divided by the sum of true positive and false negative voxels.
    True positive and false negative values are computed on soft masks, see ``"numeric_score"``.

    Args:
        prediction (ndarray): First array.
        groundtruth (ndarray): Second array.
        err_value (float): Value returned in case of error.

    Returns:
        float: Recall score.
    """
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FN) <= 0.0:
        return err_value
    TPR = np.divide(TP, TP + FN)
    return TPR


def specificity_score(prediction, groundtruth, err_value=0.0):
    """True negative rate (TNR).

    Specificity equals the number of true negative voxels divided by the sum of true negative and false positive voxels.
    True negative and false positive values are computed on soft masks, see ``"numeric_score"``.

    Args:
        prediction (ndarray): First array.
        groundtruth (ndarray): Second array.
        err_value (float): Value returned in case of error.

    Returns:
        float: Specificity score.
    """
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TN + FP) <= 0.0:
        return err_value
    TNR = np.divide(TN, TN + FP)
    return TNR


def intersection_over_union(prediction, groundtruth, err_value=0.0):
    """Intersection of two (soft) arrays over their union (IoU).

    Args:
        prediction (ndarray): First array.
        groundtruth (ndarray): Second array.
        err_value (float): Value returned in case of error.

    Returns:
        float: IoU.
    """
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FP + FN) <= 0.0:
        return err_value
    return TP / (TP + FP + FN)


def accuracy_score(prediction, groundtruth, err_value=0.0):
    """Accuracy.

    Accuracy equals the number of true positive and true negative voxels divided by the total number of voxels.
    True positive/negative and false positive/negative values are computed on soft masks, see ``"numeric_score"``.

    Args:
        prediction (ndarray): First array.
        groundtruth (ndarray): Second array.

    Returns:
        float: Accuracy.
    """
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N = FP + FN + TP + TN
    if N <= 0.0:
        return err_value
    accuracy = np.divide(TP + TN, N)
    return accuracy


def multi_class_dice_score(im1, im2):
    """Dice score for multi-label images.

    Multi-class Dice score equals the average of the Dice score for each class.
    The first dimension of the input arrays is assumed to represent the classes.

    Args:
        im1 (ndarray): First array.
        im2 (ndarray): Second array.

    Returns:
        float: Multi-class dice.
    """
    dice_per_class = 0
    n_classes = im1.shape[0]

    for i in range(n_classes):
        dice_per_class += dice_score(im1[i,], im2[i,], empty_score=1.0)

    return dice_per_class / n_classes


def plot_roc_curve(tpr, fpr, opt_thr_idx, fname_out):
    """Plot ROC curve.

    Args:
        tpr (list): True positive rates.
        fpr (list): False positive rates.
        opt_thr_idx (int): Index of the optimal threshold.
        fname_out (str): Output filename.
    """
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, marker='o')
    plt.plot([fpr[opt_thr_idx]], [tpr[opt_thr_idx]], color="darkgreen", marker="o", linestyle="None")
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.savefig(fname_out)


def plot_dice_thr(thr_list, dice_list, opt_thr_idx, fname_out):
    """Plot Dice results against thresholds.

    Args:
        thr_list (list): Thresholds list.
        dice_list (list): Dice results.
        opt_thr_idx (int): Index of the optimal threshold.
        fname_out (str): Output filename.
    """
    plt.figure()
    lw = 2
    plt.plot(thr_list, dice_list, color='darkorange', lw=lw, marker='o')
    plt.plot([thr_list[opt_thr_idx]], [dice_list[opt_thr_idx]], color="darkgreen", marker="o", linestyle="None")
    plt.xlim([0.0, 1.0])
    plt.ylim([min(dice_list) - 0.02, max(dice_list) + 0.02])
    plt.xlabel('Thresholds')
    plt.ylabel('Dice')
    plt.title('Threshold analysis')
    plt.savefig(fname_out)
