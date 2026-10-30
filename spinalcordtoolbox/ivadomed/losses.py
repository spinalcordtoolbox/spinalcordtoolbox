import torch
import torch.nn as nn
import scipy
import scipy.ndimage
import numpy as np


class MultiClassDiceLoss(nn.Module):
    """Multi-class Dice Loss.

    Inspired from https://arxiv.org/pdf/1802.10508.

    Args:
        classes_of_interest (list): List containing the index of a class which its dice will be added to the loss.
            If is None all classes are considered.

    Attributes:
        classes_of_interest (list): List containing the index of a class which its dice will be added to the loss.
            If is None all classes are considered.
        dice_loss (DiceLoss): Class computing the Dice loss.
    """

    def __init__(self, classes_of_interest=None):
        super(MultiClassDiceLoss, self).__init__()
        self.classes_of_interest = classes_of_interest
        self.dice_loss = DiceLoss()

    def forward(self, prediction, target):
        dice_per_class = 0
        n_classes = prediction.shape[1]

        if self.classes_of_interest is None:
            self.classes_of_interest = range(n_classes)

        for i in self.classes_of_interest:
            dice_per_class += self.dice_loss(prediction[:, i, ], target[:, i, ])

        return dice_per_class / len(self.classes_of_interest)


class DiceLoss(nn.Module):
    """DiceLoss.

    .. seealso::
        Milletari, Fausto, Nassir Navab, and Seyed-Ahmad Ahmadi. "V-net: Fully convolutional neural networks for
        volumetric medical image segmentation." 2016 fourth international conference on 3D vision (3DV). IEEE, 2016.

    Args:
        smooth (float): Value to avoid division by zero when images and predictions are empty.

    Attributes:
        smooth (float): Value to avoid division by zero when images and predictions are empty.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        iflat = prediction.reshape(-1)
        tflat = target.reshape(-1)
        intersection = (iflat * tflat).sum()

        return - (2.0 * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth)


class BinaryCrossEntropyLoss(nn.Module):
    """(`BinaryCrossEntropyLoss <https://pytorch.org/docs/master/generated/torch.nn.BCELoss.html#bceloss>`__).

    Attributes:
        loss_fct (BCELoss): Binary cross entropy loss function from torch library.
    """

    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.loss_fct = nn.BCELoss()

    def forward(self, prediction, target):
        return self.loss_fct(prediction, target.float())


class FocalLoss(nn.Module):
    """FocalLoss.

    .. seealso::
        Lin, Tsung-Yi, et al. "Focal loss for dense object detection."
        Proceedings of the IEEE international conference on computer vision. 2017.

    Args:
        gamma (float): Value from 0 to 5, Control between easy background and hard ROI
            training examples. If set to 0, equivalent to cross-entropy.
        alpha (float): Value from 0 to 1, usually corresponding to the inverse of class frequency to address class
            imbalance.
        eps (float): Epsilon to avoid division by zero.

    Attributes:
        gamma (float): Value from 0 to 5, Control between easy background and hard ROI
            training examples. If set to 0, equivalent to cross-entropy.
        alpha (float): Value from 0 to 1, usually corresponding to the inverse of class frequency to address class
            imbalance.
        eps (float): Epsilon to avoid division by zero.
    """
    def __init__(self, gamma=2, alpha=0.25, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, input, target):
        input = input.clamp(self.eps, 1. - self.eps)

        cross_entropy = - (target * torch.log(input) + (1 - target) * torch.log(1 - input))  # eq1
        logpt = - cross_entropy
        pt = torch.exp(logpt)  # eq2

        at = self.alpha * target + (1 - self.alpha) * (1 - target)
        balanced_cross_entropy = - at * logpt  # eq3

        focal_loss = balanced_cross_entropy * ((1 - pt) ** self.gamma)  # eq5

        return focal_loss.sum()
        # return focal_loss.mean()


class FocalDiceLoss(nn.Module):
    """FocalDiceLoss.

    .. seealso::
        Wong, Ken CL, et al. "3D segmentation with exponential logarithmic loss for highly unbalanced object sizes."
        International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2018.

    Args:
        beta (float): Value from 0 to 1, indicating the weight of the dice loss.
        gamma (float): Value from 0 to 5, Control between easy background and hard ROI
            training examples. If set to 0, equivalent to cross-entropy.
        alpha (float): Value from 0 to 1, usually corresponding to the inverse of class frequency to address class
            imbalance.

    Attributes:
        beta (float): Value from 0 to 1, indicating the weight of the dice loss.
        gamma (float): Value from 0 to 5, Control between easy background and hard ROI
            training examples. If set to 0, equivalent to cross-entropy.
        alpha (float): Value from 0 to 1, usually corresponding to the inverse of class frequency to address class
            imbalance.
    """

    def __init__(self, beta=1, gamma=2, alpha=0.25):
        super().__init__()
        self.beta = beta
        self.focal = FocalLoss(gamma, alpha)
        self.dice = DiceLoss()

    def forward(self, input, target):
        dc_loss = - self.dice(input, target)
        fc_loss = self.focal(input, target)

        # used to fine tune beta
        # with torch.no_grad():
        #     print('DICE loss:', dc_loss.cpu().numpy(), 'Focal loss:', fc_loss.cpu().numpy())
        #     log_dc_loss = torch.log(torch.clamp(dc_loss, 1e-7))
        #     log_fc_loss = torch.log(torch.clamp(fc_loss, 1e-7))
        #     print('Log DICE loss:', log_dc_loss.cpu().numpy(), 'Log Focal loss:', log_fc_loss.cpu().numpy())
        #     print('*'*20)

        loss = torch.log(torch.clamp(fc_loss, 1e-7)) - self.beta * torch.log(torch.clamp(dc_loss, 1e-7))

        return loss


class GeneralizedDiceLoss(nn.Module):
    """GeneralizedDiceLoss.

    .. seealso::
        Sudre, Carole H., et al. "Generalised dice overlap as a deep learning loss function for highly unbalanced
        segmentations." Deep learning in medical image analysis and multimodal learning for clinical decision support.
        Springer, Cham, 2017. 240-248.

    Args:
        epsilon (float): Epsilon to avoid division by zero.
        include_background (bool): If True, then an extra channel is added, which represents the background class.

    Attributes:
        epsilon (float): Epsilon to avoid division by zero.
        include_background (bool): If True, then an extra channel is added, which represents the background class.
    """

    def __init__(self, epsilon=1e-5, include_background=True):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.include_background = include_background

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        if self.include_background:
            # init
            size_background = [input.size(0), 1] + list(input.size())[2:]
            input_background = torch.zeros(size_background, dtype=input.dtype)
            target_background = torch.zeros(size_background, dtype=target.dtype)
            # fill with opposite
            input_background[input.sum(1)[:, None, :, :] == 0] = 1
            target_background[target.sum(1)[:, None, :, :] == 0] = 1
            # Concat
            input = torch.cat([input, input_background.to(input.device)], dim=1)
            target = torch.cat([target, target_background.to(target.device)], dim=1)

        # Compute class weights
        target = target.float()
        axes_to_sum = tuple(range(2, len(target.shape)))
        target_sum = target.sum(axis=axes_to_sum)
        class_weights = nn.Parameter(1. / (target_sum * target_sum).clamp(min=self.epsilon))
        # W Intersection
        intersect = ((input * target).sum(axis=axes_to_sum) * class_weights).sum()

        # W Union
        denominator = ((input + target).sum(axis=axes_to_sum) * class_weights).sum()

        return - 2. * intersect / denominator.clamp(min=self.epsilon)


class TverskyLoss(nn.Module):
    """Tversky Loss.

    .. seealso::
        Salehi, Seyed Sadegh Mohseni, Deniz Erdogmus, and Ali Gholipour. "Tversky loss function for image segmentation
        using 3D fully convolutional deep networks." International Workshop on Machine Learning in Medical Imaging.
        Springer, Cham, 2017.

    Args:
        alpha (float): Weight of false positive voxels.
        beta  (float): Weight of false negative voxels.
        smooth (float): Epsilon to avoid division by zero, when both Numerator and Denominator of Tversky are zeros.

    Attributes:
        alpha (float): Weight of false positive voxels.
        beta  (float): Weight of false negative voxels.
        smooth (float): Epsilon to avoid division by zero, when both Numerator and Denominator of Tversky are zeros.

    Notes:
        - setting alpha=beta=0.5: Equivalent to DiceLoss.
        - default parameters were suggested by https://arxiv.org/pdf/1706.05721.pdf .
    """

    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def tversky_index(self, y_pred, y_true):
        """Compute Tversky index.

        Args:
            y_pred (torch Tensor): Prediction.
            y_true (torch Tensor): Target.

        Returns:
            float: Tversky index.
        """
        # Compute TP
        y_true = y_true.float()
        tp = torch.sum(y_true * y_pred)
        # Compute FN
        fn = torch.sum(y_true * (1 - y_pred))
        # Compute FP
        fp = torch.sum((1 - y_true) * y_pred)
        # Compute Tversky for the current class, see Equation 3 of the original paper
        numerator = tp + self.smooth
        denominator = tp + self.alpha * fp + self.beta * fn + self.smooth
        tversky_label = numerator / denominator
        return tversky_label

    def forward(self, input, target):
        n_classes = input.shape[1]
        tversky_sum = 0.

        # TODO: Add class_of_interest?
        for i_label in range(n_classes):
            # Get samples for a given class
            y_pred, y_true = input[:, i_label, ], target[:, i_label, ]
            # Compute Tversky index
            tversky_sum += self.tversky_index(y_pred, y_true)

        return - tversky_sum / n_classes


class FocalTverskyLoss(TverskyLoss):
    """Focal Tversky Loss.

    .. seealso::
        Abraham, Nabila, and Naimul Mefraz Khan. "A novel focal tversky loss function with improved attention u-net for
        lesion segmentation." 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019). IEEE, 2019.

    Args:
        alpha (float): Weight of false positive voxels.
        beta  (float): Weight of false negative voxels.
        gamma (float): Typically between 1 and 3. Control between easy background and hard ROI training examples.
        smooth (float): Epsilon to avoid division by zero, when both Numerator and Denominator of Tversky are zeros.

    Attributes:
        gamma (float): Typically between 1 and 3. Control between easy background and hard ROI training examples.

    Notes:
        - setting alpha=beta=0.5 and gamma=1: Equivalent to DiceLoss.
        - default parameters were suggested by https://arxiv.org/pdf/1810.07842.pdf .
    """

    def __init__(self, alpha=0.7, beta=0.3, gamma=1.33, smooth=1.0):
        super(FocalTverskyLoss, self).__init__()
        self.gamma = gamma
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)

    def forward(self, input, target):
        n_classes = input.shape[1]
        focal_tversky_sum = 0.

        # TODO: Add class_of_interest?
        for i_label in range(n_classes):
            # Get samples for a given class
            y_pred, y_true = input[:, i_label, ], target[:, i_label, ]
            # Compute Tversky index
            tversky_index = self.tversky.tversky_index(y_pred, y_true)
            # Compute Focal Tversky loss, Equation 4 in the original paper
            focal_tversky_sum += torch.pow(1 - tversky_index, exponent=1 / self.gamma)

        return focal_tversky_sum / n_classes


class L2loss(nn.Module):
    """
    Euclidean loss also known as L2 loss. Compute the sum of the squared difference between the two images.
    """

    def __init__(self):
        super(L2loss, self).__init__()

    def forward(self, input, target):
        return torch.sum((input - target) ** 2) / 2


class AdapWingLoss(nn.Module):
    """
    Adaptive Wing loss
    Used for heatmap ground truth.

    .. seealso::
        Wang, Xinyao, Liefeng Bo, and Li Fuxin. "Adaptive wing loss for robust face alignment via heatmap regression."
        Proceedings of the IEEE International Conference on Computer Vision. 2019.

    Args:
        theta (float): Threshold between linear and non linear loss.
        alpha (float): Used to adapt loss shape to input shape and make loss smooth at 0 (background).
        It needs to be slightly above 2 to maintain ideal properties.
        omega (float): Multiplicating factor for non linear part of the loss.
        epsilon (float): factor to avoid gradient explosion. It must not be too small
    """

    def __init__(self, theta=0.5, alpha=2.1, omega=14, epsilon=1):
        self.theta = theta
        self.alpha = alpha
        self.omega = omega
        self.epsilon = epsilon
        super(AdapWingLoss, self).__init__()

    def forward(self, input, target):
        eps = self.epsilon
        # Compute adaptative factor
        A = self.omega * (1 / (1 + torch.pow(self.theta / eps,
                                             self.alpha - target))) * \
            (self.alpha - target) * torch.pow(self.theta / eps,
                                              self.alpha - target - 1) * (1 / eps)

        # Constant term to link linear and non linear part
        C = (self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / eps, self.alpha - target)))

        batch_size = target.size()[0]
        hm_num = target.size()[1]

        mask = torch.zeros_like(target)
        kernel = scipy.ndimage.generate_binary_structure(2, 2)
        # For 3D segmentation tasks
        if len(input.shape) == 5:
            kernel = scipy.ndimage.generate_binary_structure(3, 2)

        for i in range(batch_size):
            img_list = list()
            img_list.append(np.round(target[i].cpu().numpy() * 255))
            img_merge = np.concatenate(img_list)
            img_dilate = scipy.ndimage.binary_opening(img_merge, np.expand_dims(kernel, axis=0))
            img_dilate[img_dilate < 51] = 1  # 0*omega+1
            img_dilate[img_dilate >= 51] = 1 + self.omega  # 1*omega+1
            img_dilate = np.array(img_dilate, dtype=int)

            mask[i] = torch.tensor(img_dilate)

        diff_hm = torch.abs(target - input)
        AWingLoss = A * diff_hm - C
        idx = diff_hm < self.theta
        AWingLoss[idx] = self.omega * torch.log(1 + torch.pow(diff_hm / eps, self.alpha - target))[idx]

        AWingLoss *= mask
        sum_loss = torch.sum(AWingLoss)
        all_pixel = torch.sum(mask)
        mean_loss = sum_loss  # / all_pixel

        return mean_loss


class LossCombination(nn.Module):
    """
    Loss that sums other implemented losses.

    Args:
        losses_list (list): list of losses that will be summed. Elements should be string.
        params_list (list): list of params for the losses, contain None or dictionnary definition of params for the loss
        at same index. If no params list is given all default parameter will be used.
        (e.g., losses_list = ["L2loss","DiceLoss"]
               params_list = [None,{"param1:0.5"}])

    returns:
            tensor: sum of losses computed on (input,target) with the params
    """

    def __init__(self, losses_list, params_list=None):
        self.losses_list = losses_list
        self.params_list = params_list
        super(LossCombination, self).__init__()

    def forward(self, input, target):
        output = []
        for i in range(len(self.losses_list)):
            loss_class = eval(self.losses_list[i])
            if self.params_list is not None:
                if self.params_list[i] is not None:
                    loss_fct = loss_class(**self.params_list[i])
                else:
                    loss_fct = loss_class()
                output.append(loss_fct(input, target).unsqueeze(0))
            else:
                output.append(loss_class()(input, target).unsqueeze(0))

        return torch.sum(torch.cat(output))
