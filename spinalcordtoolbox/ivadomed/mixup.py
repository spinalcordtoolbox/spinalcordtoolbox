import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path


def mixup(data, targets, alpha, debugging=False, ofolder=None):
    """Compute the mixup data.

    .. seealso::
        Zhang, Hongyi, et al. "mixup: Beyond empirical risk minimization."
        arXiv preprint arXiv:1710.09412 (2017).

    Args:
        data (Tensor): Input images.
        targets (Tensor): Input masks.
        alpha (float): MixUp parameter.
        debugging (Bool): If True, then samples of mixup are saved as png files.
        ofolder (str): If debugging, output folder where "mixup" folder is created and samples are saved.

    Returns:
        Tensor, Tensor: Mixed image, Mixed mask.
    """
    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]

    lambda_ = np.random.beta(alpha, alpha)
    lambda_ = max(lambda_, 1 - lambda_)  # ensure lambda_ >= 0.5
    lambda_tensor = torch.FloatTensor([lambda_]).to(data.device)

    data = data * lambda_tensor + data2 * (1 - lambda_tensor)
    targets = targets * lambda_tensor + targets2 * (1 - lambda_tensor)

    if debugging:
        save_mixup_sample(ofolder, data, targets, lambda_tensor)

    return data, targets


def save_mixup_sample(ofolder, input_data, labeled_data, lambda_tensor):
    """Save mixup samples as png files in a "mixup" folder.

    Args:
        ofolder (str): Output folder where "mixup" folder is created and samples are saved.
        input_data (Tensor): Input image.
        labeled_data (Tensor): Input masks.
        lambda_tensor (Tensor):
    """
    # Mixup folder
    mixup_folder = Path(ofolder, 'mixup')
    if not mixup_folder.is_dir():
        mixup_folder.mkdir(parents=True)
    # Random sample
    random_idx = np.random.randint(0, input_data.size()[0])
    # Output fname
    ofname = str(lambda_tensor.cpu().data.numpy()[0]) + '_' + str(random_idx).zfill(3) + '.png'
    ofname = Path(mixup_folder, ofname)
    # Tensor to Numpy
    x = input_data.cpu().data.numpy()[random_idx, 0, :, :]
    y = labeled_data.cpu().data.numpy()[random_idx, 0, :, :]
    # Plot
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(x, interpolation='nearest', aspect='auto', cmap='gray')
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(y, interpolation='nearest', aspect='auto', cmap='jet', vmin=0, vmax=1)
    plt.savefig(ofname, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()
