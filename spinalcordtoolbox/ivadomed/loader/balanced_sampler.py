from __future__ import annotations
import torch
import numpy as np
import typing

from typing import Union

if typing.TYPE_CHECKING:
    from ivadomed.loader.bids_dataset import BidsDataset
    from ivadomed.loader.bids3d_dataset import Bids3DDataset


class BalancedSampler(torch.utils.data.sampler.Sampler):
    """Estimate sampling weights in order to rebalance the
    class distributions from an imbalanced dataset.

    Args:
        dataset (BidsDataset): Dataset containing input, gt and metadata.
        metadata (str): Indicates which metadata to use to balance the sampler.

    Attributes:
        indices (list): List from 0 to length of dataset (number of elements in the dataset).
        nb_samples (int): Number of elements in the dataset.
        weights (Tensor): Weight of each dataset element equal to 1 over the frequency of a
            given label (inverse of the frequency).
        metadata_dict (dict): Stores the mapping from metadata string to index (int).
        label_idx (int): Keeps track of the label indices already used for the metadata_dict.
    """

    def __init__(self, dataset: Union[BidsDataset, Bids3DDataset], metadata: str = 'gt') -> None:
        self.indices = list(range(len(dataset)))

        self.nb_samples = len(self.indices)
        self.metadata_dict = {}
        self.label_idx = 0

        cmpt_label = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx, metadata)
            if label in cmpt_label:
                cmpt_label[label] += 1
            else:
                cmpt_label[label] = 1

        weights = [1.0 / cmpt_label[self._get_label(dataset, idx, metadata)]
                   for idx in self.indices]

        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset: Union[BidsDataset, Bids3DDataset], idx: int, metadata: str) -> int:
        """Returns 1 if sample is not empty, 0 if it is empty (only zeros).

        Args:
            dataset (BidsDataset): Dataset containing input, gt and metadata.
            idx (int): Element index.

        Returns:
            int: 0 or 1.
        """
        if metadata != 'gt':
            label_str = dataset[idx]['input_metadata'][0][metadata]
            if label_str not in self.metadata_dict:
                self.metadata_dict[label_str] = self.label_idx
                self.label_idx += 1
            return self.metadata_dict[label_str]

        else:
            # For now, only supported with single label
            sample_gt = np.array(dataset[idx]['gt'][0])
            if np.any(sample_gt):
                return 1
            else:
                return 0

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.nb_samples, replacement=True))

    def __len__(self):
        return self.num_samples
