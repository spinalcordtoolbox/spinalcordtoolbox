from __future__ import annotations
import collections.abc
import re
import sys
import os
import joblib
import gc
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from ivadomed import utils as imed_utils
from ivadomed.keywords import SplitDatasetKW, LoaderParamsKW, ROIParamsKW, ContrastParamsKW
import nibabel as nib
import random
import typing
if typing.TYPE_CHECKING:
    from typing import Union
    from typing import Optional

__numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}

TRANSFORM_PARAMS = ['elastic', 'rotation', 'scale', 'offset', 'crop_params', 'reverse',
                    'translation', 'gaussian_noise']

# Ordered list of supported file extensions
# TODO: Implement support of the following OMETIFF formats (#739):
# [".ome.tif", ".ome.tiff", ".ome.tf2", ".ome.tf8", ".ome.btf"]
# They are included in the list to avoid a ".ome.tif" or ".ome.tiff" following the ".tif" or ".tiff" pipeline
EXT_LST = [".nii", ".nii.gz", ".ome.tif", ".ome.tiff", ".ome.tf2", ".ome.tf8", ".ome.btf", ".tif",
           ".tiff", ".png", ".jpg", ".jpeg"]


def split_dataset(df: pd.DataFrame, split_method: str, data_testing: dict, random_seed: int, train_frac: float = 0.8,
                  test_frac: float = 0.1) -> (list, list, Union[list, object]):
    """Splits dataset into training, validation and testing sets by applying train, test and validation fractions
    according to the split_method.
    The "data_testing" parameter can be used to specify the data_type and data_value to include in the testing set,
    the dataset is then split as not to mix the data_testing between the training/validation set and the testing set.

    Args:
        df (pd.DataFrame): Dataframe containing all BIDS image files indexed and their metadata.
        split_method (str): Used to specify on which metadata to split the dataset (eg. "participant_id", "sample_id", etc.)
        data_testing (dict): Used to specify data_type and data_value to include in the testing set.
        random_seed (int): Random seed to ensure reproducible splits.
        train_frac (float): Between 0 and 1. Represents the train set proportion.
        test_frac (float): Between 0 and 1. Represents the test set proportion.
    Returns:
        list, list, list: Train, validation and test filenames lists.
    """

    # Get data_type and data_value from split parameters
    # If no data_type is provided, data_type is the same as split_method
    data_type = data_testing['data_type'] if data_testing['data_type'] else split_method
    data_value = data_testing['data_value']

    if not split_method in df:
        raise KeyError("No split_method '{}' was not found in metadata".format(split_method))
    if not data_type in df:
        logger.warning("No data_type named '{}' was found in metadata. Not taken into account "
                       "to split the dataset.".format(data_type))
        data_type = split_method

    # Filter dataframe with rows where split_method is not NAN
    df = df[df[split_method].notna()]

    # If no data_value list is provided, create a random data_value according to data_type and test_fraction
    # Split the TEST and remainder set using sklearn function
    if len(data_value) == 0 and test_frac != 0:
        data_value = sorted(df[data_type].unique().tolist())
        test_frac = test_frac if test_frac >= 1 / len(data_value) else 1 / len(data_value)
        data_value, _ = train_test_split(data_value, train_size=test_frac, random_state=random_seed)
    if len(data_value) != 0:
        for value in data_value:
            if value not in df[data_type].values:
                    logger.warning("No data_value '{}' was found in '{}'. Not taken into account "
                                   "to split the dataset.".format(value, data_type))
    X_test = df[df[data_type].isin(data_value)]['filename'].unique().tolist()
    X_remain = df[~df[data_type].isin(data_value)][split_method].unique().tolist()

    # List dataset unique values according to split_method
    # Update train fraction to apply to X_remain
    data = sorted(df[split_method].unique().tolist())
    train_frac_update = train_frac * len(data) / len(X_remain)
    if ((train_frac_update > (1 - 1 / len(X_remain)) and len(X_remain) < 2) or train_frac_update > 1):
        raise RuntimeError("{}/{} '{}' remaining for training and validation sets, train_fraction {} is too large, "
                           "validation set would be empty.".format(len(X_remain), len(data), split_method, train_frac))

    # Split remainder in TRAIN and VALID sets according to train_frac_update using sklearn function
    X_train, X_val = train_test_split(X_remain, train_size=train_frac_update, random_state=random_seed)

    # Print the real train, validation and test fractions after splitting
    real_train_frac = len(X_train)/len(data)
    real_valid_frac = len(X_val)/len(data)
    real_test_frac = 1 - real_train_frac - real_valid_frac
    logger.warning("After splitting: train, validation and test fractions are respectively {}, {} and {}"
                   " of {}.".format(round(real_train_frac, 3), round(real_valid_frac, 3),
                   round(real_test_frac, 3), split_method))

    # Convert train and valid sets from list of "split_method" to list of "filename"
    X_train = df[df[split_method].isin(X_train)]['filename'].unique().tolist()
    X_val = df[df[split_method].isin(X_val)]['filename'].unique().tolist()

    # Make sure that test dataset is unseen during training
    # (in cases where there are multiple "data_type" for a same "split_method")
    X_train = list(set(X_train) - set(X_test))
    X_val = list(set(X_val) - set(X_test))

    return X_train, X_val, X_test


def get_new_subject_file_split(df: pd.DataFrame, split_method: str, data_testing: dict, random_seed: int,
                               train_frac: float, test_frac: float, path_output: str, balance: str,
                               subject_selection: dict = None) -> (list, list, list):
    """Randomly split dataset between training / validation / testing.

    Randomly split dataset between training / validation / testing\
        and save it in path_output + "/split_datasets.joblib".

    Args:
        df (pd.DataFrame): Dataframe containing all BIDS image files indexed and their metadata.
        split_method (str): Used to specify on which metadata to split the dataset (eg. "participant_id", "sample_id", etc.)
        data_testing (dict): Used to specify the data_type and data_value to include in the testing set.
        random_seed (int): Random seed.
        train_frac (float): Training dataset proportion, between 0 and 1.
        test_frac (float): Testing dataset proportionm between 0 and 1.
        path_output (str): Output folder.
        balance (str): Metadata contained in "participants.tsv" file with categorical values. Each category will be
        evenly distributed in the training, validation and testing datasets.
        subject_selection (dict): Used to specify a custom subject selection from a dataset.

    Returns:
        list, list list: Training, validation and testing filenames lists.
    """
    if subject_selection is not None:
        # Verify subject_selection format
        if not (len(subject_selection["metadata"]) == len(subject_selection["n"]) == len(subject_selection["value"])):
            raise ValueError("All lists in subject_selection parameter should have the same length.")

        sampled_dfs = []
        random.seed(random_seed)
        for m, n, v in zip(subject_selection["metadata"], subject_selection["n"], subject_selection["value"]):
            participants = random.sample(df[df[m] == v]['participant_id'].unique().tolist(), n)
            for participant in participants:
                sampled_dfs.append(df[df['participant_id'] == participant])

        if len(sampled_dfs) != 0:
            df = pd.concat(sampled_dfs)

    # If balance, then split the dataframe for each categorical value of the "balance" column
    if balance:
        if balance in df.keys():
            df_list = [df[df[balance] == k] for k in df[balance][df[balance].notna()].unique().tolist()]
        else:
            logger.warning("No column named '{}' was found in 'participants.tsv' file. Not taken into account to split "
                           "the dataset.".format(balance))
            df_list = [df]
    else:
        df_list = [df]

    train_lst, valid_lst, test_lst = [], [], []
    for df_tmp in df_list:
        # Split dataset on each section of subjects
        train_tmp, valid_tmp, test_tmp = split_dataset(df=df_tmp,
                                                       split_method=split_method,
                                                       data_testing=data_testing,
                                                       random_seed=random_seed,
                                                       train_frac=train_frac,
                                                       test_frac=test_frac)
        # Update the dataset lists
        train_lst += train_tmp
        valid_lst += valid_tmp
        test_lst += test_tmp

    # save the subject distribution
    split_dct = {'train': train_lst, 'valid': valid_lst, 'test': test_lst}
    split_path = Path(path_output, "split_datasets.joblib")
    joblib.dump(split_dct, split_path)

    return train_lst, valid_lst, test_lst


def get_subdatasets_subject_files_list(split_params: dict, df: pd.DataFrame, path_output: str,
                                       subject_selection: dict = None) -> (list, list, list):
    """Get lists of subject filenames for each sub-dataset between training / validation / testing.

    Args:
        split_params (dict): Split parameters, see :doc:`configuration_file` for more details.
        df (pd.DataFrame): Dataframe containing all BIDS image files indexed and their metadata.
        path_output (str): Output folder.
        subject_selection (dict): Used to specify a custom subject selection from a dataset.

    Returns:
        list, list list: Training, validation and testing filenames lists.
    """
    if split_params[SplitDatasetKW.FNAME_SPLIT]:
        # Load subjects lists
        old_split = joblib.load(split_params[SplitDatasetKW.FNAME_SPLIT])
        train_lst, valid_lst, test_lst = old_split['train'], old_split['valid'], old_split['test']

        # Backward compatibility for subject_file_lst containing participant_ids instead of filenames
        df_subjects = df[df['filename'].isin(train_lst)]
        if df_subjects.empty:
            df_train = df[df['participant_id'].isin(train_lst)]
            train_lst = sorted(df_train['filename'].to_list())

        df_subjects = df[df['filename'].isin(valid_lst)]
        if df_subjects.empty:
            df_valid = df[df['participant_id'].isin(valid_lst)]
            valid_lst = sorted(df_valid['filename'].to_list())

        df_subjects = df[df['filename'].isin(test_lst)]
        if df_subjects.empty:
            df_test = df[df['participant_id'].isin(test_lst)]
            test_lst = sorted(df_test['filename'].to_list())
    else:
        train_lst, valid_lst, test_lst = get_new_subject_file_split(df=df,
                                                                    split_method=split_params[SplitDatasetKW.SPLIT_METHOD],
                                                                    data_testing=split_params[SplitDatasetKW.DATA_TESTING],
                                                                    random_seed=split_params[SplitDatasetKW.RANDOM_SEED],
                                                                    train_frac=split_params[SplitDatasetKW.TRAIN_FRACTION],
                                                                    test_frac=split_params[SplitDatasetKW.TEST_FRACTION],
                                                                    path_output=path_output,
                                                                    balance=split_params[SplitDatasetKW.BALANCE]
                                                                    if SplitDatasetKW.BALANCE in split_params else None,
                                                                    subject_selection=subject_selection)
    return train_lst, valid_lst, test_lst


def imed_collate(batch: dict) -> dict | list | str | torch.Tensor:
    """Collates data to create batches

    Args:
        batch (dict): Contains input and gt data with their corresponding metadata.

    Returns:
        list or dict or str or tensor: Collated data.
    """
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        stacked = torch.stack(batch, 0)
        return stacked
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return __numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], str):
        return batch
    elif isinstance(batch[0], collections.abc.Mapping):
        return {key: imed_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.abc.Sequence):
        return [imed_collate(samples) for samples in batch]

    return batch


def filter_roi(roi_data: np.ndarray, nb_nonzero_thr: int) -> bool:
    """Filter slices from dataset using ROI data.

    This function filters slices (roi_data) where the number of non-zero voxels within the
    ROI slice (e.g. centerline, SC segmentation) is inferior or equal to a given threshold
    (nb_nonzero_thr).

    Args:
        roi_data (nd.array): ROI slice.
        nb_nonzero_thr (int): Threshold.

    Returns:
        bool: True if the slice needs to be filtered, False otherwise.
    """
    # Discard slices with less nonzero voxels than nb_nonzero_thr
    return not np.any(roi_data) or np.count_nonzero(roi_data) <= nb_nonzero_thr


def orient_img_hwd(data: np.ndarray, slice_axis: int) -> np.ndarray:
    """Orient a given RAS image to height, width, depth according to slice axis.

    Args:
        data (ndarray): RAS oriented data.
        slice_axis (int): Indicates the axis used for the 2D slice extraction:
            Sagittal: 0, Coronal: 1, Axial: 2.

    Returns:
        ndarray: Array oriented with the following dimensions: (height, width, depth).
    """
    if slice_axis == 0:
        return data.transpose(2, 1, 0)
    elif slice_axis == 1:
        return data.transpose(2, 0, 1)
    elif slice_axis == 2:
        return data


def orient_img_ras(data: np.ndarray, slice_axis: int) -> np.ndarray:
    """Orient a given array with dimensions (height, width, depth) to RAS orientation.

    Args:
        data (ndarray): Data with following dimensions (Height, Width, Depth).
        slice_axis (int): Indicates the axis used for the 2D slice extraction:
            Sagittal: 0, Coronal: 1, Axial: 2.

    Returns:
        ndarray: Array oriented in RAS.
    """

    if slice_axis == 0:
        return data.transpose(2, 1, 0) if len(data.shape) == 3 else data.transpose(0, 3, 2, 1)
    elif slice_axis == 1:
        return data.transpose(1, 2, 0) if len(data.shape) == 3 else data.transpose(0, 2, 3, 1)
    elif slice_axis == 2:
        return data


def orient_shapes_hwd(data: list | tuple, slice_axis: int) -> np.ndarray:
    """Swap dimensions according to match the height, width, depth orientation.

    Args:
        data (list or tuple): Shape or numbers associated with each image dimension
            (e.g. image resolution).
        slice_axis (int): Indicates the axis used for the 2D slice extraction:
            Sagittal: 0, Coronal: 1, Axial: 2.

    Returns:
        ndarray: Reoriented vector.
    """
    if slice_axis == 0:
        return np.array(data)[[2, 1, 0]]
    elif slice_axis == 1:
        return np.array(data)[[2, 0, 1]]
    elif slice_axis == 2:
        return np.array(data)


def update_metadata(metadata_src_lst: list, metadata_dest_lst: list) -> list:
    """Update metadata keys with a reference metadata.

    A given list of metadata keys will be changed and given the values of the reference metadata.

    Args:
        metadata_src_lst (list): List of source metadata used as reference for the
            destination metadata.
        metadata_dest_lst (list): List of metadate that needs to be updated.

    Returns:
        list: updated metadata list.
    """
    if metadata_src_lst and metadata_dest_lst:
        if not isinstance(metadata_dest_lst[0], list):  # annotation from one rater only
            metadata_dest_lst[0]._update(metadata_src_lst[0], TRANSFORM_PARAMS)
        else:  # annotations from several raters
            for idx, _ in enumerate(metadata_dest_lst[0]):
                metadata_dest_lst[0][idx]._update(metadata_src_lst[0], TRANSFORM_PARAMS)
    return metadata_dest_lst


def reorient_image(arr: np.ndarray, slice_axis: int, nib_ref: nib, nib_ref_canonical: nib) -> nd.ndarray:
    """Reorient an image to match a reference image orientation.

    It reorients a array to a given orientation and convert it to a nibabel object using the
    reference nibabel header.

    Args:
        arr (ndarray): Input array, array to re orient.
        slice_axis (int): Indicates the axis used for the 2D slice extraction:
            Sagittal: 0, Coronal: 1, Axial: 2.
        nib_ref (nibabel): Reference nibabel object, whose header is used.
        nib_ref_canonical (nibabel): `nib_ref` that has been reoriented to canonical orientation (RAS).
    """
    # Orient image in RAS according to slice axis
    arr_ras = orient_img_ras(arr, slice_axis)

    # https://gitship.com/neuroscience/nibabel/blob/master/nibabel/orientations.py
    ref_orientation = nib.orientations.io_orientation(nib_ref.affine)
    ras_orientation = nib.orientations.io_orientation(nib_ref_canonical.affine)
    # Return the orientation that transforms from ras to ref_orientation
    trans_orient = nib.orientations.ornt_transform(ras_orientation, ref_orientation)
    # apply transformation
    return nib.orientations.apply_orientation(arr_ras, trans_orient)


def get_file_extension(filename: str) -> Optional[str]:
    """ Get file extension if it is supported
    Args:
        filename (str): Path of the file.

    Returns:
        str: File extension
    """
    # Find the first match from the list of supported file extensions
    extension = next((ext for ext in EXT_LST if filename.lower().endswith(ext)), None)
    return extension


def update_filename_to_nifti(filename: str) -> str:
    """ 
    Update filename extension to 'nii.gz' if not a NifTI file.
    
    This function is used to help make non-NifTI files (e.g. PNG/TIF/JPG)
    compatible with NifTI-only pipelines. The expectation is that a NifTI
    version of the file has been created alongside the original file, which
    allows the extension to be cleanly swapped for a `.nii.gz` extension.

    Args:
        filename (str): Path of original file.

    Returns:
        str: Path of the corresponding NifTI file.
    """
    extension = get_file_extension(filename)
    if not "nii" in extension:
        filename = filename.replace(extension, ".nii.gz")
    return filename


def dropout_input(seg_pair: dict) -> dict:
    """Applies input-level dropout: zero to all channels minus one will be randomly set to zeros. This function verifies
    if some channels are already empty. Always at least one input channel will be kept.

    Args:
        seg_pair (dict): Batch containing torch tensors (input and gt) and metadata.

    Return:
        seg_pair (dict): Batch containing torch tensors (input and gt) and metadata with channel(s) dropped.
    """
    n_channels = seg_pair['input'].size(0)
    # Verify if the input is multichannel
    if n_channels > 1:
        # Verify if some channels are already empty
        n_unique_values = [len(torch.unique(input_data)) > 1 for input_data in seg_pair['input']]
        idx_empty = np.where(np.invert(n_unique_values))[0]

        # Select how many channels will be dropped between 0 and n_channels - 1 (keep at least one input)
        n_dropped = random.randint(0, n_channels - 1)

        if n_dropped > len(idx_empty):
            # Remove empty channel to the number of channels to drop
            n_dropped = n_dropped - len(idx_empty)
            # Select which channels will be dropped
            idx_dropped = []
            while len(idx_dropped) != n_dropped:
                idx = random.randint(0, n_channels - 1)
                # Don't include the empty channel in the dropped channels
                if idx not in idx_empty:
                    idx_dropped.append(idx)
        else:
            idx_dropped = idx_empty

        seg_pair['input'][idx_dropped] = torch.zeros_like(seg_pair['input'][idx_dropped])

    else:
        logger.warning("\n Impossible to apply input-level dropout since input is not multi-channel.")

    return seg_pair


def create_temp_directory() -> str:
    """Creates a temporary directory and returns its path.
    This temporary directory is only deleted when explicitly requested.

    Returns:
        str: Path of the temporary directory.
    """
    import datetime
    time_stamp = datetime.datetime.now().isoformat().replace(":", "")
    temp_folder_location = mkdtemp(prefix="ivadomed_", suffix=f"_{time_stamp}")
    return temp_folder_location

def get_obj_size(obj) -> int:
    """
    Returns the size of an object in bytes. Used to gauge whether storing object in memory vs write to disk.

    Source: https://stackoverflow.com/a/53705610

    Args:
        obj:

    Returns:

    """
    marked = {id(obj)}
    obj_q = [obj]
    object_size = 0

    while obj_q:
        object_size += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return object_size
