import copy
from loguru import logger
from typing import List
from ivadomed import transforms as imed_transforms
from ivadomed import utils as imed_utils
from ivadomed.loader.bids3d_dataset import Bids3DDataset
from ivadomed.loader.bids_dataframe import BidsDataframe
from ivadomed.loader.bids_dataset import BidsDataset
from ivadomed.keywords import ROIParamsKW, TransformationKW, ModelParamsKW, ConfigKW
from ivadomed.loader.slice_filter import SliceFilter
from ivadomed.loader.patch_filter import PatchFilter
import torch


def load_dataset(bids_df: BidsDataframe,
                 data_list: List[str],
                 transforms_params: dict,
                 model_params: dict,
                 target_suffix: List[str],
                 roi_params: dict,
                 contrast_params: dict,
                 slice_filter_params: dict,
                 patch_filter_params: dict,
                 slice_axis: str,
                 multichannel: bool,
                 dataset_type: str = "training",
                 requires_undo: bool = False,
                 metadata_type: str = None,
                 object_detection_params: dict = None,
                 soft_gt: bool = False,
                 device: torch.device = None,
                 cuda_available: bool = None,
                 is_input_dropout: bool = False,
                 **kwargs) -> Bids3DDataset:
    """Get loader appropriate loader according to model type. Available loaders are Bids3DDataset for 3D data,
    BidsDataset for 2D data and HDF5Dataset for HeMIS.

    Args:

        bids_df (BidsDataframe): Object containing dataframe with all BIDS image files and their metadata.
        data_list (list): Subject names list.
        transforms_params (dict): Dictionary containing transformations for "training", "validation", "testing" (keys),
            eg output of imed_transforms.get_subdatasets_transforms.
        model_params (dict): Dictionary containing model parameters.
        target_suffix (list of str): List of suffixes for target masks.
        roi_params (dict): Contains ROI related parameters.
        contrast_params (dict): Contains image contrasts related parameters.
        slice_filter_params (dict): Contains slice_filter_params, see :doc:`configuration_file` for more details.
        patch_filter_params (dict): Contains patch_filter_params, see :doc:`configuration_file` for more details.
        slice_axis (string): Choice between "axial", "sagittal", "coronal" ; controls the axis used to extract the 2D
            data from 3D NifTI files. 2D PNG/TIF/JPG files use default "axial.
        multichannel (bool): If True, the input contrasts are combined as input channels for the model. Otherwise, each
            contrast is processed individually (ie different sample / tensor).
        metadata_type (str): Choice between None, "mri_params", "contrasts".
        dataset_type (str): Choice between "training", "validation" or "testing".
        requires_undo (bool): If True, the transformations without undo_transform will be discarded.
        object_detection_params (dict): Object dection parameters.
        soft_gt (bool): If True, ground truths are not binarized before being fed to the network. Otherwise, ground
        truths are thresholded (0.5) after the data augmentation operations.
        device (torch.device): Device to use for the model training.
        cuda_available (bool): If True, cuda is available.
        is_input_dropout (bool): Return input with missing modalities.

    Returns:
        BidsDataset

    Note: For more details on the parameters transform_params, target_suffix, roi_params, contrast_params,
    slice_filter_params, patch_filter_params and object_detection_params see :doc:`configuration_file`.
    """
    # Compose transforms
    tranform_lst, _ = imed_transforms.prepare_transforms(copy.deepcopy(transforms_params), requires_undo)

    # If ROICrop is not part of the transforms, then enforce no slice filtering based on ROI data.
    if TransformationKW.ROICROP not in transforms_params:
        roi_params[ROIParamsKW.SLICE_FILTER_ROI] = None

    if model_params[ModelParamsKW.NAME] == ConfigKW.MODIFIED_3D_UNET \
            or (ModelParamsKW.IS_2D in model_params and not model_params[ModelParamsKW.IS_2D]):
        dataset = Bids3DDataset(bids_df=bids_df,
                                subject_file_lst=data_list,
                                target_suffix=target_suffix,
                                roi_params=roi_params,
                                contrast_params=contrast_params,
                                metadata_choice=metadata_type,
                                slice_axis=imed_utils.AXIS_DCT[slice_axis],
                                transform=tranform_lst,
                                multichannel=multichannel,
                                subvolume_filter_fn=PatchFilter(**patch_filter_params, is_train=False if dataset_type == "testing" else True),
                                model_params=model_params,
                                object_detection_params=object_detection_params,
                                soft_gt=soft_gt,
                                is_input_dropout=is_input_dropout)
    else:
        # Task selection
        task = imed_utils.get_task(model_params[ModelParamsKW.NAME])

        dataset = BidsDataset(bids_df=bids_df,
                              subject_file_lst=data_list,
                              target_suffix=target_suffix,
                              roi_params=roi_params,
                              contrast_params=contrast_params,
                              model_params=model_params,
                              metadata_choice=metadata_type,
                              slice_axis=imed_utils.AXIS_DCT[slice_axis],
                              transform=tranform_lst,
                              multichannel=multichannel,
                              slice_filter_fn=SliceFilter(**slice_filter_params, device=device,
                                                          cuda_available=cuda_available),
                              patch_filter_fn=PatchFilter(**patch_filter_params,
                                                          is_train=False if dataset_type == "testing" else True),
                              soft_gt=soft_gt,
                              object_detection_params=object_detection_params,
                              task=task,
                              is_input_dropout=is_input_dropout)
        dataset.load_filenames()

    if model_params[ModelParamsKW.NAME] == ConfigKW.MODIFIED_3D_UNET:
        logger.info(f"Loaded {len(dataset)} volumes of shape {dataset.length} for the {dataset_type} set.")
    elif model_params[ModelParamsKW.NAME] != ConfigKW.HEMIS_UNET and dataset.length:
        logger.info(f"Loaded {len(dataset)} {slice_axis} patches of shape {dataset.length} for the {dataset_type} set.")
    else:
        logger.info(f"Loaded {len(dataset)} {slice_axis} slices for the { dataset_type} set.")

    return dataset
