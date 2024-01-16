"""
Ensemble inference

Copyright (c) 2023 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import logging
import os
from pathlib import Path

from ivadomed import inference as imed_inference
import nibabel as nib
import numpy as np
import torch
from monai.transforms import SaveImage

from spinalcordtoolbox.utils.fs import tmp_create, extract_fname
from spinalcordtoolbox.utils.sys import run_proc
from spinalcordtoolbox.deepseg.monai import create_nnunet_from_plans, prepare_data, sliding_window_inference_wrapped

logger = logging.getLogger(__name__)


def segment_and_average_volumes(model_paths, input_filenames, options):
    """
        Run `ivadomed.inference.segment_volume()` once per model, then average the outputs.

        :param model_paths: A list of folder paths. The folders must contain:
            (1) the model ('folder_model/folder_model.pt') to use
            (2) its configuration file ('folder_model/folder_model.json') used for the training,
            see https://github.com/neuropoly/ivadomed/wiki/configuration-file
        :param input_filenames: list of image filenames (e.g. .nii.gz) to segment. Multichannel models require multiple
            images to segment, i.e.., len(fname_images) > 1.
        :param options: A dictionary containing optional configuration settings, as specified by the
            ivadomed.inference.segment_volume function.

        :return: list, list: List of nibabel objects containing the soft segmentation(s), one per prediction class, \
            List of target suffix associated with each prediction
    """
    if not isinstance(model_paths, list):
        raise TypeError("'model_paths' must be a list of model path strings.")
    if not len(model_paths) > 0:
        raise ValueError("'model_paths' must contain one or more model path strings.")

    # Fetch the name of the model (to be used in logging)
    name_model = Path(model_paths[0]).parts[-1]
    logger.info(f"\nRunning inference for model '{name_model}'...")

    # Perform inference once per model
    nii_lsts, target_lsts = [], []
    for path_model in model_paths:
        if len(model_paths) > 1:  # We have an ensemble, so output messages to distinguish between seeds
            name_seed = Path(path_model).parts[-2]
            logger.info(f"\nUsing '{name_seed}'...")
        nii_lst, target_lst = imed_inference.segment_volume(path_model, input_filenames, options=options)
        nii_lsts.append(nii_lst)
        target_lsts.append(target_lst)

    # If we have a single model, skip averaging
    if len(model_paths) == 1:
        nii_lst = nii_lsts[0]
    # Otherwise, we have a model ensemble, so average the image data
    else:
        logger.info(f"\nAveraging outputs across the ensemble for '{name_model}'...")
        nii_lst = []
        # NB: `nii_lsts` is a list of lists, with each sublist being the *per-model* predictions. Example:
        #         [
        #             [m1_prediction_1.nii.gz, m1_prediction_2.nii.gz, ...],  # model 1 predictions
        #             [m2_prediction_1.nii.gz, m2_prediction_2.nii.gz, ...]   # model 2 predictions
        #         ]
        # So, we want to take: the average of "prediction_1", the average of "prediction_2", etc.
        # To do this, we unpack + zip `nii_lists`, so that "prediction_N" files are grouped as "predictions".
        for predictions in zip(*nii_lsts):
            # Average the data for each output in the ensemble
            data_stack = np.stack([pred.get_fdata() for pred in predictions], axis=0)
            data_mean = np.mean(data_stack, axis=0)
            # Take the first image's header to reuse for the averaged image
            nii_header = predictions[0].header
            # Create a new Nifti1Image containing the averaged output
            nii_lst.append(nib.Nifti1Image(data_mean, header=nii_header, affine=nii_header.get_best_affine()))

    # The 'targets' should be identical for each model, so just take the first
    target_lst = target_lsts[0]

    return nii_lst, target_lst


def segment_monai(path_model, input_filenames, threshold, device="cpu"):
    # equivalent to `with torch.no_grad()`
    torch.set_grad_enabled(False)

    # define device
    if device == "gpu" and not torch.cuda.is_available():
        logger.warning("GPU not available, using CPU instead")
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")

    # load model from checkpoint
    chkp_path = os.path.join(path_model, "best_model_loss.ckpt")
    net = create_nnunet_from_plans(chkp_path, DEVICE)

    nii_lst, target_lst = [], []
    for fname_in in input_filenames:
        tmp_dir = tmp_create("sct_deepseg")
        fname_out, target = segment_monai_single(path_img=fname_in, net=net, device=DEVICE, path_out=tmp_dir)
        # TODO: Use API binarization function when output filetype is sct.image.Image
        run_proc(["sct_maths", "-i", fname_out, "-bin", str(threshold), "-o", fname_out])
        # TODO: Change the output filetype from Nifti1Image to sct.image.Image to mitigate #3232
        nii_lst.append(nib.load(fname_out))
        target_lst.append(target)

    return nii_lst, target_lst


def segment_monai_single(path_img, path_out, net, device):
    """
    Script to run inference on a MONAI-based model for contrast-agnostic soft segmentation of the spinal cord.

    Author: Naga Karthik

    """
    # define inference patch size and center crop size
    crop_size = (64, 192, -1)
    inference_roi_size = (64, 192, 320)

    # define the dataset and dataloader
    test_loader, test_post_pred = prepare_data(path_img, path_out, crop_size=crop_size)
    batch = next(iter(test_loader))

    # run inference
    test_input = batch["image"].to(device)
    pred = sliding_window_inference_wrapped(
        batch=batch,
        test_input=test_input,
        inference_roi_size=inference_roi_size,
        predictor=net,
        test_post_pred=test_post_pred
    )

    # this takes about 0.25s on average on a CPU
    # image saver class
    _, fname, ext = extract_fname(path_img)
    postfix = "seg"
    pred_saver = SaveImage(
        output_dir=path_out, output_postfix=postfix, output_ext=ext,
        separate_folder=False, print_log=False)
    # save the prediction
    fname_out = os.path.join(path_out, f"{fname}_{postfix}{ext}")
    logger.info(f"Saving results to: {fname_out}")
    pred_saver(pred)

    return fname_out, f"_{postfix}"


def segment_nnunet():
    return