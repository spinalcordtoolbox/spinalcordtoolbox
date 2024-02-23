"""
Ensemble inference

Copyright (c) 2023 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import logging
import os
import shutil
import time
from pathlib import Path

from ivadomed import inference as imed_inference
import nibabel as nib
import numpy as np
import torch
from monai.transforms import SaveImage
from monai.inferers import sliding_window_inference

from spinalcordtoolbox.utils.fs import tmp_create, extract_fname
from spinalcordtoolbox.image import Image, get_orientation, add_suffix
from spinalcordtoolbox.math import binarize

import spinalcordtoolbox.deepseg.monai as ds_monai
import spinalcordtoolbox.deepseg.nnunet as ds_nnunet

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

        :return: list, list: List of Image objects containing the soft segmentation(s), one per prediction class, \
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

    # Convert the output Nifti1Images into SCT images, to avoid scaling issues
    # (see https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4346)
    im_lst = [Image(np.asanyarray(nii.dataobj), hdr=nii.header) for nii in nii_lst]

    return im_lst, target_lst


def segment_non_ivadomed(path_model, model_type, input_filenames, threshold, remove_temp_files=True):
    # MONAI and NNUnet have similar structure, and so we use nnunet+inference functions with the same signature
    if model_type == "monai":
        create_net = ds_monai.create_nnunet_from_plans
        inference = segment_monai
    else:
        assert model_type == "nnunet"
        create_net = ds_nnunet.create_nnunet_from_plans
        inference = segment_nnunet

    # load model from checkpoint
    net = create_net(path_model)

    im_lst, target_lst = [], []
    for fname_in in input_filenames:
        tmpdir = tmp_create(basename="sct_deepseg")
        # model may be multiclass, so the `inference` func should output a list of fnames and targets
        fnames_out, targets = inference(path_img=fname_in, tmpdir=tmpdir, predictor=net)
        for fname_out, target in zip(fnames_out, targets):
            im_out = Image(fname_out)
            if threshold is not None:
                im_out.data = binarize(im_out.data, threshold)
            im_lst.append(im_out)
            target_lst.append(target)
        if remove_temp_files:
            shutil.rmtree(tmpdir)

    return im_lst, target_lst


def segment_monai(path_img, tmpdir, predictor):
    """
    Script to run inference on a MONAI-based model for contrast-agnostic soft segmentation of the spinal cord.

    Author: Naga Karthik
    Original script: https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/blob/e65478099d026f865b7f1d7d0082e6e9a507a744/monai/run_inference_single_image.py
    """
    # Copy the file to the temporary directory using shutil.copyfile
    path_img_tmp = os.path.join(tmpdir, os.path.basename(path_img))
    shutil.copyfile(path_img, path_img_tmp)
    logger.info(f'Copied {path_img} to {path_img_tmp}')

    # define inference patch size and center crop size
    crop_size = (64, 192, -1)
    inference_roi_size = (64, 192, 320)

    # define the dataset and dataloader
    test_loader, test_post_pred = ds_monai.prepare_data(path_img_tmp, crop_size=crop_size)
    [batch] = test_loader  # we expected there to only be one batch (with one image)

    # Run MONAI prediction
    print('Starting inference...')
    start = time.time()

    # run inference
    with torch.no_grad():
        test_input = batch["image"].to(torch.device("cpu"))
        batch["pred"] = sliding_window_inference(test_input, inference_roi_size, mode="gaussian",
                                                 sw_batch_size=4, predictor=predictor, overlap=0.5, progress=False)
        pred = ds_monai.postprocessing(batch, test_post_pred)

        end = time.time()
        print('Inference done.')
        total_time = end - start
        print(f'Total inference time: {int(total_time // 60)} minute(s) {int(round(total_time % 60))} seconds')

        # this takes about 0.25s on average on a CPU
        # image saver class
        _, fname, ext = extract_fname(path_img)
        postfix = "seg"
        target = f"_{postfix}"
        pred_saver = SaveImage(
            output_dir=tmpdir, output_postfix=postfix, output_ext=ext,
            separate_folder=False, print_log=False)
        # save the prediction
        fname_out = os.path.join(tmpdir, f"{fname}_{postfix}{ext}")
        logger.info(f"Saving results to: {fname_out}")
        pred_saver(pred)

    return [fname_out], [target]


def segment_nnunet(path_img, tmpdir, predictor):
    """
    This script is used to run inference on a single subject using a nnUNetV2 model.

    Author: Jan Valosek, Naga Karthik
    Original script: https://github.com/ivadomed/model_seg_sci/blob/4184bc22ef7317b3de5f85dee28449d6f381c984/packaging/run_inference_single_subject.py

    TODO: Find a less brittle way to specify model-based parameters such as model orientation, suffix, etc.
    """
    # Copy the file to the temporary directory using shutil.copyfile
    path_img_tmp = os.path.join(tmpdir, os.path.basename(path_img))
    shutil.copyfile(path_img, path_img_tmp)
    logger.info(f'Copied {path_img} to {path_img_tmp}')

    # Get the original orientation of the image, for example LPI
    orig_orientation = get_orientation(Image(path_img_tmp))

    # Get the orientation used by the model
    if "SCI" in predictor.plans_manager.dataset_name:
        model_orientation = "RPI"
    else:
        model_orientation = "LPI"

    # Reorient the image to model orientation if not already
    img_in = Image(path_img_tmp)
    if orig_orientation != model_orientation:
        logger.info(f'Changing orientation of the input to the model orientation ({model_orientation})...')
        img_in.change_orientation(model_orientation)

    # Create directory for nnUNet prediction
    tmpdir_nnunet = os.path.join(tmpdir, 'nnUNet_prediction')
    fname_prediction = os.path.join(tmpdir_nnunet, os.path.basename(add_suffix(path_img_tmp, "_pred")))
    os.mkdir(tmpdir_nnunet)

    # Run nnUNet prediction
    print('Starting inference...')
    start = time.time()

    # NOTE: nnUNet loads `.nii.gz` images using SimpleITK. When working with SimpleITK images, the axes are [x,y,z].
    #       But, during training, when nnUNet fetches a numpy array from the SimpleITK image, the axes get swapped
    #       ([z,y,x]). Nibabel (the image processing library SCT uses internally) _doesn't_ have this axis-swapping
    #       behavior. So, when SCT fetches the numpy array, we have to swap axes [x] and [z] to mimic nnUNet's internal
    #       behavior. See also: https://github.com/MIC-DKFZ/nnUNet/issues/1951
    data = img_in.data.transpose([2, 1, 0])
    # We also need to add an axis and convert to float32 to match nnUNet's input expectations
    # (This would automatically be done by nnUNet if we were to predict from image files, rather than a npy array.)
    data = np.expand_dims(data, axis=0).astype(np.float32)
    pred = predictor.predict_single_npy_array(
        input_image=data,
        # The spacings also have to be reversed to match nnUNet's conventions.
        image_properties={'spacing': img_in.dim[6:3:-1]},
    )
    # Lastly, we undo the transpose to return the image from [z,y,x] (SimpleITK) to [x,y,z] (nibabel)
    pred = pred.transpose([2, 1, 0])
    img_out = img_in.copy()
    img_out.data = pred

    end = time.time()
    print('Inference done.')
    total_time = end - start
    print(f'Total inference time: {int(total_time // 60)} minute(s) {int(round(total_time % 60))} seconds')

    logger.info('Reorienting the prediction back to original orientation...')
    # Reorient the image back to original orientation
    if orig_orientation != model_orientation:
        img_out.change_orientation(orig_orientation)
        logger.info(f'Reorientation to original orientation {orig_orientation} done.')

    labels = {k: v for k, v in predictor.dataset_json['labels'].items() if k != 'background'}
    # for rootlets model (which has labels 'lvl1', 'lvl2', etc.), save the image directly without splitting
    is_rootlet_model = all((label == f"lvl{i}") for i, label in enumerate(labels.keys(), start=1))
    if is_rootlet_model:
        targets = ["_seg"]
        fnames_out = [fname_prediction]
        logger.info(f"Saving results to: {fname_prediction}")
        img_out.save(fname_prediction)
    # for the other multiclass models (SCI lesion/SC, mouse GM/WM, etc.), save 1 image per label
    else:
        targets, fnames_out = [], []
        for label, label_values in labels.items():
            # Binarize data array by matching array values with label values
            # e.g. if `label_values == [1, 2]`, then for the data array, 0 -> False and 1,2 -> True
            # This handles nested labels, e.g. when the SC includes both lesion labels (2) and cord label (1)
            # Numpy syntax reference: https://stackoverflow.com/a/20528566
            label_values = label_values if isinstance(label_values, list) else [label_values]      # convert to list
            bool_array = np.logical_or.reduce([img_out.data == int(val) for val in label_values])  # 'OR' each label val
            img_bin = Image(bool_array.astype(np.uint8), hdr=img_out.hdr)
            # Save the image using the label name from the dataset.json file
            target = f"_{label}_seg"
            fname_out = add_suffix(fname_prediction, target)
            logger.info(f"Saving results to: {fname_out}")
            img_bin.save(fname_out)
            targets.append(target)
            fnames_out.append(fname_out)

    return fnames_out, targets
