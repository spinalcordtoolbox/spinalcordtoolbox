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
import glob
import multiprocessing as mp

from ivadomed import inference as imed_inference
from totalspineseg.inference import inference as tss_inference
import numpy as np
import torch
from monai.transforms import SaveImage
from monai.inferers import sliding_window_inference

from spinalcordtoolbox.utils.fs import tmp_create, extract_fname
from spinalcordtoolbox.image import Image, get_orientation, add_suffix
from spinalcordtoolbox.math import binarize, remove_small_objects
from spinalcordtoolbox.deepseg_.postprocessing import keep_largest_object, fill_holes

import spinalcordtoolbox.deepseg.monai as ds_monai
import spinalcordtoolbox.deepseg.nnunet as ds_nnunet

from spinalcordtoolbox.utils.sys import LazyLoader

nib = LazyLoader("nib", globals(), "nibabel")

logger = logging.getLogger(__name__)


def segment_and_average_volumes(model_paths, input_filenames, options, use_gpu=False):
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
        :param use_gpu: bool. Whether to try to perform inference using CUDA. (NB: Only a single GPU will be used.)

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
        # NB: ivadomed turns on GPU inference via specifying a single GPU ID. This isn't the recommended way to do
        #     things, since it prevents us from running multi-GPU jobs. I think ivadomed did things this way to limit
        #     which GPU is used, but we can already accomplish this using the more universal 'CUDA_VISIBLE_DEVICES'
        #     environment variable). For now, the best we can do on our end is to select the first GPU from the list
        #     of available GPUs.
        gpu_id = 0 if use_gpu else None  # NB: If e.g. 'CUDA_VISIBLE_DEVICES=2,3,4', then 0 will refer to GPU 2.
        nii_lst, target_lst = imed_inference.segment_volume(path_model, input_filenames, gpu_id=gpu_id, options=options)
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


def segment_non_ivadomed(path_model, model_type, input_filenames, threshold, keep_largest, fill_holes_in_pred,
                         remove_small, use_gpu=False, remove_temp_files=True, extra_inference_kwargs=None):
    # MONAI and NNUnet have similar structure, and so we use nnunet+inference functions with the same signature
    # NB: For TotalSpineSeg, we don't need to create the network ourselves
    if "totalspineseg" in path_model:
        def create_net(pm, _): return pm  # just echo `path_model` back as 'net'
        inference = segment_totalspineseg
    elif model_type == "monai":
        create_net = ds_monai.create_nnunet_from_plans
        inference = segment_monai
    else:
        assert model_type == "nnunet"
        create_net = ds_nnunet.create_nnunet_from_plans
        inference = segment_nnunet

    device = torch.device("cuda" if use_gpu else "cpu")

    # load model from checkpoint
    net = create_net(path_model, device)

    im_lst, target_lst = [], []
    for fname_in in input_filenames:
        tmpdir = tmp_create(basename="sct_deepseg")
        # model may be multiclass, so the `inference` func should output a list of fnames and targets
        fnames_out, targets = inference(path_img=fname_in, tmpdir=tmpdir, predictor=net, device=device, **extra_inference_kwargs)
        for fname_out, target in zip(fnames_out, targets):
            im_out = Image(fname_out)
            # Apply postprocessing (replicates existing functionality from ivadomed package)
            # 1. Binarize predictions based on the threshold value
            # NOTE: Any post-processing is done only after thresholding (as post-processing operations
            # expect binary masks)
            if threshold:  # Default: None, but `-thr 0` will also turn off binarization
                im_out.data = binarize(im_out.data, threshold)
            # 2. Keep the largest connected object
            if keep_largest != 0:
                im_out.data[im_out.data < 0.001] = 0  # Replicates ivadomed's `@binarize_with_low_threshold`
                im_out.data = keep_largest_object(im_out.data, x_cOm=None, y_cOm=None)
            # 3. Fill holes
            if fill_holes_in_pred != 0:
                im_out.data = fill_holes(im_out.data)
            # 4. Remove small objects
            if remove_small is not None:
                unit = 'mm3' if 'mm3' in remove_small[-1] else 'vox'
                thr = [int(t.replace(unit, "")) for t in remove_small]
                im_out.data = remove_small_objects(im_out.data, im_out.dim[4:7], unit, thr)
            im_lst.append(im_out)
            target_lst.append(target)
        if remove_temp_files:
            shutil.rmtree(tmpdir)

    return im_lst, target_lst


def segment_monai(path_img, tmpdir, predictor, device: torch.device):
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
    # NOTE: this is hard-coded to "edge" based on extensive experiments comparing "edge" vs "zero" padding
    # at test-time.
    pad_mode = "edge"

    # define the dataset and dataloader
    test_loader, test_post_pred = ds_monai.prepare_data(path_img_tmp, crop_size=crop_size, padding=pad_mode)
    [batch] = test_loader  # we expected there to only be one batch (with one image)

    # Run MONAI prediction
    print('Starting inference...')
    start = time.time()

    # run inference
    with torch.no_grad():
        test_input = batch["image"].to(device)
        batch["pred"] = sliding_window_inference(test_input, inference_roi_size, mode="gaussian",
                                                 sw_batch_size=4, predictor=predictor, overlap=0.5, progress=False,
                                                 sw_device=device)
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


def segment_nnunet(path_img, tmpdir, predictor, device: torch.device):
    """
    This script is used to run inference on a single subject using a nnUNetV2 model.

    Author: Jan Valosek, Naga Karthik
    Original script: https://github.com/ivadomed/model_seg_sci/blob/4184bc22ef7317b3de5f85dee28449d6f381c984/packaging/run_inference_single_subject.py

    TODO: Find a less brittle way to specify model-based parameters such as model orientation, suffix, etc.
    """
    # NB: `device` should already be set when the `predictor` is initialized. We only have the `device` parameter here
    #     to match the function signature of `segment_monai`, which *does* require `device` at inference time.
    if device != predictor.device:
        logger.warning(f"Param `device` (value: {device}) is ignored in favor of `predictor.device` (value: "
                       f"{predictor.device}). To change the device, please modify the initialization of the predictor.")

    # Copy the file to the temporary directory using shutil.copyfile
    path_img_tmp = os.path.join(tmpdir, os.path.basename(path_img))
    shutil.copyfile(path_img, path_img_tmp)
    logger.info(f'Copied {path_img} to {path_img_tmp}')

    # Get the original orientation of the image, for example LPI
    orig_orientation = get_orientation(Image(path_img_tmp))

    # Get the orientation used by the model
    # Check if predictor.dataset_json['image_orientation'] key exists, if so, read the orientation from there
    # NOTE: the 'image_orientation' key-value pair needs to be manually added to the dataset.json file
    if 'image_orientation' in predictor.dataset_json:
        model_orientation = predictor.dataset_json['image_orientation']
        print(f"Orientation (based on dataset.json): {model_orientation}")
    else:
        if "RegionBasedLesionSeg" in predictor.plans_manager.dataset_name:
            model_orientation = "AIL"
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
    # for the canal model, keep only the largest object
    if 'canal_seg' in labels.keys():
        logger.info('Keeping only the largest component.')
        img_out.data = keep_largest_object(img_out.data, x_cOm=None, y_cOm=None)
    # for rootlets model (which has labels 'lvl1', 'lvl2', etc.), save the image directly without splitting
    is_rootlet_model = all((label == f"lvl{i}") for i, label in enumerate(labels.keys(), start=1))
    if is_rootlet_model:
        targets = ["_rootlets"]
        outputs = [img_out]
    # for spinal cord segmentation models with only 1 output label, save the image directly with specific "_seg" suffix
    # this is added to preserve the typical expected output for SC models (`sct_propseg`, `sct_deepseg_sc`, etc.)
    # see also: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/4805
    elif sorted(labels.keys()) == ['sc']:
        targets = ["_seg"]
        outputs = [img_out]
    # for the other multiclass models (SCI lesion/SC, mouse GM/WM, etc.), save 1 image per label
    else:
        targets, outputs = [], []
        for label, label_values in labels.items():
            # Binarize data array by matching array values with label values
            # e.g. if `label_values == [1, 2]`, then for the data array, 0 -> False and 1,2 -> True
            # This handles nested labels, e.g. when the SC includes both lesion labels (2) and cord label (1)
            # Numpy syntax reference: https://stackoverflow.com/a/20528566
            label_values = label_values if isinstance(label_values, list) else [label_values]      # convert to list
            bool_array = np.logical_or.reduce([img_out.data == int(val) for val in label_values])  # 'OR' each label val
            img_bin = Image(bool_array.astype(np.uint8), hdr=img_out.hdr)
            target = f"_{label}_seg"
            targets.append(target)
            outputs.append(img_bin)

    # Save each image using the suffixes determined above
    fnames_out = []
    for target, img_out in zip(targets, outputs):
        fname_out = add_suffix(fname_prediction, target)
        logger.info(f"Saving results to: {fname_out}")
        img_out.save(fname_out)
        fnames_out.append(fname_out)

    return fnames_out, targets


def segment_totalspineseg(path_img, tmpdir, predictor, device, step1_only=False):
    # for totalspineseg, the 'predictor' is just the model path
    path_model = predictor
    # fetch the release subdirectory from the model path
    installed_releases = sorted(
        os.path.basename(release_path) for release_path in
        glob.glob(os.path.join(path_model, 'nnUNet', 'results', 'r*'), recursive=True)
    )
    # There should always be a release subdirectory, hence the 'assert'
    assert installed_releases, f"No 'nnUNet/results/rYYYYMMDD' subdirectory found in {path_model}"

    # Copy the file to the temporary directory using shutil.copyfile
    path_img_tmp = os.path.join(tmpdir, os.path.basename(path_img))
    shutil.copyfile(path_img, path_img_tmp)
    logger.info(f'Copied {path_img} to {path_img_tmp}')

    # Create directory for nnUNet prediction
    tmpdir_nnunet = os.path.join(tmpdir, 'nnUNet_prediction')
    os.mkdir(tmpdir_nnunet)

    # totalspineseg uses multiprocessing internally, so here we try to
    # set multiprocessing method to 'spawn' to avoid deadlock issues
    # - 'spawn' is already default on Windows/macOS, but 'fork' is default on Linux
    # - 'spawn' is reliable though slower, but 'fork' causes intermittent stalling
    mp.set_start_method("spawn", force=True)

    tss_inference(
        input_path=path_img_tmp,
        output_path=tmpdir_nnunet,
        data_path=path_model,
        # totalspineseg requires explicitly specifying the release subdirectory
        default_release=installed_releases[-1],  # use the most recent release
        # totalspineseg expects the device type, not torch.device
        device=device,
        # Try to address stalling due to the use of concurrent.futures in totalspineseg
        max_workers=1,
        max_workers_nnunet=1,
        # Optional argument to choose which models to run
        step1_only=bool(step1_only)
    )
    fnames_out, targets = [], []
    expected_outputs = ["step1_canal", "step1_cord", "step1_levels", "step1_output"]
    if not step1_only:
        expected_outputs.append("step2_output")
    for output_dirname in expected_outputs:
        fnames_out.append(os.path.join(tmpdir_nnunet, output_dirname, os.path.basename(path_img)))
        targets.append(f"_{output_dirname}")

    return fnames_out, targets
