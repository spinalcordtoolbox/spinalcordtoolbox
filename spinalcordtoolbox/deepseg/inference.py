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
from monai.inferers import sliding_window_inference

from spinalcordtoolbox.utils.fs import tmp_create, extract_fname
from spinalcordtoolbox.utils.sys import run_proc
from spinalcordtoolbox.image import Image, get_orientation, add_suffix
from spinalcordtoolbox.math import binarize
from spinalcordtoolbox.deepseg.monai import create_nnunet_from_plans, prepare_data, postprocessing

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
    batch["pred"] = sliding_window_inference(test_input, inference_roi_size, mode="gaussian",
                                             sw_batch_size=4, predictor=net, overlap=0.5, progress=False)
    pred = postprocessing(batch, test_post_pred)

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


def segment_nnunet(path_model, input_filenames, threshold):
    """
    This script is used to run inference on a single subject using a nnUNetV2 model.

    Note: conda environment with nnUNetV2 is required to run this script.
    For details how to install nnUNetV2, see:
    https://github.com/ivadomed/utilities/blob/main/quick_start_guides/nnU-Net_quick_start_guide.md#installation

    Author: Jan Valosek, Naga Karthik

    Example spinal cord segmentation:
        python run_inference_single_subject.py
            -i sub-001_T2w.nii.gz
            -o sub-001_T2w_seg_nnunet.nii.gz
            -path-model /path/to/model
            -pred-type sc
            -tile-step-size 0.5

    Example lesion segmentation:
        python run_inference_single_subject.py
            -i sub-001_T2w.nii.gz
            -o sub-001_T2w_lesion_seg_nnunet.nii.gz
            -path-model /path/to/model
            -pred-type lesion
            -tile-step-size 0.5
    """

    import os
    import shutil
    import time

    import torch
    import numpy as np
    from batchgenerators.utilities.file_and_folder_operations import join
    # from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data as predictor
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    def main(fname_file, path_model, use_gpu=False, use_best_checkpoint=False, tile_step_size=0.5):
        print(f'Found {fname_file} file.')

        # Create temporary directory in the temp to store the reoriented images
        tmpdir = tmp_create(basename="sciseg_prediction")
        # Copy the file to the temporary directory using shutil.copyfile
        fname_file_tmp = os.path.join(tmpdir, os.path.basename(fname_file))
        shutil.copyfile(fname_file, fname_file_tmp)
        print(f'Copied {fname_file} to {fname_file_tmp}')

        # Get the original orientation of the image, for example LPI
        orig_orientation = get_orientation(Image(fname_file_tmp))

        # Get the orientation used by the model
        if "sci_multiclass" in path_model:
            model_orientation = "RPI"
        else:
            assert "rootlets" in path_model
            model_orientation = "LPI"

        # Reorient the image to model orientation if not already
        img_in = Image(fname_file_tmp)
        if orig_orientation != model_orientation:
            img_in.change_orientation(model_orientation)

        # Use all the folds available in the model folder by default
        folds_avail = [int(f.split('_')[-1]) for f in os.listdir(path_model) if f.startswith('fold_')]

        # Create directory for nnUNet prediction
        tmpdir_nnunet = os.path.join(tmpdir, 'nnUNet_prediction')
        fname_prediction = os.path.join(tmpdir_nnunet, os.path.basename(add_suffix(fname_file_tmp, "_pred")))
        os.mkdir(tmpdir_nnunet)

        # Run nnUNet prediction
        print('Starting inference...')
        start = time.time()

        # instantiate the nnUNetPredictor
        predictor = nnUNetPredictor(
            tile_step_size=tile_step_size,  # changing it from 0.5 to 0.9 makes inference faster
            use_gaussian=True,  # applies gaussian noise and gaussian blur
            use_mirroring=False,  # test time augmentation by mirroring on all axes
            perform_everything_on_gpu=True if use_gpu else False,
            device=torch.device('cuda') if use_gpu else torch.device('cpu'),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
        print(f'Running inference on device: {predictor.device}')

        # initializes the network architecture, loads the checkpoint
        predictor.initialize_from_trained_model_folder(
            join(path_model),
            use_folds=folds_avail,
            checkpoint_name='checkpoint_final.pth' if not use_best_checkpoint else 'checkpoint_best.pth',
        )
        print('Model loaded successfully. Fetching test data...')

        data = np.expand_dims(img_in.data, axis=0).transpose([0, 3, 2, 1]).astype(np.float32)
        pred = predictor.predict_single_npy_array(
            input_image=data,
            image_properties={'spacing': img_in.dim[4:7]},
        ).transpose([2, 1, 0])
        img_out = img_in.copy()
        img_out.data = pred

        end = time.time()
        print('Inference done.')
        total_time = end - start
        print(f'Total inference time: {int(total_time // 60)} minute(s) {int(round(total_time % 60))} seconds')

        print('Re-orienting the prediction back to original orientation...')
        # Reorient the image back to original orientation
        if orig_orientation != model_orientation:
            img_out.change_orientation(orig_orientation)
            print(f'Reorientation to original orientation {orig_orientation} done.')

        # for SCI multiclass model, split the predictions into sc-seg and lesion-seg
        if "sci_multiclass" in path_model:
            targets = [f"_{pred_type}_seg" for pred_type in ['sc', 'lesion']]
            fnames_out = [add_suffix(fname_prediction, target) for target in targets]
            for i, fname_out in enumerate(fnames_out):
                img_bin = img_out.copy()
                img_bin.data = binarize(img_bin.data, i)
                img_bin.save(fname_out)
        else:
            targets = ["_seg"]
            fnames_out = [fname_prediction]
            img_out.save(fname_prediction)

        print('-' * 50)
        print(f'Created {fnames_out}')
        print('-' * 50)

        return fnames_out, targets

    nii_lst, target_lst = [], []
    for fname_in in input_filenames:
        fnames_out, targets = main(fname_file=fname_in, path_model=path_model)
        for fname_out, target in zip(fnames_out, targets):
            # TODO: Use API binarization function when output filetype is sct.image.Image
            run_proc(["sct_maths", "-i", fname_out, "-bin", str(threshold), "-o", fname_out])
            # TODO: Change the output filetype from Nifti1Image to sct.image.Image to mitigate #3232
            nii_lst.append(nib.load(fname_out))
            target_lst.append(target)

    return nii_lst, target_lst