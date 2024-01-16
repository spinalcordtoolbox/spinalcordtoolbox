"""
Ensemble inference

Copyright (c) 2023 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import logging
from pathlib import Path

from ivadomed import inference as imed_inference
import nibabel as nib
import numpy as np

from spinalcordtoolbox.utils.fs import tmp_create, extract_fname
from spinalcordtoolbox.utils.sys import run_proc

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


def segment_monai(path_model, input_filenames, threshold):
    """
    Script to run inference on a MONAI-based model for contrast-agnostic soft segmentation of the spinal cord.

    Author: Naga Karthik

    """

    import argparse
    import json
    import os
    from time import time

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
    from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
    from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
    from loguru import logger
    from monai.data import (DataLoader, CacheDataset, load_decathlon_datalist, decollate_batch)
    from monai.inferers import sliding_window_inference
    from monai.transforms import (Compose, EnsureTyped, Invertd, SaveImage, Spacingd,
                                  LoadImaged, NormalizeIntensityd, EnsureChannelFirstd,
                                  DivisiblePadd, Orientationd, ResizeWithPadOrCropd)

    # NNUNET global params
    INIT_FILTERS = 32
    ENABLE_DS = True

    nnunet_plans = {
        "UNet_class_name": "PlainConvUNet",
        "UNet_base_num_features": INIT_FILTERS,
        "n_conv_per_stage_encoder": [2, 2, 2, 2, 2, 2],
        "n_conv_per_stage_decoder": [2, 2, 2, 2, 2],
        "pool_op_kernel_sizes": [
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [1, 2, 2]
        ],
        "conv_kernel_sizes": [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3]
        ],
        "unet_max_num_features": 320,
    }

    def get_parser():
        parser = argparse.ArgumentParser(description="Run inference on a MONAI-trained model")

        parser.add_argument("--path-img", type=str, required=True,
                            help="Path to the image to run inference on")
        parser.add_argument("--chkp-path", type=str, required=True, help="Path to the checkpoint folder")
        parser.add_argument("--path-out", type=str, required=True,
                            help="Path to the output folder where to store the predictions and associated metrics")
        parser.add_argument('-crop', '--crop-size', type=str, default="64x192x-1",
                            help='Size of the window used to crop the volume before inference (NOTE: Images are resampled'
                                 ' to 1mm isotropic before cropping). The window is centered in the middle of the volume.'
                                 ' Dimensions are in the order R-L, A-P, I-S. Use -1 for no cropping in a specific axis,'
                                 ' example: “64x160x-1”. NOTE: heavy R-L cropping is recommended for positioning the SC at'
                                 ' the center of the image. Default: 64x192x-1')
        parser.add_argument('--device', default="gpu", type=str, choices=["gpu", "cpu"],
                            help='Device to run inference on. Default: cpu')

        return parser

    # ===========================================================================
    #                          Test-time Transforms
    # ===========================================================================
    def inference_transforms_single_image(crop_size):
        return Compose([
            LoadImaged(keys=["image"], image_only=False),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RPI"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=2),
            ResizeWithPadOrCropd(keys=["image"], spatial_size=crop_size, ),
            DivisiblePadd(keys=["image"], k=2 ** 5),
            # pad inputs to ensure divisibility by no. of layers nnUNet has (5)
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        ])

    # ===========================================================================
    #                              Model utils
    # ===========================================================================
    class InitWeights_He(object):
        def __init__(self, neg_slope=1e-2):
            self.neg_slope = neg_slope

        def __call__(self, module):
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
                module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
                if module.bias is not None:
                    module.bias = nn.init.constant_(module.bias, 0)

    # ============================================================================
    #               Define the network based on nnunet_plans dict
    # ============================================================================
    def create_nnunet_from_plans(plans, num_input_channels: int, num_classes: int, deep_supervision: bool = True):
        """
        Adapted from nnUNet's source code:
        https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/utilities/get_network_from_plans.py#L9

        """
        num_stages = len(plans["conv_kernel_sizes"])

        dim = len(plans["conv_kernel_sizes"][0])
        conv_op = convert_dim_to_conv_op(dim)

        segmentation_network_class_name = plans["UNet_class_name"]
        mapping = {
            'PlainConvUNet': PlainConvUNet,
            'ResidualEncoderUNet': ResidualEncoderUNet
        }
        kwargs = {
            'PlainConvUNet': {
                'conv_bias': True,
                'norm_op': get_matching_instancenorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
            },
            'ResidualEncoderUNet': {
                'conv_bias': True,
                'norm_op': get_matching_instancenorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
            }
        }
        assert segmentation_network_class_name in mapping.keys(), 'The network architecture specified by the plans file ' \
                                                                  'is non-standard (maybe your own?). Yo\'ll have to dive ' \
                                                                  'into either this ' \
                                                                  'function (get_network_from_plans) or ' \
                                                                  'the init of your nnUNetModule to accomodate that.'
        network_class = mapping[segmentation_network_class_name]

        conv_or_blocks_per_stage = {
            'n_conv_per_stage'
            if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': plans["n_conv_per_stage_encoder"],
            'n_conv_per_stage_decoder': plans["n_conv_per_stage_decoder"]
        }

        # network class name!!
        model = network_class(
            input_channels=num_input_channels,
            n_stages=num_stages,
            features_per_stage=[min(plans["UNet_base_num_features"] * 2 ** i,
                                    plans["unet_max_num_features"]) for i in range(num_stages)],
            conv_op=conv_op,
            kernel_sizes=plans["conv_kernel_sizes"],
            strides=plans["pool_op_kernel_sizes"],
            num_classes=num_classes,
            deep_supervision=deep_supervision,
            **conv_or_blocks_per_stage,
            **kwargs[segmentation_network_class_name]
        )
        model.apply(InitWeights_He(1e-2))
        if network_class == ResidualEncoderUNet:
            model.apply(init_last_bn_before_add_to_0)

        return model

    # ===========================================================================
    #                   Prepare temporary dataset for inference
    # ===========================================================================
    def prepare_data(path_image, path_out, crop_size=(64, 160, 320)):
        # create a temporary datalist containing the image
        # boiler plate keys to be defined in the MSD-style datalist
        params = {}
        params["description"] = "my-awesome-SC-image"
        params["labels"] = {
            "0": "background",
            "1": "soft-sc-seg"
        }
        params["modality"] = {
            "0": "MRI"
        }
        params["tensorImageSize"] = "3D"
        params["test"] = [
            {
                "image": path_image
            }
        ]

        final_json = json.dumps(params, indent=4, sort_keys=True)
        jsonFile = open(path_out + "/temp_msd_datalist.json", "w")
        jsonFile.write(final_json)
        jsonFile.close()

        dataset = os.path.join(path_out, "temp_msd_datalist.json")
        test_files = load_decathlon_datalist(dataset, True, "test")

        # define test transforms
        transforms_test = inference_transforms_single_image(crop_size=crop_size)

        # define post-processing transforms for testing; taken (with explanations) from
        # https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/torch/unet_inference_dict.py#L66
        test_post_pred = Compose([
            EnsureTyped(keys=["pred"]),
            Invertd(keys=["pred"], transform=transforms_test,
                    orig_keys=["image"],
                    meta_keys=["pred_meta_dict"],
                    nearest_interp=False, to_tensor=True),
        ])
        test_ds = CacheDataset(data=test_files, transform=transforms_test, cache_rate=0.75, num_workers=8)

        return test_ds, test_post_pred

    # ===========================================================================
    #                           Inference method
    # ===========================================================================
    def main(argv):
        parser = get_parser()
        args = parser.parse_args(argv)

        # define device
        if args.device == "gpu" and not torch.cuda.is_available():
            logger.warning("GPU not available, using CPU instead")
            DEVICE = torch.device("cpu")
        else:
            DEVICE = torch.device("cuda" if torch.cuda.is_available() and args.device == "gpu" else "cpu")

        # define root path for finding datalists
        path_image = args.path_img
        results_path = args.path_out
        chkp_path = os.path.join(args.chkp_path, "best_model_loss.ckpt")

        # save terminal outputs to a file
        logger.add(os.path.join(results_path, "logs.txt"), rotation="10 MB", level="INFO")

        logger.info(f"Saving results to: {results_path}")
        if not os.path.exists(results_path):
            os.makedirs(results_path, exist_ok=True)

        # define inference patch size and center crop size
        crop_size = tuple([int(i) for i in args.crop_size.split("x")])
        inference_roi_size = (64, 192, 320)

        # define the dataset and dataloader
        test_ds, test_post_pred = prepare_data(path_image, results_path, crop_size=crop_size)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

        # define model
        net = create_nnunet_from_plans(plans=nnunet_plans, num_input_channels=1, num_classes=1,
                                       deep_supervision=ENABLE_DS)

        # define list to collect the test metrics
        test_step_outputs = []
        test_summary = {}

        # iterate over the dataset and compute metrics
        with torch.no_grad():
            for batch in test_loader:

                # compute time for inference per subject
                start_time = time()

                # get the test input
                test_input = batch["image"].to(DEVICE)

                # this loop only takes about 0.2s on average on a CPU
                checkpoint = torch.load(chkp_path, map_location=torch.device(DEVICE))["state_dict"]
                # NOTE: remove the 'net.' prefix from the keys because of how the model was initialized in lightning
                # https://discuss.pytorch.org/t/missing-keys-unexpected-keys-in-state-dict-when-loading-self-trained-model/22379/14
                for key in list(checkpoint.keys()):
                    if 'net.' in key:
                        checkpoint[key.replace('net.', '')] = checkpoint[key]
                        del checkpoint[key]

                # load the trained model weights
                net.load_state_dict(checkpoint)
                net.to(DEVICE)
                net.eval()

                # run inference
                batch["pred"] = sliding_window_inference(test_input, inference_roi_size, mode="gaussian",
                                                         sw_batch_size=4, predictor=net, overlap=0.5, progress=False)

                # take only the highest resolution prediction
                batch["pred"] = batch["pred"][0]

                # NOTE: monai's models do not normalize the output, so we need to do it manually
                if bool(F.relu(batch["pred"]).max()):
                    batch["pred"] = F.relu(batch["pred"]) / F.relu(batch["pred"]).max()
                else:
                    batch["pred"] = F.relu(batch["pred"])

                post_test_out = [test_post_pred(i) for i in decollate_batch(batch)]

                pred = post_test_out[0]['pred'].cpu()

                # clip the prediction between 0.5 and 1
                # turns out this sets the background to 0.5 and the SC to 1 (which is not correct)
                # details: https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/issues/71
                pred = torch.clamp(pred, 0.5, 1)
                # set background values to 0
                pred[pred <= 0.5] = 0

                # get subject name
                subject_name = (batch["image_meta_dict"]["filename_or_obj"][0]).split("/")[-1].replace(".nii.gz", "")
                logger.info(f"Saving subject: {subject_name}")

                # this takes about 0.25s on average on a CPU
                # image saver class
                pred_saver = SaveImage(
                    output_dir=results_path, output_postfix="pred", output_ext=".nii.gz",
                    separate_folder=False, print_log=False)
                # save the prediction
                pred_saver(pred)

                end_time = time()
                metrics_dict = {
                    "subject_name_and_contrast": subject_name,
                    "inference_time_in_sec": round((end_time - start_time), 2),
                }
                test_step_outputs.append(metrics_dict)

            # save the test summary
            test_summary["metrics_per_subject"] = test_step_outputs

            # compute the average inference time
            avg_inference_time = np.stack([x["inference_time_in_sec"] for x in test_step_outputs]).mean()

            # store the average metrics in a dict
            avg_metrics = {
                "avg_inference_time_in_sec": round(avg_inference_time, 2),
            }
            test_summary["metrics_avg_across_cohort"] = avg_metrics

            logger.info("========================================================")
            logger.info(f"      Inference Time per Subject: {avg_inference_time:.2f}s")
            logger.info("========================================================")

            # dump the test summary to a json file
            with open(os.path.join(results_path, "test_summary.json"), "w") as f:
                json.dump(test_summary, f, indent=4, sort_keys=True)

            # free up memory
            test_step_outputs.clear()
            test_summary.clear()
            os.remove(os.path.join(results_path, "temp_msd_datalist.json"))

    nii_lst, target_lst = [], []
    for fname_in in input_filenames:
        tmp_dir = tmp_create("sct_deepseg")
        main(argv=[
            "--path-img", fname_in,
            "--chkp-path", path_model,
            "--path-out", tmp_dir
        ])
        _, fname, ext = extract_fname(fname_in)
        fname_out = os.path.join(tmp_dir, f"{fname}_pred{ext}")
        # TODO: Use API binarization function when output filetype is sct.image.Image
        run_proc(["sct_maths", "-i", fname_out, "-bin", str(threshold), "-o", fname_out])
        # TODO: Change the output filetype from Nifti1Image to sct.image.Image to mitigate #3232
        nii_lst.append(nib.load(fname_out))
        target_lst.append("_seg")

    return nii_lst, target_lst

def segment_nnunet():
    return