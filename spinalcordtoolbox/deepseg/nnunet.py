import os
import glob

import torch

# This is just to silence nnUNet warnings. These variables should have no purpose/effect.
# There are sadly no other workarounds at the moment, see:
# https://github.com/MIC-DKFZ/nnUNet/blob/227d68e77f00ec8792405bc1c62a88ddca714697/nnunetv2/paths.py#L21
os.environ['nnUNet_raw'] = "./nnUNet_raw"
os.environ['nnUNet_preprocessed'] = "./nnUNet_preprocessed"
os.environ['nnUNet_results'] = "./nnUNet_results"

from batchgenerators.utilities.file_and_folder_operations import join  # noqa: E402
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor   # noqa: E402


def create_nnunet_from_plans(path_model):
    tile_step_size = 0.5
    fold_dirs = [os.path.basename(path) for path in glob.glob(os.path.join(path_model, "fold_*"))]
    if not fold_dirs:
        raise FileNotFoundError(f"No 'fold_*' directories found in model path: {path_model}")
    folds_avail = 'all' if fold_dirs == ['fold_all'] else [int(f.split('_')[-1]) for f in fold_dirs]

    # We prioritize 'checkpoint_final.pth', but fallback to 'checkpoint_best.pth' if not available
    checkpoints = {os.path.basename(path) for path in glob.glob(os.path.join(path_model, "**", "checkpoint_*.pth"))}
    for checkpoint_name in ['checkpoint_final.pth', 'checkpoint_best.pth', None]:
        if checkpoint_name in checkpoints:
            break
    if checkpoint_name is None:
        raise ValueError(f"Couldn't find 'checkpoint_final.pth' or 'checkpoint_best.pth' in {path_model}")

    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=tile_step_size,  # changing it from 0.5 to 0.9 makes inference faster
        use_gaussian=True,  # applies gaussian noise and gaussian blur
        use_mirroring=False,  # test time augmentation by mirroring on all axes
        perform_everything_on_gpu=False,
        device=torch.device('cpu'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    print(f'Running inference on device: {predictor.device}')

    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(path_model),
        use_folds=folds_avail,
        checkpoint_name=checkpoint_name,
    )
    print('Model loaded successfully. Fetching test data...')

    return predictor
