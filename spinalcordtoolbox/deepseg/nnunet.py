import os
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
    use_best_checkpoint = False
    tile_step_size = 0.5
    folds_avail = 'all' if os.listdir(path_model) == 'fold_all' else \
        [int(f.split('_')[-1]) for f in os.listdir(path_model) if f.startswith('fold_')]

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
        checkpoint_name='checkpoint_final.pth' if not use_best_checkpoint else 'checkpoint_best.pth',
    )
    print('Model loaded successfully. Fetching test data...')

    return predictor
