import os
import torch

from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def create_nnunet_from_plans(path_model, use_gpu=False, use_best_checkpoint=False, tile_step_size=0.5):
    folds_avail = [int(f.split('_')[-1]) for f in os.listdir(path_model) if f.startswith('fold_')]

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

    return predictor