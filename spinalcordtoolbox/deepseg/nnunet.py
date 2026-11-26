import os
import glob
from typing import TYPE_CHECKING

# This is just to silence nnUNet warnings. These variables should have no purpose/effect.
# There are sadly no other workarounds at the moment, see:
# https://github.com/MIC-DKFZ/nnUNet/blob/227d68e77f00ec8792405bc1c62a88ddca714697/nnunetv2/paths.py#L21
os.environ['nnUNet_raw'] = "./nnUNet_raw"
os.environ['nnUNet_preprocessed'] = "./nnUNet_preprocessed"
os.environ['nnUNet_results'] = "./nnUNet_results"

from batchgenerators.utilities.file_and_folder_operations import join  # noqa: E402
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor   # noqa: E402
import importlib   # noqa: E402

if TYPE_CHECKING:
    import torch


def create_nnunet_from_plans(path_model, device: 'torch.device', single_fold: bool = False,
                             test_time_aug: bool = False) -> nnUNetPredictor:
    """
    When creating the nnunet for the `lesion_ms` model, if you want quicker inference using only a single fold
    (instead of the full 5-fold ensemble), set `single_fold=True`.
    """
    tile_step_size = 0.5
    # get the nnunet trainer directory
    trainer_dirs = glob.glob(os.path.join(path_model, "nnUNetTrainer*"))
    if len(trainer_dirs) != 1:
        raise FileNotFoundError(f"Could not find 'nnUNetTrainer*' directory inside model path: {path_model} "
                                "Please make sure the release keeps the nnUNet output structure intact "
                                "by also including the 'nnUNetTrainer*' directory.")
    path_model = trainer_dirs[0]
    fold_dirs = [os.path.basename(path) for path in glob.glob(os.path.join(path_model, "fold_*"))]
    if not fold_dirs:
        raise FileNotFoundError(f"No 'fold_*' directories found in model path: {path_model}")
    folds_avail = 'all' if fold_dirs == ['fold_all'] else [int(f.split('_')[-1]) for f in fold_dirs]
    if single_fold:
        # save temporary copy of available folds
        folds_avail_temp = folds_avail.copy()
        # use only fold 1 it exists for all models (as it was the best for the lesion_ms model)
        # otherwise use the first fold available
        folds_avail = [1] if 1 in folds_avail else [folds_avail[0]]
        print(f'Using single fold: {folds_avail} for inference instead of the full ensemble of {sorted(folds_avail_temp)}')

    # We prioritize 'checkpoint_final.pth', but fallback to 'checkpoint_best.pth' if not available
    checkpoints = {os.path.basename(path) for path in glob.glob(os.path.join(path_model, "**", "checkpoint_*.pth"))}
    for checkpoint_name in ['checkpoint_final.pth', 'checkpoint_best.pth']:
        if checkpoint_name in checkpoints:
            break  # Use the checkpoint that was found
    else:
        raise ValueError(f"Couldn't find 'checkpoint_final.pth' or 'checkpoint_best.pth' in {path_model}")

    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=tile_step_size,  # changing it from 0.5 to 0.9 makes inference faster
        use_gaussian=True,  # applies gaussian noise and gaussian blur
        use_mirroring=test_time_aug,  # test time augmentation by mirroring on all axes
        perform_everything_on_device=False,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    print(f'Running inference on device: {predictor.device}')

    trainer_class = load_trainer_class_if_available(path_model)

    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(path_model),
        use_folds=folds_avail,
        checkpoint_name=checkpoint_name,
        trainer_class=trainer_class
    )
    print('Model loaded successfully.')

    return predictor


def load_trainer_class_if_available(path_model):
    """
    This functions load a custom nnUNet trainer class if available in the model folder.
    """
    trainer_file = os.path.join(path_model, "trainer_class.py")
    if os.path.exists(trainer_file):
        spec = importlib.util.spec_from_file_location("trainer_class", trainer_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "get_trainer_class"):
            return module.get_trainer_class()
    return None
