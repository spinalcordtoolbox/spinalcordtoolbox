import copy
import datetime
import random
import time
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

from ivadomed import losses as imed_losses
from ivadomed import mixup as imed_mixup
from ivadomed import metrics as imed_metrics
from ivadomed import models as imed_models
from ivadomed import utils as imed_utils
from ivadomed import visualize as imed_visualize
from ivadomed.loader import utils as imed_loader_utils
from ivadomed.loader.balanced_sampler import BalancedSampler
from ivadomed.keywords import ModelParamsKW, ConfigKW, BalanceSamplesKW, TrainingParamsKW, MetadataKW, WandbKW

cudnn.benchmark = True


def train(model_params, dataset_train, dataset_val, training_params, path_output, device, wandb_params=None,
          cuda_available=True, metric_fns=None, n_gif=0, resume_training=False, debugging=False):
    """Main command to train the network.

    Args:
        model_params (dict): Model's parameters.
        dataset_train (imed_loader): Training dataset.
        dataset_val (imed_loader): Validation dataset.
        training_params (dict):
        path_output (str): Folder where log files, best and final models are saved.
        device (str): Indicates the CPU or GPU ID.
        cuda_available (bool): If True, CUDA is available.
        metric_fns (list): List of metrics, see :mod:`ivadomed.metrics`.
        n_gif (int): Generates a GIF during training if larger than zero, one frame per epoch for a given slice. The
            parameter indicates the number of 2D slices used to generate GIFs, one GIF per slice. A GIF shows
            predictions of a given slice from the validation sub-dataset. They are saved within the output path.
        resume_training (bool): Load a saved model ("checkpoint.pth.tar" in the path_output) for resume
                                training. This training state is saved everytime a new best model is saved in the log
                                directory.
        debugging (bool): If True, extended verbosity and intermediate outputs.

    Returns:
        float, float, float, float: best_training_dice, best_training_loss, best_validation_dice,
            best_validation_loss.
    """
    # Write the metrics, images, etc to TensorBoard format
    writer = SummaryWriter(log_dir=path_output)

    # Enable wandb tracking  if the required params are found in the config file and the api key is correct
    wandb_tracking = imed_utils.initialize_wandb(wandb_params)

    if wandb_tracking:
        # Collect all hyperparameters into a dictionary
        cfg = { **training_params, **model_params}

        # Get the actual project, group, and run names if they exist, else choose the temporary names as default
        project_name = wandb_params.get(WandbKW.PROJECT_NAME, "temp_project")
        group_name = wandb_params.get(WandbKW.GROUP_NAME, "temp_group")
        run_name = wandb_params.get(WandbKW.RUN_NAME, "temp_run")

        if project_name == "temp_project" or group_name == "temp_group" or run_name == "temp_run":     
            logger.info("{PROJECT/GROUP/RUN} name not found, initializing as {'temp_project'/'temp_group'/'temp_run'}")

        # Initialize WandB with metrics and hyperparameters
        wandb.init(project=project_name, group=group_name, name=run_name, config=cfg, dir=path_output)

    # BALANCE SAMPLES AND PYTORCH LOADER
    conditions = all([training_params[TrainingParamsKW.BALANCE_SAMPLES][BalanceSamplesKW.APPLIED],
                      model_params[ModelParamsKW.NAME] != "HeMIS"])
    sampler_train, shuffle_train = get_sampler(dataset_train, conditions,
                                               training_params[TrainingParamsKW.BALANCE_SAMPLES][BalanceSamplesKW.TYPE])

    train_loader = DataLoader(dataset_train, batch_size=training_params[TrainingParamsKW.BATCH_SIZE],
                              shuffle=shuffle_train, pin_memory=True, sampler=sampler_train,
                              collate_fn=imed_loader_utils.imed_collate,
                              num_workers=0)

    gif_dict = {"image_path": [], "slice_id": [], "gif": []}
    if dataset_val:
        sampler_val, shuffle_val = get_sampler(dataset_val, conditions,
                                               training_params[TrainingParamsKW.BALANCE_SAMPLES][BalanceSamplesKW.TYPE])

        val_loader = DataLoader(dataset_val, batch_size=training_params[TrainingParamsKW.BATCH_SIZE],
                                shuffle=shuffle_val, pin_memory=True, sampler=sampler_val,
                                collate_fn=imed_loader_utils.imed_collate,
                                num_workers=0)

        # Init GIF
        if n_gif > 0:
            indexes_gif = random.sample(range(len(dataset_val)), n_gif)
        for i_gif in range(n_gif):
            random_metadata = dict(dataset_val[indexes_gif[i_gif]][MetadataKW.INPUT_METADATA][0])
            gif_dict["image_path"].append(random_metadata[MetadataKW.INPUT_FILENAMES])
            gif_dict["slice_id"].append(random_metadata[MetadataKW.SLICE_INDEX])
            gif_obj = imed_visualize.AnimatedGif(size=dataset_val[indexes_gif[i_gif]]["input"].numpy()[0].shape)
            gif_dict["gif"].append(copy.copy(gif_obj))

    # GET MODEL
    if training_params["transfer_learning"]["retrain_model"]:
        logger.info("Loading pretrained model's weights: {}.")
        logger.info("\tFreezing the {}% first layers.".format(
            100 - training_params["transfer_learning"]['retrain_fraction'] * 100.))
        old_model_path = training_params["transfer_learning"]["retrain_model"]
        fraction = training_params["transfer_learning"]['retrain_fraction']
        if 'reset' in training_params["transfer_learning"]:
            reset = training_params["transfer_learning"]['reset']
        else:
            reset = True
        # Freeze first layers and reset last layers
        model = imed_models.set_model_for_retrain(old_model_path, retrain_fraction=fraction, map_location=device,
                                                  reset=reset)
    else:
        logger.info("Initialising model's weights from scratch.")
        model_class = getattr(imed_models, model_params[ModelParamsKW.NAME])
        model = model_class(**model_params)
    if cuda_available:
        model.cuda()

    num_epochs = training_params["training_time"]["num_epochs"]

    # OPTIMIZER
    initial_lr = training_params["scheduler"]["initial_lr"]
    # filter out the parameters you are going to fine-tuning
    params_to_opt = filter(lambda p: p.requires_grad, model.parameters())
    # Using Adam
    optimizer = optim.Adam(params_to_opt, lr=initial_lr)
    scheduler, step_scheduler_batch = get_scheduler(copy.copy(training_params["scheduler"]["lr_scheduler"]), optimizer,
                                                    num_epochs)
    logger.info("Scheduler parameters: {}".format(training_params["scheduler"]["lr_scheduler"]))

    # Only call wandb methods if required params are found in the config file
    if wandb_tracking:
        # Logs gradients (at every log_freq steps) to the dashboard.
        wandb.watch(model, log="gradients", log_freq=wandb_params["log_grads_every"])

    # Resume
    start_epoch = 1
    resume_path = Path(path_output, "checkpoint.pth.tar")
    if resume_training:
        model, optimizer, gif_dict, start_epoch, val_loss_total_avg, scheduler, patience_count = load_checkpoint(
            model=model,
            optimizer=optimizer,
            gif_dict=gif_dict,
            scheduler=scheduler,
            fname=str(resume_path))
        # Individually transfer the optimizer parts
        # TODO: check if following lines are needed
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    # LOSS
    logger.info("Selected Loss: {}".format(training_params["loss"]["name"]))
    logger.info("\twith the parameters: {}".format(
        [training_params["loss"][k] for k in training_params["loss"] if k != "name"]))
    loss_fct = get_loss_function(copy.copy(training_params["loss"]))
    loss_dice_fct = imed_losses.DiceLoss()  # For comparison when another loss is used

    # INIT TRAINING VARIABLES
    best_training_dice, best_training_loss = float("inf"), float("inf")
    best_validation_loss, best_validation_dice = float("inf"), float("inf")
    patience_count = 0
    begin_time = time.time()

    # EPOCH LOOP
    for epoch in tqdm(range(num_epochs), desc="Training", initial=start_epoch):
        epoch = epoch + start_epoch
        start_time = time.time()

        lr = scheduler.get_last_lr()[0]
        writer.add_scalar('learning_rate', lr, epoch)
        if wandb_tracking:
            wandb.log({"learning_rate": lr})

        # Training loop -----------------------------------------------------------
        model.train()
        train_loss_total, train_dice_loss_total = 0.0, 0.0
        num_steps = 0
        for i, batch in enumerate(train_loader):
            # GET SAMPLES
            if model_params[ModelParamsKW.NAME] == ConfigKW.HEMIS_UNET:
                input_samples = imed_utils.cuda(imed_utils.unstack_tensors(batch["input"]), cuda_available)
            else:
                input_samples = imed_utils.cuda(batch["input"], cuda_available)
            gt_samples = imed_utils.cuda(batch["gt"], cuda_available, non_blocking=True)

            # MIXUP
            if training_params["mixup_alpha"]:
                input_samples, gt_samples = imed_mixup.mixup(input_samples, gt_samples, training_params["mixup_alpha"],
                                                             debugging and epoch == 1, path_output)

            # RUN MODEL
            if model_params[ModelParamsKW.NAME] == ConfigKW.HEMIS_UNET or \
                    (ModelParamsKW.FILM_LAYERS in model_params and any(model_params[ModelParamsKW.FILM_LAYERS])):
                metadata = get_metadata(batch[MetadataKW.INPUT_METADATA], model_params)
                preds = model(input_samples, metadata)
            else:
                preds = model(input_samples)

            # LOSS
            loss = loss_fct(preds, gt_samples)
            train_loss_total += loss.item()
            train_dice_loss_total += loss_dice_fct(preds, gt_samples).item()

            # UPDATE OPTIMIZER
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step_scheduler_batch:
                scheduler.step()
            num_steps += 1

            # Save image at every 50th step if debugging is true
            if i%50 == 0 and debugging:
                imed_visualize.save_img(writer, epoch, "Train", input_samples, gt_samples, preds,
                                                wandb_tracking=wandb_tracking,
                                                is_three_dim=not model_params[ModelParamsKW.IS_2D])

        if not step_scheduler_batch:
            scheduler.step()

        # TRAINING LOSS
        train_loss_total_avg = train_loss_total / num_steps
        msg = "Epoch {} training loss: {:.4f}.".format(epoch, train_loss_total_avg)
        train_dice_loss_total_avg = train_dice_loss_total / num_steps
        if training_params["loss"]["name"] != "DiceLoss":
            msg += "\tDice training loss: {:.4f}.".format(train_dice_loss_total_avg)
        logger.info(msg)
        tqdm.write(msg)

        # CURRICULUM LEARNING
        if model_params[ModelParamsKW.NAME] == ConfigKW.HEMIS_UNET:
            # Increase the probability of a missing modality
            model_params[ModelParamsKW.MISSING_PROBABILITY] **= model_params[ModelParamsKW.MISSING_PROBABILITY_GROWTH]
            dataset_train.update(p=model_params[ModelParamsKW.MISSING_PROBABILITY])

        # Validation loop -----------------------------------------------------
        model.eval()
        val_loss_total, val_dice_loss_total = 0.0, 0.0
        num_steps = 0
        metric_mgr = imed_metrics.MetricManager(metric_fns)
        if dataset_val:
            for i, batch in enumerate(val_loader):
                with torch.no_grad():
                    # GET SAMPLES
                    if model_params[ModelParamsKW.NAME] == ConfigKW.HEMIS_UNET:
                        input_samples = imed_utils.cuda(imed_utils.unstack_tensors(batch["input"]), cuda_available)
                    else:
                        input_samples = imed_utils.cuda(batch["input"], cuda_available)
                    gt_samples = imed_utils.cuda(batch["gt"], cuda_available, non_blocking=True)

                    # RUN MODEL
                    if model_params[ModelParamsKW.NAME] == ConfigKW.HEMIS_UNET or \
                            (ModelParamsKW.FILM_LAYERS in model_params and any(model_params[ModelParamsKW.FILM_LAYERS])):
                        metadata = get_metadata(batch[MetadataKW.INPUT_METADATA], model_params)
                        preds = model(input_samples, metadata)
                    else:
                        preds = model(input_samples)

                    # LOSS
                    loss = loss_fct(preds, gt_samples)
                    val_loss_total += loss.item()
                    val_dice_loss_total += loss_dice_fct(preds, gt_samples).item()

                    # Add frame to GIF
                    for i_ in range(len(input_samples)):
                        im, pr, met = input_samples[i_].cpu().numpy()[0], preds[i_].cpu().numpy()[0], \
                                      batch[MetadataKW.INPUT_METADATA][i_][0]
                        for i_gif in range(n_gif):
                            if gif_dict["image_path"][i_gif] == met.__getitem__('input_filenames') and \
                                    gif_dict["slice_id"][i_gif] == met.__getitem__('slice_index'):
                                overlap = imed_visualize.overlap_im_seg(im, pr)
                                gif_dict["gif"][i_gif].add(overlap, label=str(epoch))

                num_steps += 1

                # METRICS COMPUTATION
                gt_npy = gt_samples.cpu().numpy()
                preds_npy = preds.data.cpu().numpy()
                metric_mgr(preds_npy, gt_npy)

                # Save image at every 10th step if debugging is true
                if i%50 == 0 and debugging:
                    imed_visualize.save_img(writer, epoch, "Validation", input_samples, gt_samples, preds,
                                            wandb_tracking=wandb_tracking, 
                                            is_three_dim=not model_params[ModelParamsKW.IS_2D])

            # METRICS COMPUTATION FOR CURRENT EPOCH
            val_loss_total_avg_old = val_loss_total_avg if epoch > 1 else None
            metrics_dict = metric_mgr.get_results()
            metric_mgr.reset()
            val_loss_total_avg = val_loss_total / num_steps
            # log losses on Tensorboard by default
            writer.add_scalars('Validation/Metrics', metrics_dict, epoch)
            writer.add_scalars('losses', {
                'train_loss': train_loss_total_avg,
                'val_loss': val_loss_total_avg,
            }, epoch)
            # log on wandb if the corresponding dictionary is provided
            if wandb_tracking:
                wandb.log({"validation-metrics": metrics_dict})
                wandb.log({"losses": {
                    'train_loss': train_loss_total_avg,
                    'val_loss': val_loss_total_avg,
                }})
            msg = "Epoch {} validation loss: {:.4f}.".format(epoch, val_loss_total_avg)
            val_dice_loss_total_avg = val_dice_loss_total / num_steps
            if training_params["loss"]["name"] != "DiceLoss":
                msg += "\tDice validation loss: {:.4f}.".format(val_dice_loss_total_avg)
            logger.info(msg)
            end_time = time.time()
            total_time = end_time - start_time
            msg_epoch = "Epoch {} took {:.2f} seconds.".format(epoch, total_time)
            logger.info(msg_epoch)

            # UPDATE BEST RESULTS
            if val_loss_total_avg < best_validation_loss:
                # Save checkpoint
                state = {'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'gif_dict': gif_dict,
                         'scheduler': scheduler,
                         'patience_count': patience_count,
                         'validation_loss': val_loss_total_avg}
                torch.save(state, resume_path)

                # Save best model file
                model_path = Path(path_output, "best_model.pt")
                torch.save(model, model_path)

                # Update best scores
                best_validation_loss, best_training_loss = val_loss_total_avg, train_loss_total_avg
                best_validation_dice, best_training_dice = val_dice_loss_total_avg, train_dice_loss_total_avg

            # EARLY STOPPING
            if epoch > 1:
                val_diff = (val_loss_total_avg_old - val_loss_total_avg) * 100 / abs(val_loss_total_avg)
                if val_diff < training_params["training_time"]["early_stopping_epsilon"]:
                    patience_count += 1
                if patience_count >= training_params["training_time"]["early_stopping_patience"]:
                    logger.info("Stopping training due to {} epochs without improvements".format(patience_count))
                    break

    # Save final model
    final_model_path = Path(path_output, "final_model.pt")
    torch.save(model, final_model_path)

    # Save best model in output path
    if resume_path.is_file():
        state = torch.load(resume_path)
        model_path = Path(path_output, "best_model.pt")
        model.load_state_dict(state['state_dict'])
        torch.save(model, model_path)
        # Save best model as ONNX in the model directory
        try:
            # Convert best model to ONNX and save it in model directory
            best_model_path = Path(path_output, model_params[ModelParamsKW.FOLDER_NAME],
                                   model_params[ModelParamsKW.FOLDER_NAME] + ".onnx")
            imed_utils.save_onnx_model(model, input_samples, str(best_model_path))
            logger.info(f"Model saved as '.onnx': {best_model_path}")
        except Exception as e:
            logger.warning(f"Failed to save the model as '.onnx': {e}")

        # Save best model as PT in the model directory
        best_model_path = Path(path_output, model_params[ModelParamsKW.FOLDER_NAME],
                               model_params[ModelParamsKW.FOLDER_NAME] + ".pt")
        torch.save(model, best_model_path)
        logger.info(f"Model saved as '.pt': {best_model_path}")

    # Save GIFs
    gif_folder = Path(path_output, "gifs")
    if n_gif > 0 and not gif_folder.is_dir():
        gif_folder.mkdir(parents=True)
    for i_gif in range(n_gif):
        fname_out = gif_dict["image_path"][i_gif].split(os.sep)[-3] + "__"
        fname_out += gif_dict["image_path"][i_gif].split(os.sep)[-1].split(".nii.gz")[0].split(
            gif_dict["image_path"][i_gif].split(os.sep)[-3] + "_")[1] + "__"
        fname_out += str(gif_dict["slice_id"][i_gif]) + ".gif"
        path_gif_out = Path(gif_folder, fname_out)
        gif_dict["gif"][i_gif].save(str(path_gif_out))

    writer.close()
    wandb.finish()
    final_time = time.time()
    duration_time = final_time - begin_time
    logger.info('begin ' + time.strftime('%H:%M:%S', time.localtime(begin_time)) + "| End " +
          time.strftime('%H:%M:%S', time.localtime(final_time)) +
          "| duration " + str(datetime.timedelta(seconds=duration_time)))

    return best_training_dice, best_training_loss, best_validation_dice, best_validation_loss


def get_sampler(ds, balance_bool, metadata):
    """Get sampler.

    Args:
        ds (BidsDataset): BidsDataset object.
        balance_bool (bool): If True, a sampler is generated that balance positive and negative samples.

    Returns:
        If balance_bool is True: Returns BalancedSampler, Bool: Sampler and boolean for shuffling (set to False).
        Otherwise: Returns None and True.
    """
    if balance_bool:
        return BalancedSampler(ds, metadata), False
    else:
        return None, True


def get_scheduler(params, optimizer, num_epochs=0):
    """Get scheduler.

    Args:
        params (dict): scheduler parameters, see `PyTorch documentation <https://pytorch.org/docs/stable/optim.html>`__
        optimizer (torch optim):
        num_epochs (int): number of epochs.

    Returns:
        torch.optim, bool, which indicates if the scheduler is updated for each batch (True), or for each epoch (False).
    """
    step_scheduler_batch = False
    scheduler_name = params["name"]
    del params["name"]
    if scheduler_name == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    elif scheduler_name == "CosineAnnealingWarmRestarts":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **params)
    elif scheduler_name == "CyclicLR":
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, **params, mode="triangular2", cycle_momentum=False)
        step_scheduler_batch = True
    else:
        raise ValueError(
            "{} is an unknown LR Scheduler name, please choose between 'CosineAnnealingLR', "
            "'CosineAnnealingWarmRestarts', or 'CyclicLR'".format(scheduler_name))

    return scheduler, step_scheduler_batch


def get_loss_function(params):
    """Get Loss function.

    Args:
        params (dict): See :mod:`ivadomed.losses`.

    Returns:
        imed_losses object.
    """
    # Loss function name
    loss_name = params["name"]
    del params["name"]

    # Check if implemented
    loss_function_available = ["DiceLoss", "FocalLoss", "GeneralizedDiceLoss", "FocalDiceLoss", "MultiClassDiceLoss",
                               "BinaryCrossEntropyLoss", "TverskyLoss", "FocalTverskyLoss", "AdapWingLoss", "L2loss",
                               "LossCombination"]
    if loss_name not in loss_function_available:
        raise ValueError(
            "Unknown Loss function: {}, please choose between {}".format(loss_name, loss_function_available))

    loss_class = getattr(imed_losses, loss_name)
    loss_fct = loss_class(**params)
    return loss_fct


def get_metadata(metadata, model_params):
    """Get metadata during batch loop.

    Args:
        metadata (batch):
        model_params (dict):

    Returns:
        If FiLMedUnet, Returns a list of metadata, that have been transformed by the One Hot Encoder.
        If HeMISUnet, Returns a numpy array where each row represents a sample and each column represents a contrast.
    """
    if model_params[ModelParamsKW.NAME] == ConfigKW.HEMIS_UNET:
        return np.array([m[0]["missing_mod"] for m in metadata])
    else:
        return [model_params[ModelParamsKW.FILM_ONEHOTENCODER].transform([metadata[k][0]['film_input']]).tolist()[0]
                for k in range(len(metadata))]


def load_checkpoint(model, optimizer, gif_dict, scheduler, fname):
    """Load checkpoint.

    This function check if a checkpoint is available. If so, it updates the state of the input objects.

    Args:
        model (nn.Module): Init model.
        optimizer (torch.optim): Model's optimizer.
        gif_dict (dict): Dictionary containing a GIF of the training.
        scheduler (_LRScheduler): Learning rate scheduler.
        fname (str): Checkpoint filename.

    Return:
        nn.Module, torch, dict, int, float, _LRScheduler, int
    """
    start_epoch = 1
    validation_loss = 0
    patience_count = 0
    try:
        logger.info("Loading checkpoint: {}".format(fname))
        checkpoint = torch.load(fname)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        validation_loss = checkpoint['validation_loss']
        scheduler = checkpoint['scheduler']
        gif_dict = checkpoint['gif_dict']
        patience_count = checkpoint['patience_count']
        logger.info("... Resume training from epoch #{}".format(start_epoch))
    except:
        logger.warning("\nNo checkpoint found at: {}".format(fname))

    return model, optimizer, gif_dict, start_epoch, validation_loss, scheduler, patience_count
