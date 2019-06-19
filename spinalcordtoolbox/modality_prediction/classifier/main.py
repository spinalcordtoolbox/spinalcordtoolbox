from tensorboardX import SummaryWriter
import time
import shutil
import sys
import pickle
import nibabel as nib
import numpy as np
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

import loader
import model as M
import utils

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


def cmd_train(context):
    """Main command do train the network.
    :param context: this is a dictionary with all data from the
                    configuration file:
                        - 'command': run the specified command (e.g. train, test)
                        - 'gpu': ID of the used GPU
                        - 'bids_path_train': list of relative paths of the BIDS folders of each training center
                        - 'bids_path_validation': list of relative paths of the BIDS folders of each validation center
                        - 'bids_path_test': list of relative paths of the BIDS folders of each test center
                        - 'batch_size'
                        - 'dropout_rate'
                        - 'batch_norm_momentum'
                        - 'num_epochs'
                        - 'initial_lr': initial learning rate
                        - 'log_directory': folder name where log files are saved
    """
    # Set the GPU
    gpu_number = context["gpu"]
    torch.cuda.set_device(gpu_number)


    # This code will iterate over the folders and load the data, filtering
    # the slices without labels and then concatenating all the datasets together

    # Training dataset -------------------------------------------------------
    ds_train = loader.BIDSIterator(context["bids_path_train"])

    print(f"Loaded {len(ds_train)} axial slices for the training set.")
    train_loader = DataLoader(ds_train, batch_size=context["batch_size"],
                              shuffle=True, pin_memory=True,
                              num_workers=1)
    
    # Validation dataset ------------------------------------------------------
    ds_val = loader.BIDSIterator(context["bids_path_validation"])

    print(f"Loaded {len(ds_val)} axial slices for the validation set.")
    val_loader = DataLoader(ds_val, batch_size=context["batch_size"],
                              shuffle=True, pin_memory=True,
                              num_workers=1)

    # Model definition ---------------------------------------------------------
    model = M.Classifier(drop_rate=context["dropout_rate"],
                       bn_momentum=context["batch_norm_momentum"])

    model.cuda()

    num_epochs = context["num_epochs"]
    initial_lr = context["initial_lr"]

    # Using SGD with cosine annealing learning rate
    optimizer = optim.SGD(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    # Write the metrics, images, etc to TensorBoard format
    writer = SummaryWriter(logdir=context["log_directory"])
    
    # Binary Cross Entropy Loss
    criterion = nn.CrossEntropyLoss()


    # Training loop -----------------------------------------------------------
    best_validation_loss = float("inf")

    lst_train_loss = []
    lst_val_loss = []
    lst_accuracy = []

    for epoch in tqdm(range(1, num_epochs+1), desc="Training"):
        start_time = time.time()

        scheduler.step()

        lr = scheduler.get_lr()[0]
        writer.add_scalar('learning_rate', lr, epoch)

        model.train()
        train_loss_total = 0.0
        num_steps = 0

        for i, batch in enumerate(train_loader):
            input_samples = batch["data"]
            input_labels = batch["label"]

            var_input = input_samples.cuda()
            var_labels = input_labels.cuda(non_blocking=True)

            outputs = model(var_input)

            loss = criterion(outputs, var_labels)
            train_loss_total += loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            num_steps += 1

        train_loss_total_avg = train_loss_total / num_steps
        lst_train_loss.append(train_loss_total_avg)

        tqdm.write(f"Epoch {epoch} training loss: {train_loss_total_avg:.4f}.")

        # Validation loop -----------------------------------------------------
        model.eval()
        val_loss_total = 0.0
        num_steps = 0

        #setting the lists for confusion matrix
        true_labels = []
        guessed_labels = []


        for i, batch in enumerate(val_loader):
            input_samples = batch["data"]
            input_labels = batch["label"]
    
            with torch.no_grad():
                var_input = input_samples.cuda()
                var_labels = input_labels.cuda(non_blocking=True)


                outputs = model(var_input)
                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, var_labels)
                val_loss_total += loss.item()

                val_accuracy += int((var_labels == preds).sum())

            num_steps += 1

        accuracy = accuracy_score(true_labels, guessed_labels)
        recall = recall_score(true_labels, guessed_labels, average='macro')
        precision = precision_score(true_labels, guessed_labels, average='macro')

           
        val_loss_total_avg = val_loss_total / num_steps
        lst_val_loss.append(val_loss_total_avg)

        tqdm.write(f"Epoch {epoch} validation loss: {val_loss_total_avg:.4f}.")
        tqdm.write(f"Epoch {epoch} accuracy : {accuracy :.4f}.")

        #add metrics for tensorboard
        writer.add_scalars('validation metrics', {
            'accuracy' : accuracy,
            'recall_avg' : recall,
            'precision_avg' : precision,
        }, epoch)

        writer.add_scalars('losses', {
            'train_loss': train_loss_total_avg,
            'val_loss': val_loss_total_avg,
        }, epoch)

        end_time = time.time()
        total_time = end_time - start_time
        tqdm.write("Epoch {} took {:.2f} seconds.".format(epoch, total_time))

        if val_loss_total_avg < best_validation_loss:
            best_validation_loss = val_loss_total_avg
            torch.save(model.state_dict(), "./"+context["log_directory"]+"/best_model.pt")

    # save final model
    torch.save(model.state_dict(), "./"+context["log_directory"]+"/final_model.pt")

    return


def cmd_test(context):

    # Set the GPU
    gpu_number = context["gpu"]
    torch.cuda.set_device(gpu_number)

    # Testing dataset -------------------------------------------------------
    ds_test = loader.BIDSIterator(context["bids_path_test"])

    print(f"Loaded {len(ds_test)} axial slices for the testing set.")
    test_loader = DataLoader(ds_test, batch_size=context["batch_size"],
                              shuffle=True, pin_memory=True,
                              num_workers=1)

    model = M.Classifier()
    model.load_state_dict(torch.load("./"+context["log_directory"]+"/best_model.pt", map_location="cuda:0"))
    model.cuda()
    model.eval()
    
    #setting the lists for confusion matrix
    true_labels = []
    guessed_labels = []

    for i, batch in enumerate(test_loader):
        input_samples = batch["data"]
        input_labels = batch["label"]
        
        true_labels += [int(x) for x in input_labels]

        with torch.no_grad():
            test_input = input_samples.cuda()
            test_labels = input_labels.cuda(non_blocking=True)            
            
            outputs = model(test_input)
            _, preds = torch.max(outputs, 1)
            
            lst_labels = [int(x) for x in preds]
            guessed_labels += lst_labels
                    
    accuracy = accuracy_score(true_labels, guessed_labels)
    recall = recall_score(true_labels, guessed_labels, average=None)
    precision = precision_score(true_labels, guessed_labels, average=None)
    
    np.set_printoptions(precision=2)
    
    if not(os.path.exists("./temp/")):
        os.makedirs("./temp/")
        
    class_names = ["T1w", "T2star", "T2w"]
    # Plot normalized confusion matrix
    plot_confusion_matrix(true_labels, guessed_labels, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig("./temp/test_cm.png")
    plot_metrics(np.array([recall, precision]), accuracy, class_names)
    plt.savefig("./temp/test_accuracy.png")

    
    tqdm.write(f"Accuracy over test slices : {accuracy}")
    tqdm.write(f"Recall over test slices : {recall}")
    tqdm.write(f"Precision over test slices : {precision}")
 

    return

def run_main():
    if len(sys.argv) <= 1:
        print("\npython main.py [config.json]\n")
        return

    with open(sys.argv[1], "r") as fhandle:
        context = json.load(fhandle)

    command = context["command"]

    if command == 'train':
        cmd_train(context)
        shutil.copyfile(sys.argv[1], "./"+context["log_directory"]+"/config_file.json")
    elif command == 'test':
        cmd_test(context)

if __name__ == "__main__":
    run_main()


    