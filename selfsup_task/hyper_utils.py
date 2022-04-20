import random
import numpy as np
import torch
import torchvision
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

import os
import ssl
import sys 

PROJECT_PATH =  os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.insert(0,PROJECT_PATH)

from self_sup_classes.barlow import BarlowTwins
from utils.training_utils import train_one_epoch,eval_one_epoch,scan_best_lr
from utils.logging_utils import save_checkpoint, save_dict_to_pickle

os.environ['KMP_DUPLICATE_LIB_OK']='True'
ssl._create_default_https_context = ssl._create_unverified_context

'''
Here we define the two main functions used when finding the optimal transformations and hyperparameters.
To test the performance of some hyperparameters we, first train the model in an unsupervised way (self_supervised_train func),
save the state dict, then we train the model in a supervised way (supervised_train func) and get the accuracy.
'''

def supervised_train(config, checkpoints_path, train_loader, val_loader):
    '''
    :param config: dictionary where most parameters are defined
    :param checkpoints_path: path where to save the checkpoints
    :param train_loader: loader of the training part of the dataset
    :param val_loader:loader of the validation part of the dataset

    This function trains a resnet-18 in a supervised way (using the labels)
    Notice that one cahn choose between training the entire model or only a linear projector (last layer of the model)
    '''
    #Logging
    writer = SummaryWriter(checkpoints_path)
    writer.add_scalar('params/learning rate initial', config["lr_sup"])
    writer.add_scalar('params/optimizer_weight_decay', config["optimizer_weight_decay"])
    writer.add_scalar('params/batch_size_sup', config["batch_size_sup"])

    #Resetting seeds to have the same initial model parameters
    torch.cuda.empty_cache()
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    #Calling resnet model
    model = torchvision.models.resnet18(zero_init_residual=True)
    model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)

    #loading state dict and adapting it for the model (from Barlow Twins model to simple resnet model)
    barlow_state_dict = torch.load(os.path.join(config["model_path"], "resnet18_best_state_dict.pt"))
    state_dict = barlow_state_dict.copy()

    for k,v in barlow_state_dict.items():
        if "backbone" not in k:
            del state_dict[k]
        else:
            state_dict[k[13:]] = state_dict.pop(k)

    model.load_state_dict(state_dict)

    #Adapt model and add linear projector
    model.fc = nn.Sequential( nn.Linear(512, config["num_classes"]))

    ##Freeze all parameters but the last one
    if config["mode"] == "linear_projector":
        for name, param in model.named_parameters():
            if "fc." in name:
                continue
            else:
                param.requires_grad = False

    #Criterion
    criterion = torch.nn.CrossEntropyLoss()

    #optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr_sup"], weight_decay=config["optimizer_weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs_sup"], verbose=True)

    #Supervised training loop
    total_loss = []
    total_loss_eval = []
    total_acc = []
    total_acc_eval = []
    best_loss = 100000

    for epoch in range(config["num_epochs_sup"]):    

        losses, targets, predictions = train_one_epoch(train_loader, model, optimizer, config, epoch, criterion=criterion, return_targets=True, writer=writer)
        losses_eval, targets_eval, predictions_eval = eval_one_epoch(val_loader, model, config, criterion, epoch, writer=writer)
        scheduler.step()
        
        mean_epoch_loss = np.mean(losses)
        mean_epoch_loss_eval = np.mean(losses_eval)

        total_loss += losses
        total_loss_eval += losses_eval

        if best_loss > mean_epoch_loss_eval:
            best_loss = mean_epoch_loss_eval
            save_checkpoint(model.state_dict(), checkpoints_path, f"resnet18_best_state_dict.pt")

        print(f"---> Avg epoch loss: {mean_epoch_loss}, avg epoch loss eval: {mean_epoch_loss_eval}" )


    save_dict_to_pickle(config, checkpoints_path)
    save_checkpoint(model.state_dict(), checkpoints_path, f"resnet18_final_state_dict.pt")
    save_checkpoint(torch.tensor(total_loss), checkpoints_path, f"epoch_losses.pt")
    save_checkpoint(torch.tensor(total_loss_eval), checkpoints_path, f"epoch_losses_eval.pt")
    writer.flush()
    writer.close()

def self_supervised_train(config, loader, checkpoints_path):
    '''
    :param config: dictionary where most parameters are defined
    :param checkpoints_path: path where to save the checkpoints
    :param loader: loader of the dataset

    This function trains a BarlowTwins model (using resnet-18 as backbone) in a self-supervised way
    '''

    #checkpoints_path = os.path.join(PROJECT_PATH, f"runs/hyperparams/{folder_name}_transform")
    writer = SummaryWriter(checkpoints_path)
    #Resetting seeds to have the same initial model parameters
    torch.cuda.empty_cache()
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    
    #Calling model, optimizer and scheduler
    back_model = torchvision.models.resnet18(zero_init_residual=True)
    model = BarlowTwins(config["barlow_lambda"]) #@@@
    model.add_backbone( 
                        backbone =back_model, 
                        latent_id = -2,
                        monochanel = True, 
                        backbone_name='resnet', 
                        verbose=False)

    model.add_projector(
                        projector_sizes = config["projector_dims"],  #@@@
                        verbose = False) 


    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["optimizer_weight_decay"]) #@@@
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"], verbose=True)



    #Self-supervised training loop
    total_loss = []
    best_loss = 100000
    for epoch in range(config["num_epochs"]):     
        losses = train_one_epoch(loader, model, optimizer, config, epoch)
        scheduler.step()
        
        mean_epoch_loss = np.mean(losses)
        total_loss += losses
        writer.add_scalar('Loss/train', mean_epoch_loss, epoch)
        
        if best_loss > mean_epoch_loss:
            best_loss = mean_epoch_loss
            save_checkpoint(model.state_dict(), checkpoints_path, f"resnet18_best_state_dict.pt")

        print(f"---> Avg epoch loss: {mean_epoch_loss}" )

    save_dict_to_pickle(config, checkpoints_path)
    save_checkpoint(model.state_dict(), checkpoints_path, f"resnet18_final_state_dict.pt")
    save_checkpoint(torch.tensor(total_loss), checkpoints_path, f"epoch_losses.pt")

    writer.flush()
    writer.close()    