import torch
import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
import pandas as pd

import os
import sys
PROJECT_PATH =  os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.insert(0,PROJECT_PATH)

from utils.metrics import accuracy



def train_one_epoch(train_loader, model, optimizer, config, epoch, criterion=False, return_targets=False, writer = False):
    '''
    :param train_loader: Loader of the training dataset
    :param model: model to train
    :param optimizer: optimizer used in the training
    :param config: dictionary that contains some parameters needed for the training
    :param epoch: current epoch of the training (useful if reporting to tensorboard)
    :param criterion: Loss function 
    :param return_targets: If True, appart from the loss, return targets and predictions
    :param writer: tensorboard writer to log the results

    Training for one epoch. 
    If return_targets=True we assume we are training a supervised model, therefore, we return the target and predicted labels
    for computing other metrics like accuracy, precision, recall, ...
    Notice that depending if the task is supervised or not, we expect the data_loader and the loss to input and output
    different parameters.
    '''
    model.to(config["device"])
    model.train()
    losses = []
    targets = []
    predictions = []
    len_data = len(train_loader)
    for batch, data in enumerate(tqdm(train_loader)):
        if return_targets:
            x = data[0]
            y = data[1]

            x = x.float()
            x, y = x.to(config["device"]), y.to(config["device"])
            batch_s = x.shape[0]

            output = model(x)
            loss = criterion(output, y)
            targets.append(y.cpu().detach().numpy())
            predictions.append(output.cpu().detach().numpy())
            losses.append(loss.item())
            if writer:
                writer.add_scalar('Loss/train', loss.item(), epoch*len_data + batch)
                writer.add_scalar('Acc/train', accuracy(y.cpu(), output.cpu()), epoch*len_data + batch)
            

        else:
            x1 = data[0][0]
            x2 = data[0][1]

            x1, x2 = x1.float(), x2.float()
            x1, x2 = x1.to(config["device"]), x2.to(config["device"])
            batch_s = x1.shape[0]
        
            loss, on_diag, off_diag = model(x1, x2)
            losses.append(loss.item())
            if writer:
                writer.add_scalar('Loss/train', loss.item(), epoch*len_data + batch)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if return_targets: 
        return(losses, targets, predictions)
    else:
        return(losses)


@torch.no_grad()
def eval_one_epoch(eval_loader, model, config, criterion, epoch, writer = False):
    '''
    :param eval_loader: Loader of the validation dataset
    :param model: model to validate
    :param config: dictionary that contains some parameters needed for the validation
    :param epoch: current epoch of the training (useful if reporting to tensorboard)
    :param criterion: Loss function 
    :param writer: tensorboard writer to log the results

    Testing/ealuating for one epoch. 
    If return_targets=True we assume we are training a supervised model, therefore, we return the target and predicted labels
    for computing other metrics like accuracy, precision, recall, ...
    Notice that depending if the task is supervised or not, we expect the data_loader and the loss to input and output
    different parameters.
    '''
    model.to(config["device"])
    model.eval()
    losses = []
    targets = []
    predictions = []
    len_data = len(eval_loader)
    for batch, data in enumerate(tqdm(eval_loader)):
        x = data[0]
        y = data[1]

        x = x.float()
        x, y = x.to(config["device"]), y.to(config["device"])
        batch_s = x.shape[0]

        output = model(x)
        loss = criterion(output, y)
        targets.append(y.cpu().detach().numpy())
        predictions.append(output.cpu().detach().numpy())
        losses.append(loss.item())
        if writer:
            writer.add_scalar('Loss/eval', loss.item(), epoch*len_data + batch)
            writer.add_scalar('Acc/eval', accuracy(y.cpu(), output.cpu()), epoch*len_data + batch)
    return(losses, targets, predictions)

        



def scan_best_lr(data_loader, model, optimizer, config, criterion=False,  return_targets=False, loss_scan_range = [-8, -1]):
    '''
    :param data_loader: Loader of the dataset
    :param model: model to scan best lr
    :param optimizer: optimizer used in the scan
    :param config: dictionary that contains some parameters needed for the scan
    :param criterion: Loss function 
    :param return_targets: If True, appart from the loss, return targets and predictions
    :param loss_scan_range: range (powers of 10) of the lr scan

    Scanning the most optimal learning rate
    If return_targets=True we assume we are scanning the lr for a supervised model. 
    Notice that depending if the task is supervised or not, we expect the data_loader and the loss to input and output
    different parameters.
    For this function we used part of the code from aidl-2022 lab code called: lab_optimizers created by Daniel Fojo
    '''
    model.to(config["device"])
    steps = len(data_loader)-1
    loss_history = []
    lr_range = np.logspace(loss_scan_range[0], loss_scan_range[1], num=steps)

    for i, (lr, data) in enumerate(zip(tqdm(lr_range), data_loader)):
        if i == steps:
            break
        optimizer.param_groups[0]["lr"] = lr
        optimizer.zero_grad()

        if return_targets:
            x = data[0]
            y = data[1]

            x = x.float()
            x, y = x.to(config["device"]), y.to(config["device"])
            batch_s = x.shape[0]

            output = model(x)
            loss = criterion(output, y)
            #loss_history.append(loss.item()/batch_s)
        else:
            x1 = data[0][0]
            x2 = data[0][1]

            x1, x2 = x1.float(), x2.float()
            x1, x2 = x1.to(config["device"]), x2.to(config["device"])
            batch_s = x1.shape[0]

            loss, on_diag, off_diag = model(x1, x2)
            #loss_history.append(loss.item())

        loss.backward()
        optimizer.step()
        loss_history.append(loss.item()/batch_s)
    return(lr_range, loss_history)


def split_dataset():
    xray=pd.read_csv('data/Frontal_Train.csv')
    test = xray.sample(n=500)
    train = xray.drop(test.index)

    test.to_csv("test.csv",index=False)
    train.to_csv("train.csv",index=False)



def freeze_model(model, str_pattern="fc."):
    '''
    Freeze the model except the layers that contain str_pattern
    '''
    #Freeze all parameters but the last one
    for name, param in model.named_parameters():
        if str_pattern in name:
            continue
        else:
            param.requires_grad = False
    return(model)


def load_resnet18_with_barlow_weights(barlow_state_dict_path, num_classes = 4):
    '''
    Initialize resnet18 model and load weights from barlow twins model
    '''
    #Calling resnet model
    model = torchvision.models.resnet18(zero_init_residual=True)
    model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)

    #loading state dict and adapting it for the model (from Barlow Twins model to simple resnet model)
    if(barlow_state_dict_path != None):
        barlow_state_dict = torch.load(barlow_state_dict_path,map_location=torch.device('cpu'))
        
        state_dict = barlow_state_dict.copy()

        for k,v in barlow_state_dict.items():
            if "backbone" not in k:
                del state_dict[k]
            else:
                state_dict[k[13:]] = state_dict.pop(k)
    
        model.load_state_dict(state_dict)

    #Adapt model and add linear projector
    model.fc = nn.Sequential( nn.Linear(512, num_classes))
    return(model)


def load_barlowmodelwithweights_supervised(dict_path, num_classes = 4):
    #Calling resnet model
    model = torchvision.models.resnet18(zero_init_residual=True)
    model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    model.fc = nn.Sequential( nn.Linear(512, num_classes))

    #loading state dict and adapting it for the model (from Barlow Twins model to simple resnet model)
    barlow_state_dict = torch.load(dict_path,map_location=torch.device('cpu'))
        
    state_dict = barlow_state_dict.copy()

    model.load_state_dict(state_dict)
    #Adapt model and add linear projector
    return(model)
