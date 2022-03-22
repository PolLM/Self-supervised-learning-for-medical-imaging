import torch
import numpy as np

'''
Training for one epoch. 
If return_targets=True we assume we are training a supervised model, therefore, we return the target and predicted labels
for computing other metrics like accuracy, precision, recall, ...
Notice that depending if the task is supervised or not, we expect the data_loader and the loss to input and output
different parameters.
'''
def train_one_epoch(train_loader, model, optimizer, criterion, config, scheduler=False , verbose=False, return_targets = False):
    model.train()
    losses = []
    targets = []
    predictions = []
    for data in train_loader:
        if return_targets:
            x = data[0]
            y = data[1]
        else:
            x = data

        x = x.float()
        x = x.to(config["device"])

        output = model(x)
        if return_targets:
            loss = criterion(output, y)
            targets += torch.flatten(y).toliost()
            predictions += torch.flatten(output).tolist()
        else:
            loss = criterion(output)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if scheduler:
            scheduler.step(loss.item()/x.shape[0])

        losses.append(loss.data.item()/x.shape[0])
    
    if return_targets: 
        return(losses, targets, predictions)
    else:
        return(losses)

# def test one epoch ...


'''
Scanning the most optimal learning rate
If return_targets=True we assume we are scanning the lr for a supervised model. 
Notice that depending if the task is supervised or not, we expect the data_loader and the loss to input and output
different parameters.
For this function we used part of the code from aidl-2022 lab code called: lab_optimizers created by Daniel Fojo
'''
def scan_best_lr(data_loader, model, optimizer, criterion, config,  return_targets = False, loss_scan_range = [-9, 0]):
    steps = len(data_loader)
    loss_history = []
    lr_range = np.logspace(loss_scan_range[0], loss_scan_range[1], num=steps)

    for lr, data in zip(lr_range, data_loader):
        if return_targets:
            x = data[0]
            y = data[1]
        else:
            x = data

        x = x.float()
        x = x.to(config["device"])
        optimizer.param_groups[0]["lr"] = lr
        optimizer.zero_grad()
        output = model(x)
        if return_targets:
            loss = criterion(output, y)
        else:
            loss = criterion(output)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item()/x.shape[0])
    return(lr_range, loss_history)
