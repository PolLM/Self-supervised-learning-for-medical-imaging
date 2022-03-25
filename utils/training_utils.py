import torch
import numpy as np
from tqdm import tqdm

'''
Training for one epoch. 
If return_targets=True we assume we are training a supervised model, therefore, we return the target and predicted labels
for computing other metrics like accuracy, precision, recall, ...
Notice that depending if the task is supervised or not, we expect the data_loader and the loss to input and output
different parameters.
'''
def train_one_epoch(train_loader, model, optimizer, config, criterion=False, return_targets=False):
    model.to(config["device"])
    model.train()
    losses = []
    targets = []
    predictions = []
    for data in tqdm(train_loader):
        if return_targets:
            x = data[0]
            y = data[1]

            x = x.float()
            x, y = x.to(config["device"]), y.to(config["device"])
            batch_s = x.shape[0]

            output = model(x)
            loss = criterion(output, y)
            targets += torch.flatten(y).toliost()
            predictions += torch.flatten(output).tolist()
            losses.append(loss.item()/batch_s)
        else:
            x1 = data[0][0]
            x2 = data[0][1]

            x1, x2 = x1.float(), x2.float()
            x1, x2 = x1.to(config["device"]), x2.to(config["device"])
            batch_s = x1.shape[0]
        
            loss, on_diag, off_diag = model(x1, x2)
            losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        
    
    if return_targets: 
        return(losses, targets, predictions)
    else:
        return(losses)

'''
Testing/ealuating for one epoch. 
If return_targets=True we assume we are training a supervised model, therefore, we return the target and predicted labels
for computing other metrics like accuracy, precision, recall, ...
Notice that depending if the task is supervised or not, we expect the data_loader and the loss to input and output
different parameters.
'''
@torch.no_grad()
def eval_one_epoch(eval_loader, model, optimizer, config, criterion=False):
    model.to(config["device"])
    model.evaluation()
    losses = []
    targets = []
    predictions = []
    for data in tqdm(eval_loader):
        x = data[0]
        y = data[1]

        x = x.float()
        x, y = x.to(config["device"]), y.to(config["device"])
        batch_s = x.shape[0]

        output = model(x)
        loss = criterion(output, y)
        targets += torch.flatten(y).toliost()
        predictions += torch.flatten(output).tolist()
        losses.append(loss.item()/batch_s)
        return(losses, targets, predictions)

        


'''
Scanning the most optimal learning rate
If return_targets=True we assume we are scanning the lr for a supervised model. 
Notice that depending if the task is supervised or not, we expect the data_loader and the loss to input and output
different parameters.
For this function we used part of the code from aidl-2022 lab code called: lab_optimizers created by Daniel Fojo
'''
def scan_best_lr(data_loader, model, optimizer, config, criterion=False,  return_targets=False, loss_scan_range = [-8, -1]):
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
            loss_history.append(loss.item()/batch_s)
        else:
            x1 = data[0][0]
            x2 = data[0][1]

            x1, x2 = x1.float(), x2.float()
            x1, x2 = x1.to(config["device"]), x2.to(config["device"])
            batch_s = x1.shape[0]

            loss, on_diag, off_diag = model(x1, x2)
            loss_history.append(loss.item())

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
