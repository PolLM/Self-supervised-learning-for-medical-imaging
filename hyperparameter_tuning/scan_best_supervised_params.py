import random
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pylab as plt
from PIL import Image
import os
import ssl
import sys 

from ray import tune

PROJECT_PATH =  os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.insert(0,PROJECT_PATH)

from self_sup_classes.barlow import BarlowTwins
from utils.training_utils import train_one_epoch,eval_one_epoch,scan_best_lr
from utils.logging_utils import save_checkpoint, save_dict_to_pickle
from utils.metrics import accuracy
from augmentations.transform_utils import Transform, GaussianBlur, GaussianNoise, BrightnessModulation, ContrastModulation
from dataset_loader.CheXpertDataset import CheXpertDataset

os.environ['KMP_DUPLICATE_LIB_OK']='True'
ssl._create_default_https_context = ssl._create_unverified_context

'''
Pol: This code is used to scan the optimal hyperparameters to train the barlow-twins model
with chest X-rays.

We will use ray.tune library with Bayesian otpimization to get the best results with our limited
resources

We can't use bayessian optimization since barlow twins is a self-supervised model and, therefore, we have to 
save the model and assess its performance in a downstream task. 
'''

def supervised_train(config, i, model_type):
    '''
    #################################################################################################
    Supervised training
    On This part we iterate through all the extra transformations, we load the self-sup trained models,
    we freeze its weights, and t rain a linear predictor on top of it.
    #################################################################################################
    '''
    cropping = config["soft_crop"]
    folder_logs = f"runs\\final_trainings\\scan_supervised\\{model_type}_iter_{i}_crop_{cropping}"
    checkpoints_path = os.path.join(config["project_path"], folder_logs)
    
    print(config["project_path"])
    print(folder_logs)
    print(checkpoints_path)

    #Logging
    writer = SummaryWriter(checkpoints_path)
    writer.add_scalar('params/learning rate initial', config["lr_sup"])
    writer.add_scalar('params/optimizer', config["optimizer"])
    writer.add_scalar('params/optimizer_weight_decay', config["optimizer_weight_decay"])
    writer.add_scalar('params/h_flip', config["h_flip"])
    writer.add_scalar('params/soft_crop', config["soft_crop"])
    writer.add_scalar('params/batch_size_sup', config["batch_size_sup"])

    #Resetting seeds to have the same initial model parameters
    torch.cuda.empty_cache()
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    #Define dataset with the trasnformation
    if config["soft_crop"] != 0:
        transform = transforms.Compose([
                                        transforms.Grayscale(),
                                        transforms.CenterCrop(config["soft_crop"]),
                                        transforms.Resize(config["img_res"], interpolation=Image.BICUBIC),
                                        transforms.RandomHorizontalFlip(p=config["transforms_prob"]),
                                        transforms.ToTensor()
                                        ])
    else:

        transform = transforms.Compose([
                                        transforms.Grayscale(),
                                        transforms.Resize(config["img_res"], interpolation=Image.BICUBIC),
                                        transforms.RandomHorizontalFlip(p=config["transforms_prob"]),
                                        transforms.ToTensor()
                                        ])

    dataset = ImageFolder(r"F:/Datasets/chest-x-ray/COVID-19_Radiography_Dataset/", transform)

    #Test train split
    train_len = int(config["train_frac"]*len(dataset))
    val_len = int(config["test_frac"]*len(dataset))
    test_len = len(dataset) - train_len - val_len
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])

    train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config["batch_size_sup"], 
    shuffle=True)

    val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=config["batch_size_sup"], 
    shuffle=True)

    test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=config["batch_size_sup"], 
    shuffle=True)

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

    #Criterion
    criterion = torch.nn.CrossEntropyLoss()

    #optimizer and scheduler
    if config["optimizer"] == 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr_sup"], weight_decay=config["optimizer_weight_decay"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr_sup"], weight_decay=config["optimizer_weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs_sup"], verbose=True)

    #Supervised training loop
    total_loss = []
    total_loss_eval = []
    total_acc = []
    total_acc_eval = []
    best_loss = 100000
    ##Scan best lr
    #lr_range, loss_history = scan_best_lr(train_loader, model, optimizer, config, criterion=criterion,  return_targets=True)
    #print(len(lr_range), len(loss_history))
    #plt.plot(np.array(lr_range), np.array(loss_history))
    #plt.xscale("log")
    #plt.savefig(os.path.join(checkpoints_path, "scan_sup_lr.png"))
    #plt.show()


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
    #save_checkpoint(torch.tensor(total_acc), checkpoints_path, f"epoch_acc.pt")
    #save_checkpoint(torch.tensor(total_acc_eval), checkpoints_path, f"epoch_acc_eval.pt")

    writer.flush()
    writer.close()

    


pretrained_models = [
    r"D:\Documents\GitHub\aidl2022_final_project\runs\final_trainings\0.005_512__512__300",
    r"D:\Documents\GitHub\aidl2022_final_project\runs\final_trainings\0.005_512__512_CheXpert_50"
]

SCAN_ITERATIONS = 30

for j, model_path in enumerate(pretrained_models):
    if j==0:
        model_type = "covid_300"
    else:
        model_type = "chexpert_60"

    #Sample random Adam or SGD
    #Sample lr
    lr = [10**(-4 * np.random.uniform(0.5, 1)) for _ in range(SCAN_ITERATIONS)]
    weight_decay = [10**(-8 * np.random.uniform(0.5, 1)) for _ in range(SCAN_ITERATIONS)]
    optimizer = np.random.randint(0,2, SCAN_ITERATIONS)
    soft_crop = np.random.randint(0,2, SCAN_ITERATIONS)
    batch = np.random.randint(1,5, SCAN_ITERATIONS) * 32

    for i in range(SCAN_ITERATIONS):

        print(f"============={i}==============")

        config = {  "project_path": PROJECT_PATH,
            "model_path": model_path,
            "random_seed": 73,
            "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            "img_res": 224,
            "num_classes": 4,
            "num_epochs_sup": 10,        
            "train_frac": 0.8,
            "test_frac": 0.1,
            "val_frac": 0.1,
            "transforms_prob": 0.5,
            "h_flip": 1, 
            
            "optimizer": optimizer[i],#"Adam", #Adam, SDG
            "optimizer_weight_decay": weight_decay[i],#1e-5, # 0, 1e-4
            "soft_crop": soft_crop[i],#250, # o, 250
            "lr_sup": lr[i],#1e-4, # 1e-3, 1e-5
            "batch_size_sup": int(batch[i]),#64, # 32,64,128
            }
        supervised_train(config, i, model_type)