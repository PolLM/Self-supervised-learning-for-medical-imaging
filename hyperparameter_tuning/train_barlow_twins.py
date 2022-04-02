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

PROJECT_PATH =  os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.insert(0,PROJECT_PATH)

from self_sup_classes.barlow import BarlowTwins
from utils.training_utils import train_one_epoch,eval_one_epoch,scan_best_lr
from utils.logging_utils import save_checkpoint, save_dict_to_pickle
from utils.metrics import accuracy
from augmentations.transform_utils import Transform, GaussianBlur, GaussianNoise, BrightnessModulation, ContrastModulation

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

def self_supervised_train(config, barlow_lambda, projector_dims):

    projector_sizes = ''
    for i in projector_dims:
        projector_sizes += f"_{i}_"
    folder_name = str(barlow_lambda) + projector_sizes 

    checkpoints_path = os.path.join(PROJECT_PATH, f"runs/final_trainings/{folder_name}_300")
    writer = SummaryWriter(checkpoints_path)
    #Resetting seeds to have the same initial model parameters
    torch.cuda.empty_cache()
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    
    #Calling model, optimizer and scheduler
    back_model = torchvision.models.resnet18(zero_init_residual=True)
    model = BarlowTwins(barlow_lambda) #@@@
    model.add_backbone( 
                        backbone =back_model, 
                        latent_id = -2,
                        monochanel = True, 
                        backbone_name='resnet', 
                        verbose=False)

    model.add_projector(
                        projector_sizes = projector_dims,  #@@@
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

config = {
            "mode": 'self_supervised',
            "random_seed": 73,
            "num_epochs": 300,
            "batch_size": 128,
            "barlow_lambda": 5e-3,
            "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            "optimizer": "Adam",
            "lr": 2e-3,
            "optimizer_weight_decay": 1e-5,
            "transforms_prob": 0.5,
            "img_res": 224,
            "num_classes": 4,
            "num_epochs_sup": 5,
            "projector": [512,512],
            }


if config["mode"] == 'self_supervised':
    prob = 0.5
    transform = transforms.Compose([
                    transforms.Grayscale(),
                    transforms.RandomResizedCrop((config["img_res"], config["img_res"]), interpolation=Image.BICUBIC),
                    transforms.RandomHorizontalFlip(p=prob),
                    torchvision.transforms.RandomVerticalFlip(p=prob), 
                    torchvision.transforms.RandomEqualize(p=prob),
                    GaussianBlur(p=prob),
                    torchvision.transforms.RandomPerspective(distortion_scale=0.15, p=prob),
                    transforms.ToTensor(),
                    GaussianNoise(p=prob),])


    dataset = ImageFolder("F:/Datasets/chest-x-ray/COVID-19_Radiography_Dataset/", transform=Transform(transform, transform))
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=config["batch_size"],
                                        shuffle=True)
    
    #for barlow_lambda in [1e-1,5e-2,1e-2,5e-3,1e-3, 5e-4, 1e-4]:
    #   self_supervised_train(config, barlow_lambda, [512,512,512,512])
    self_supervised_train(config, barlow_lambda = config["barlow_lambda"], projector_dims=config["projector"])
