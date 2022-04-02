#%%
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

config = {
    "mode": 'linear_projector',#'scan_transforms',
    "random_seed": 73,
    "num_epochs": 20,
    "batch_size": 128,
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    "barlow_lambda": 5e-3,
    "optimizer": "Adam",
    "lr": 2e-3,
    "optimizer_weight_decay": 1e-5,
    "transforms_prob": 0.5,
    "img_res": 224,
    "num_classes": 4,
    "num_epochs_sup": 5,
    "train_frac": 0.8,
    "test_frac": 0.1,
    "val_frac": 0.1,
    "lr_sup": 1e-4,
    "batch_size_sup": 64,
}

prob = config["transforms_prob"]

transform1 = transforms.Compose([
                    transforms.Grayscale(),
                    transforms.Resize(config["img_res"], interpolation=Image.BICUBIC),
                    transforms.ToTensor()
                ])
transform2 = transforms.Compose([
                transforms.Grayscale(),
                transforms.RandomResizedCrop((config["img_res"], config["img_res"]), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=prob),
                torchvision.transforms.RandomVerticalFlip(p=prob), 
                torchvision.transforms.RandomEqualize(p=prob),
                GaussianBlur(p=prob),
                torchvision.transforms.RandomPerspective(distortion_scale=0.15, p=prob),
                #torchvision.transforms.RandomRotation(20),
                transforms.ToTensor(),
                GaussianNoise(p=prob),])

dataset = ImageFolder("F:/Datasets/chest-x-ray/COVID-19_Radiography_Dataset/", transform=Transform(transform1, transform2))

loader = torch.utils.data.DataLoader(dataset,
                                    batch_size=config["batch_size"],
                                    shuffle=True)


for data in loader:
    x1 = data[0][0]
    x2 = data[0][1]

    fig, ax = plt.subplots(10,2, figsize = (10,32), )
    for i in range(10):
        ax[i,0].imshow(x1[i,0])
        ax[i,1].imshow(x2[i,0])
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    A=P
                
   
# %%
