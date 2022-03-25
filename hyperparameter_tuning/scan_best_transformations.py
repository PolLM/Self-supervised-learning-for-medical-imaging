#%%
import random
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pylab as plt
from PIL import Image
import os
import ssl
import sys 
import json

PROJECT_PATH =  os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.insert(0,PROJECT_PATH)

from self_sup_classes.barlow import BarlowTwins
from utils.training_utils import train_one_epoch
from utils.logging_utils import save_checkpoint, save_dict_to_pickle
from augmentations.transform_utils import Transform, GaussianBlur, GaussianNoise, BrightnessModulation, ContrastModulation

os.environ['KMP_DUPLICATE_LIB_OK']='True'
ssl._create_default_https_context = ssl._create_unverified_context


'''
This code is used to find the best set of images transformations for the BarlowTwins 
model applied to Chest X-Ray images. 

We are going to take into consideration the following transformations, extracted from the papers:
https://arxiv.org/pdf/2101.04909.pdf
https://arxiv.org/pdf/2006.13276.pdf
https://arxiv.org/pdf/2101.05224.pdf

##################################################
#### (1) random resizing/cropping
 The cropping from the random resizing/cropping augmentation was done at an image scale uniformly 
 distributed between 20% and 100% the size of the original image.
 Google paper: random crop to 224×224 pixels

#### (2) random horizontal flipping, 

#### (3) random vertical flipping, 

#### (4) random Gaussian blur, 
For the blur augmentation, we applied the following normalized Gaussian kernel: g(x, y) = 1 σkernel 
 √ 2π exp  − 1 2 x 2 + y 2 σ 2 kernel  , (3) where σ was selected for each sample uniformly at random 
 between 0.1 and 2.0 pixels.

#### (5) Gaussian noise addition, 
We selected the standard deviation for the noise addition randomly according to the following formula: 
 σnoise = µimage SNR , (4) where SNR was selected uniformly between 4 and 8 for each sample and µimage 
 was the average pixel value of the input sample image

#### (6) histogram normalization.

#### (7) random rotation 
by angle δ ∼ U(−20, 20) degree

#### (8) random additive brightness modulation
 Random additive brightness modulation adds a δ ∼ U(−0.2, 0.2) to all channels

#### (9) random multiplicative contrast modulation
 Random multiplicative contrast modulation multiplies per-channel standard deviation by a factor s ∼ U(−0.2, 0.2)

#### (10) change of perspective

##################################################
Due to the ammount of possibilities we have decided to proceed according to the following strategy:
1. We start with a set of base transformations:
    - random resizing/cropping
    - random horizontal flipping, 

2. To this set of base tranformations we will do one extra tranformation and train the model,
and we will iterate through all the extra transformations. Set of extra transformations:
    - random Gaussian blur + Gaussian noise addition
    - histogram normalization
    - random rotation
    - random additive brightness modulation + random multiplicative contrast modulation
    - change of perspective
    - random vertical flipping

3. Once done thnis, we will try combinations of the base transformations + the most successful "extra"
transformations. 

##################################################
'''



config = {
    "mode": 'scan_transforms',
    "random_seed": 73,
    "num_epochs": 1,
    "batch_size": 128,
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    "barlow_lambda": 5e-3,
    "optimizer": "Adam",
    "lr": 2e-3,
    "optimizer_weight_decay": 1e-5,
    "transforms_prob": 0.5,
    "img_res": 224
}

prob = config["transforms_prob"]

base_transforms = [
                transforms.Grayscale(),
                transforms.RandomResizedCrop((config["img_res"], config["img_res"]), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=prob),
                
                ]

extra_transforms = {
                "None"                    :[transforms.ToTensor()],
                "gaussian_blur_noise"     :[GaussianBlur(p=prob), transforms.ToTensor(), GaussianNoise(p=prob)],
                "equalize"                :[torchvision.transforms.RandomEqualize(p=prob), transforms.ToTensor()],
                "rotation"                :[torchvision.transforms.RandomRotation(20), transforms.ToTensor()],
                "crightness_contrast"     :[transforms.ToTensor(), BrightnessModulation(p=prob), ContrastModulation(p=prob)],
                "perspective"             :[torchvision.transforms.RandomPerspective(distortion_scale=0.15, p=prob), transforms.ToTensor()],
                "vertical_flip"           :[torchvision.transforms.RandomVerticalFlip(p=prob), transforms.ToTensor()],
}


for extra_T_name, extra_T in extra_transforms.items():
    print("%"*40)
    print(f"Applying transformation:  {extra_T_name}")
    print("%"*40)
    
    #Logging
    checkpoints_path = os.path.join(PROJECT_PATH, f"runs/transforms/{extra_T_name}_transform")
    writer = SummaryWriter(checkpoints_path)
    #Resetting seeds to have the same initial model parameters
    torch.cuda.empty_cache()
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    #Define dataset with the trasnformations
    transform = transforms.Compose(base_transforms +  extra_T)

    dataset = ImageFolder("F:/Datasets/chest-x-ray/COVID-19_Radiography_Dataset/", transform=Transform(transform, transform))

    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=config["batch_size"],
                                        shuffle=False)
    
    #Calling model, optimizer and scheduler
    back_model = torchvision.models.resnet18(zero_init_residual=True)
    model = BarlowTwins(config["barlow_lambda"])
    model.add_backbone( 
                        backbone =back_model, 
                        latent_id = -2,
                        monochanel = True, 
                        backbone_name='resnet', 
                        verbose=False)

    model.add_projector(
                        projector_sizes = [512, 512, 512, 512], 
                        verbose = False) 


    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["optimizer_weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"], verbose=True)

    #Self-supervised training loop
    total_loss = []
    best_loss = 100000
    for epoch in range(config["num_epochs"]):     
        losses = train_one_epoch(loader, model, optimizer, config)
        scheduler.step()
        
        mean_epoch_loss = np.mean(losses)
        total_loss.append(mean_epoch_loss)
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

    #Supervised training loop


# %%
