#%%
import random
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

from PIL import Image
import os
import ssl
import sys 

PROJECT_PATH =  os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.insert(0,PROJECT_PATH)

from augmentations.transform_utils import Transform, GaussianBlur, GaussianNoise, BrightnessModulation, ContrastModulation
from selfsup_task.hyper_utils import self_supervised_train, supervised_train

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
    "mode": 'full_network',
    "checkpoins_basepath": os.path.join(PROJECT_PATH, f"runs/test"), #path where to save the logs, change if necessary
    "selfsup_dataset_path": "F:/Datasets/chest-x-ray/COVID-19_Radiography_Dataset/", #path of the selfsup dataset, change if necessary
    "sup_dataset_path": "F:/Datasets/chest-x-ray/COVID-19_Radiography_Dataset/", #path of the supervised dataset, change if necessary
    "random_seed": 73,
    "num_epochs": 20,
    "batch_size": 128,
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    "barlow_lambda": 5e-3,
    "projector_dims": [512, 512, 512, 512],
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
                "brightness_contrast"     :[transforms.ToTensor(), BrightnessModulation(p=prob), ContrastModulation(p=prob)],
                "perspective"             :[torchvision.transforms.RandomPerspective(distortion_scale=0.15, p=prob), transforms.ToTensor()],
                "vertical_flip"           :[torchvision.transforms.RandomVerticalFlip(p=prob), transforms.ToTensor()],
}

'''
2nd order tranfromations.
Uncomment to train with the second iteration of transformations, taking into account 
more than one single extra transformation
'''
#extra_transforms = {
#                "vertical_flip-equalize"   :[
#                                            torchvision.transforms.RandomVerticalFlip(p=prob), 
#                                            torchvision.transforms.RandomEqualize(p=prob),
#                                            transforms.ToTensor()],
#
#                "vertical_flip-equalize-perspective"     :[
#                                            torchvision.transforms.RandomVerticalFlip(p=prob), 
#                                            torchvision.transforms.RandomEqualize(p=prob),
#                                            torchvision.transforms.RandomPerspective(distortion_scale=0.15, p=prob),
#                                            transforms.ToTensor()],
#
#                "vertical_flip-equalize-perspective-blur"     :[
#                                            torchvision.transforms.RandomVerticalFlip(p=prob), 
#                                            torchvision.transforms.RandomEqualize(p=prob),
#                                            GaussianBlur(p=prob),
#                                            torchvision.transforms.RandomPerspective(distortion_scale=0.15, p=prob),
#                                            transforms.ToTensor(),
#                                            GaussianNoise(p=prob),],                                            
#
#                "vertical_flip-equalize-perspective-blur-rotation"     :[
#                                            torchvision.transforms.RandomVerticalFlip(p=prob), 
#                                            torchvision.transforms.RandomEqualize(p=prob),
#                                            GaussianBlur(p=prob),
#                                            torchvision.transforms.RandomPerspective(distortion_scale=0.15, p=prob),
#                                            torchvision.transforms.RandomRotation(20),
#                                            transforms.ToTensor(),
#                                            GaussianNoise(p=prob),], 


if config["mode"] == "scan_transforms":
    '''
    Self Supervised training
    On This part we iterate through all the extra transformations and train the model in a 
    self-supervised way (barlow twins). We log the rsults and save the last and best model state dict
    '''
    for extra_T_name, extra_T in extra_transforms.items():
        print("%"*40)
        print(f"Applying transformation:  {extra_T_name}")
        print("%"*40)
        
        #Logging
        checkpoints_path = os.path.join(config["checkpoins_basepath"], f"{extra_T_name}_transform")
        #Resetting seeds to have the same initial model parameters
        torch.cuda.empty_cache()
        random.seed(config["random_seed"])
        np.random.seed(config["random_seed"])
        torch.manual_seed(config["random_seed"])

        #Define dataset with the trasnformations
        transform = transforms.Compose(base_transforms +  extra_T)

        dataset = ImageFolder(config["selfsup_dataset_path"], transform=Transform(transform, transform))

        loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=config["batch_size"],
                                            shuffle=True)
        
        self_supervised_train(config, loader, checkpoints_path)


elif config["mode"] == "linear_projector" or config["mode"] == "full_network":
    '''
    Supervised training
    On This part we iterate through all the extra transformations, we load the self-sup trained models,
    we freeze its weights (if needed), and train the model.
    '''
    for extra_T_name, extra_T in extra_transforms.items():
        print("%"*40)
        print(f"Applying transformation:  {extra_T_name}")
        print("%"*40)
        
        #Logging
        training_mode = config["mode"]
        checkpoints_path = os.path.join(config["checkpoins_basepath"], f"{extra_T_name}_ACC_{training_mode}_prediction")
        config["model_path"] = os.path.join(config["checkpoins_basepath"], f"{extra_T_name}_transform") #defining the path where the model's state_dict is saved
        
        #Define dataset with the trasnformations
        transform = transforms.Compose([
                                        transforms.Grayscale(),
                                        transforms.Resize(config["img_res"], interpolation=Image.BICUBIC),
                                        transforms.ToTensor()
                                        ])

        dataset = ImageFolder(config["sup_dataset_path"], transform)

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

        supervised_train(config, checkpoints_path, train_loader, val_loader)
        
        


# %%
