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
from dataset_loader.CheXpertDataset import CheXpertDataset
from selfsup_task.hyper_utils import self_supervised_train, supervised_train

os.environ['KMP_DUPLICATE_LIB_OK']='True'
ssl._create_default_https_context = ssl._create_unverified_context

'''
This code trains the Barlow Twins model with the learnings from all the hyperparameter 
scans that we have performed previously. Notice that one can train the model with any
chest Xray dataset. In our case we have trained the model with two datasets:
    - Covid dataset, 300 epochs, the same dataset used to find the best self-sup hyperparameters, contains approx 21k images
    - CheXpert dataset, 60 epochs, a completely new dataset that contains approx. 190k images
'''  

config = {
    "mode": "covid_dataset", #covid_dataset,  chexpert_dataset
    "checkpoins_basepath": os.path.join(PROJECT_PATH, f"runs/final_trainings"), #path where to save the logs, change if necessary
    "selfsup_dataset_path": "F:/Datasets/chest-x-ray/COVID-19_Radiography_Dataset/", #path of the selfsup dataset, change to CheXpert if necessary
    "random_seed": 73,
    "num_epochs": 300, #60 if training CheXpert
    "batch_size": 128,
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    "barlow_lambda": 5e-3,
    "projector_dims": [512,512],
    "optimizer": "Adam",
    "lr": 2e-3,
    "optimizer_weight_decay": 1e-5,
    "transforms_prob": 0.5,
    "img_res": 224,
}

prob = config["transforms_prob"]
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


if config["mode"] == "covid_dataset":
    dataset = ImageFolder(config["selfsup_dataset_path"], transform=Transform(transform, transform))
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=config["batch_size"],
                                        shuffle=True)
elif config["mode"] == "chexpert_dataset":
    dataset = CheXpertDataset(r"F:\Datasets\CheXpert-v1.0-small\Frontal_Train.csv",r"F:\Datasets", transform=Transform(transform, transform))

    loader= torch.utils.data.DataLoader(dataset,
                                            batch_size=config["batch_size"],
                                            shuffle=True)

dataset_mode = config["mode"]
epochs = config["num_epochs"]
checkpoints_path = os.path.join(config["checkpoins_basepath"], f"final_train_{dataset_mode}_{epochs}")

self_supervised_train(config, loader, checkpoints_path)