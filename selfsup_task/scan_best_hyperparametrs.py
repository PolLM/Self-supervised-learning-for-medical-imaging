#%%
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
from selfsup_task.hyper_utils import self_supervised_train, supervised_train

from augmentations.transform_utils import Transform, GaussianBlur, GaussianNoise, BrightnessModulation, ContrastModulation

os.environ['KMP_DUPLICATE_LIB_OK']='True'
ssl._create_default_https_context = ssl._create_unverified_context



'''
This code is used to scan the optimal hyperparameters to train the barlow-twins model
with chest X-rays. We scan two parameters:
    - Barlow twins linear projector (projection layers at the end of the two twin networks)
    - Barlow twins lambda (defines the weight of the off-diagonal elements of the correlation matrix in the loss)

We can't use bayessian optimization since barlow twins is a self-supervised model and, therefore, we have to 
save the model and assess its performance in a downstream task. 

Also, since both variables are discrete we can't perform a random sampling search to explore the variable space.
'''

config = {
    "mode": 'self_supervised',#self_supervised , full_network, linear_projector
    "checkpoins_basepath": os.path.join(PROJECT_PATH, f"runs/hyperparams"), #path where to save the logs, change if necessary
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

PROJECTORS = [[512,512],[512,1024],[512,2048], [512,512,512,512],[512,512,512,1024], [512,512,512,2048]]
LMABDAS = [1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4]

if config["mode"] == 'self_supervised':
    '''
    Self Supervised training
    On This part we iterate through all the hyperparameter combinations and train the model in a 
    self-supervised way (barlow twins). We log the rsults and save the last and best model state dict
    '''
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


    dataset = ImageFolder(config["selfsup_dataset_path"], transform=Transform(transform, transform))
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=config["batch_size"],
                                        shuffle=True)
    
    
    for projector in PROJECTORS:
        for barlow_lambda in LMABDAS:

            config["barlow_lambda"] = barlow_lambda
            config["projector_dims"] = projector

            projector_sizes = ''
            for i in config["projector_dims"]:
                projector_sizes += f"_{i}_"
            folder_name = str(barlow_lambda) + projector_sizes
            checkpoints_path = os.path.join(config["checkpoins_basepath"], f"{folder_name}_transform")

            self_supervised_train(config, loader, checkpoints_path)

elif config["mode"] == "linear_projector" or config["mode"] == "full_network":
    '''
    Supervised training
    On This part we iterate through all the self supervised models trained with different hyperparameters, 
    we freeze its weights (if necessary), and train them in a supervised way.
    '''
    for projector in PROJECTORS:
        for barlow_lambda in LMABDAS:

            config["barlow_lambda"] = barlow_lambda
            config["projector_dims"] = projector

            training_mode = config["mode"]
            projector_sizes = ''
            for i in config["projector_dims"]:
                projector_sizes += f"_{i}_"
            folder_name = str(barlow_lambda) + projector_sizes
            checkpoints_path = os.path.join(config["checkpoins_basepath"], f"{folder_name}_transform" + f"ACC_{training_mode}_iter_prediction")
            config["model_path"] = os.path.join(config["checkpoins_basepath"], f"{folder_name}_transform")
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
