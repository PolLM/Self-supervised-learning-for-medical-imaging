import random
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder


from PIL import Image
import os
import ssl
import sys 


PROJECT_PATH =  os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.insert(0,PROJECT_PATH)
from selfsup_task.hyper_utils import supervised_train

os.environ['KMP_DUPLICATE_LIB_OK']='True'
ssl._create_default_https_context = ssl._create_unverified_context

'''
This code is used to scan the optimal hyperparameters to train the models in a supervised way.
At this point we have already pre-trained (self-supervised) the models and saved their weights
and now we load the models and find the most optimal hyperparameters to train them.
'''
    
config = {  
    "mode": "full_network",#full_network, linear_projector
    "checkpoins_basepath": os.path.join(PROJECT_PATH, f"runs/supervised_hyperparams"), #path where to save the logs, change if necessary
    "sup_dataset_path": "F:/Datasets/chest-x-ray/COVID-19_Radiography_Dataset/", #path of the supervised dataset, change if necessary
    "random_seed": 73,
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    "img_res": 224,
    "optimizer": "Adam",
    "num_classes": 4,       
    "train_frac": 0.8,
    "test_frac": 0.1,
    "val_frac": 0.1,
    "transforms_prob": 0.5,
    }

#Dictionary with the path to the state-dict of all the self-supervised models we want to 
#strat from our training
PRETRAINED_MODELS = {
    "Covid_300": r"D:\Documents\GitHub\aidl2022_final_project\runs\final_trainings\0.005_512__512__300",
    "CheXpert_60": r"D:\Documents\GitHub\aidl2022_final_project\runs\final_trainings\0.005_512__512_CheXpert_50"
}

#Number of trials
SCAN_ITERATIONS = 50

#Random sampling of the hyperparameters (not using ray.tune due to some problems with the gpu)
lr = [10**(-5 * np.random.uniform(0.4, 1)) for _ in range(SCAN_ITERATIONS)]
weight_decay = [10**(-8 * np.random.uniform(0.4, 1)) for _ in range(SCAN_ITERATIONS)]
num_epochs = np.random.randint(5,21, SCAN_ITERATIONS)
batch_size = np.random.randint(1,5, SCAN_ITERATIONS)*32

#Loop over pre-trained models
for model_type, model_path in PRETRAINED_MODELS.items():
    #Loop over trials
    for i in range(SCAN_ITERATIONS):

        print(f"============={i}==============")

        config["model_path"] = model_path
        config["optimizer_weight_decay"]= weight_decay[i]
        config["lr_sup"]= lr[i]
        config["num_epochs_sup"]= int(num_epochs[i])
        config["batch_size_sup"] = int(batch_size[i])

        #Resetting seeds to have the same initial model parameters
        torch.cuda.empty_cache()
        random.seed(config["random_seed"])
        np.random.seed(config["random_seed"])
        torch.manual_seed(config["random_seed"])

        #Define dataset with the trasnformation
        transform = transforms.Compose([
                                        transforms.Grayscale(),
                                        transforms.Resize(config["img_res"], interpolation=Image.BICUBIC),
                                        transforms.RandomHorizontalFlip(p=config["transforms_prob"]),
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

        mode = config["mode"]
        checkpoints_path = os.path.join(config["checkpoins_basepath"], f"scan_supervised_{model_type}_{mode}_iter_{i}")
        supervised_train(config, checkpoints_path, train_loader, val_loader)