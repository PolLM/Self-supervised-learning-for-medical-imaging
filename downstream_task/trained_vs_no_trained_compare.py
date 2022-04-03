import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models

from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
import math
from compare_networks import compare_networks
import os
from torchvision import transforms


model_dict = "/content/resnet18_20epoch_self_supervised/resnet18_best_state_dict.pt"
#batch_size = 128
batch_size = 128
num_classes = 4

ListOfCompares = [0.01,0.05,0.10,0.15,0.20,0.25]
ListOfTags=      ["1 percent of samples",
                    "5 percent of samples",
                    "10 percent of samples",
                    "15 percent of samples",
                    "20 percent of samples",
                    "25 percent of samples"]

config = {
    "mode": 'scan_scheduler',
    "random_seed": 73,
    "num_epochs": 25,
    "batch_size": 128,
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    "barlow_lambda": 5e-3,
    "optimizer": "Adam",
    "lr": 2e-3,
    "optimizer_weight_decay": 1e-5,
}

transform = transforms.Compose([
    
    transforms.Grayscale(),
    transforms.Resize(224),
    # you can add other transformations in this list
    transforms.ToTensor()
])

dataset = ImageFolder("/content/COVID-19_Radiography_Dataset/",transform)


val_len = 200
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset)-val_len, val_len])

writer = SummaryWriter(log_dir="/content/Compare_Dataset/")

# Train supervised: 25 epochs 
# provar 3 epochs només amb fc. 
# Tot descongelat 
# Dataset covid 
print(f'Total samples: {len(train_dataset)} as Training dataset')
print(f'Total samples: {len(val_dataset)} as Validation dataset')
for idx,perc in enumerate(ListOfCompares):
    
    comparison = compare_networks(model_dict,batch_size,num_classes,config)
    tr_split_len = math.floor(len(train_dataset) * perc)
    print(f'Starting loop {idx}')
    print(f'Taking {perc*100} percent of samples: {tr_split_len}')
    dataset_reduced = torch.utils.data.random_split(train_dataset, [tr_split_len, len(train_dataset)-tr_split_len])[0]
    criterion = nn.CrossEntropyLoss()
    tag = ListOfTags[idx]
    comparison.train(dataset_reduced, val_dataset, config['num_epochs'],criterion,config,writer,tag)    

writer.flush()
writer.close()
