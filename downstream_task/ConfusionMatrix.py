import os
import torch
import torch.nn as nn
from torchvision.transforms import transforms
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import random
import ssl
import matplotlib.pylab as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'
ssl._create_default_https_context = ssl._create_unverified_context

from torch.utils.data import DataLoader
import numpy as np
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import sys
from torch.utils.tensorboard import SummaryWriter


PROJECT_PATH =  os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.insert(0,PROJECT_PATH)


@torch.no_grad()
def CreateConfusionMatrix(model, dataloader_valid, config):
    '''
        Takes one model, executes the validation for one epoch and computes the confusion matrix for a 4 classes classification problem.
        
    :param model: nn.Model object to extract the confusion matrix from.
    :param dataloader_valid: DataLoader object containing the validation dataset.
    :param config: Configuration dictionary containing at least a 'device' type entry.
    '''
    correct_pred = 0
    total_pred = 0
    pred_matrix = np.zeros((4,4))

    # softmax to be added
    softmax = nn.Softmax()
    for i, (images,labels) in enumerate(dataloader_valid):
        images, labels = images.to(config['device']), labels.to(config['device'])
        outputs = softmax(model(images))

        _, predicted = torch.max(outputs.data,1)


        pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

        for elem in range(len(pred)):
            pred_matrix[labels[elem]][pred[elem]] += 1

        total_pred += labels.size(0)
        correct_pred += pred.eq(labels.view_as(pred)).sum().item()

    return correct_pred/total_pred,pred_matrix


model_dict = "..."

batch_size = 128
num_classes = 4

config = {
    "mode": 'scan_scheduler',
    "random_seed": 73,
    "num_epochs": 25,
    "batch_size": 128,
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    "barlow_lambda": 5e-3,
    "optimizer": "Adam",
    "lr": 2e-3, #1e-4 for better results
    "optimizer_weight_decay": 1e-5,
}

np.random.seed(config['random_seed'])

transform = transforms.Compose([  
    transforms.Grayscale(),
    transforms.Resize(224),
    # you can add other transformations in this list
    transforms.ToTensor()
])

dataset = ImageFolder("/content/COVID-19_Radiography_Dataset/",transform)


test_len = int(len(dataset)*0.1) 
val_len = int(len(dataset)*0.1)
train_len = len(dataset)-test_len-val_len
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len,val_len,test_len])

val_dataloader = DataLoader(test_dataset,batch_size=128)

model = load_barlowmodelwithweights_supervised(model_dict).to(config["device"])

_, matrix = valid_one_epoch(model,val_dataloader,config)
  
