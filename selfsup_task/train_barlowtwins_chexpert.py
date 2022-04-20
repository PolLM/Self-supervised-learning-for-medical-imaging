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

def self_supervised_train(config, barlow_lambda, projector_dims, dataset_name="CheXpert"):

    projector_sizes = ''
    for i in projector_dims:
        projector_sizes += f"_{i}_"
    folder_name = str(barlow_lambda) + projector_sizes + dataset_name

    checkpoints_path = os.path.join(PROJECT_PATH, f"runs/final_trainings/{folder_name}_50")
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
            "mode": 'linear_projector',
            "random_seed": 73,
            "num_epochs": 60,
            "batch_size": 128,
            "barlow_lambda": 5e-3,
            "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            "optimizer": "Adam",
            "lr": 2e-3,
            "optimizer_weight_decay": 1e-5,
            "transforms_prob": 0.5,
            "img_res": 224,
            "num_classes": 4,
            "num_epochs_sup": 25,
            "projector": [512,512],            
            "train_frac": 0.8,
            "test_frac": 0.1,
            "val_frac": 0.1,
            "lr_sup": 1e-4,
            "batch_size_sup": 64,
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


    dataset_train = CheXpertDataset(r"F:\Datasets\CheXpert-v1.0-small\Frontal_Train.csv",r"F:\Datasets", transform=Transform(transform, transform))

    loader= torch.utils.data.DataLoader(dataset_train,
                                            batch_size=config["batch_size"],
                                            shuffle=True)

    
    #for barlow_lambda in [1e-1,5e-2,1e-2,5e-3,1e-3, 5e-4, 1e-4]:
    #   self_supervised_train(config, barlow_lambda, [512,512,512,512])
    print(len(loader))
    self_supervised_train(config, barlow_lambda = config["barlow_lambda"], projector_dims=config["projector"])

elif config["mode"] == "linear_projector":
    '''
    #################################################################################################
    Supervised training
    On This part we iterate through all the extra transformations, we load the self-sup trained models,
    we freeze its weights, and t rain a linear predictor on top of it.
    #################################################################################################
    '''

    BASE_DIR = os.path.join(PROJECT_PATH, f"runs/final_trainings")
    for root, subdirectories, files in os.walk(BASE_DIR):
        for subdirectory in subdirectories:
            if "ACC" in subdirectory:# or "0.0001_512__512__512__512_1e-05" in subdirectory:
                continue
            if "CheXpert" not in subdirectory:
                continue

            print("%"*40)
            print(f"Applying hyperparams:  {subdirectory}")
            print("%"*40)
            
            #Logging
            selfsup_checkpoints_path = os.path.join(root, subdirectory)
            checkpoints_path = os.path.join(root, subdirectory + "ACC_CheXpert_full25_Hflip_prediction")

            writer = SummaryWriter(checkpoints_path)
            #Resetting seeds to have the same initial model parameters
            torch.cuda.empty_cache()
            random.seed(config["random_seed"])
            np.random.seed(config["random_seed"])
            torch.manual_seed(config["random_seed"])

            #Define dataset with the trasnformations
            transform = transforms.Compose([
                                            transforms.Grayscale(),
                                            transforms.Resize(config["img_res"], interpolation=Image.BICUBIC),
                                            #transforms.RandomResizedCrop((config["img_res"], config["img_res"]), interpolation=Image.BICUBIC),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.ToTensor()
                                            ])

            dataset = ImageFolder("F:/Datasets/chest-x-ray/COVID-19_Radiography_Dataset/", transform)

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
            barlow_state_dict = torch.load(os.path.join(selfsup_checkpoints_path, "resnet18_best_state_dict.pt"))
            state_dict = barlow_state_dict.copy()

            for k,v in barlow_state_dict.items():
                if "backbone" not in k:
                    del state_dict[k]
                else:
                    state_dict[k[13:]] = state_dict.pop(k)

            model.load_state_dict(state_dict)

            #Adapt model and add linear projector
            model.fc = nn.Sequential( nn.Linear(512, config["num_classes"]))

            ##Freeze all parameters but the last one
            #for name, param in model.named_parameters():
            #    if "fc." in name:
            #        continue
            #    else:
            #        param.requires_grad = False

            #Criterion
            criterion = torch.nn.CrossEntropyLoss()

            #optimizer and scheduler
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr_sup"], weight_decay=config["optimizer_weight_decay"])
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

                #acc = accuracy(targets, predictions)
                #acc_eval = accuracy(targets_eval, predictions_eval)
                
                mean_epoch_loss = np.mean(losses)
                mean_epoch_loss_eval = np.mean(losses_eval)

                total_loss += losses
                total_loss_eval += losses_eval
                #total_acc.append(acc)
                #total_acc_eval.append(acc)

                #writer.add_scalar('Loss/train', mean_epoch_loss, epoch)
                #writer.add_scalar('Loss/eval', mean_epoch_loss_eval, epoch)

                #writer.add_scalar('Acc/train', acc, epoch)
                #writer.add_scalar('Acc/eval', acc_eval, epoch)
                
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