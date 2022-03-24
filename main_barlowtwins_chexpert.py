#%%
import ssl
import os
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'
ssl._create_default_https_context = ssl._create_unverified_context

from torchvision import transforms
from self_sup_classes.barlow import *
from augmentations.transform_utils import *
import torch
from torchvision import models
#import torchvision.transforms as transforms
import torchvision.datasets as dsets
import time

from tqdm import tqdm
from CheXpertDataset import CheXpertDataset
import matplotlib.pylab as plt

BATCH_SIZE = 128
OUTPUT_DIM = 10
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#This is just any generic model
#model = torchvison.some_model
model = models.resnet18(pretrained=False, num_classes=OUTPUT_DIM).to(device)
#Optional: define transformations for your specific dataset.
#Generally, it is best to use the original augmentations in the
#paper, replacing the Imagenet normalization with the normalization
#for your dataset.

transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                Solarization(p=0.0),
                transforms.ToTensor(),
                transforms.Normalize([0.49139968, 0.48215827 ,0.44653124], [0.24703233, 0.24348505, 0.26158768])
            ])

#For the transform argument for the dataset, pass in 
# Twins.transform_utils.Transform(transform_1, transform_2)
#If transforms are None, the Imagenet default is used.
#dataset = dsets.CIFAR10("datasets", train=True, transform=Transform(transform, transform), download=True)
dataset = CheXpertDataset("data/Frontal_Train.csv","resized", transform=Transform(transform, transform))

loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True)
#Make the BT instance, passing the model, the latent rep layer id,
# hidden units for the projection MLP, the tradeoff factor,
# and the loss scale.
learner = BarlowTwins(model, -2, [512, 1024, 1024, 1024],
                      3.9e-3, 1).to(device)

optimizer = torch.optim.Adam(learner.parameters(), lr=0.001)

#Single training epoch
loss_list = []

start = 0
start = time.time()

for batch_idx, ((x1,x2), _) in enumerate(loader):
  #if batch_idx == 0:
  #    plt.imshow(torch.movedim(data[0][0], 0, 2), origin='lower')
  #    plt.show()
  #    plt.imshow(torch.movedim(x2[0], 0, 2), origin='lower')
  #    plt.show()
  optimizer.zero_grad()
#  (x1, x2, _) = data
  #x1 = torch.reshape(x1,(BATCH_SIZE,3,224,224))
  x1, x2 = x1.to(device), x2.to(device)
  loss = learner(x1, x2)
  
  loss.backward()
  optimizer.step()
  loss_list.append(loss.item())
  if(batch_idx % 20 == 0):
    print("End Batch "+str(batch_idx))
    print("Elapsed time: "+str(time.time() - start))
    start = time.time()

plt.plot(loss_list)
plt.show()