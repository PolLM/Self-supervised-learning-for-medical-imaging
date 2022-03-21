#We are going to scan the best transforms to apply to the medical images

from logging import exception
import os
import numpy as np
import time

import sys 
PROJECT_PATH =  os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.insert(0,PROJECT_PATH)

from torchvision import transforms
from self_sup_classes.barlow import *

from tqdm import tqdm
import matplotlib.pylab as plt

#We are going to take into consideration the following transformations,extracted from the papers:
#https://arxiv.org/pdf/2101.04909.pdf
#https://arxiv.org/pdf/2006.13276.pdf
#https://arxiv.org/pdf/2101.05224.pdf

##### (1) random resizing/cropping
# The cropping from the random resizing/cropping augmentation was done at an image scale uniformly 
# distributed between 20% and 100% the size of the original image.
# Google paper: random crop to 224×224 pixels

##### (2) random horizontal flipping, 
#

##### (3) random vertical flipping, 
#

##### (4) random Gaussian blur, 
#For the blur augmentation, we applied the following normalized Gaussian kernel: g(x, y) = 1 σkernel 
# √ 2π exp  − 1 2 x 2 + y 2 σ 2 kernel  , (3) where σ was selected for each sample uniformly at random 
# between 0.1 and 2.0 pixels.

##### (5) Gaussian noise addition, 
#We selected the standard deviation for the noise addition randomly according to the following formula: 
# σnoise = µimage SNR , (4) where SNR was selected uniformly between 4 and 8 for each sample and µimage 
# was the average pixel value of the input sample image

##### (6) histogram normalization.
#

##### (7) random rotation 
#by angle δ ∼ U(−20, 20) degree

##### (8) random additive brightness modulation
# Random additive brightness modulation adds a δ ∼ U(−0.2, 0.2) to all channels

##### (9) random multiplicative contrast modulation
# Random multiplicative contrast modulation multiplies per-channel standard deviation by a factor s ∼ U(−0.2, 0.2)

##### (10) change of perspective

###################################################
# We are going to train the model in a self-supervised approach by applying:
# all transformations, none of them, and all combinations of all transformations but one.
#Probability 

###################################################

config = {
    "batch_size": ,
    "model": ,

}