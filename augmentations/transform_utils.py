from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms
import random
import torch
import numpy as np
'''
Adapted from https://github.com/facebookresearch/barlowtwins
'''

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

'''
Adapted from https://github.com/facebookresearch/barlowtwins
'''
class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

'''
Function adapted from: https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
Using noise samplig from paper: https://arxiv.org/pdf/2101.04909.pdf
'''
class GaussianNoise(object):
    def __init__(self, p, mean = 0):
        self.p = p
        self.mean = mean

    def __call__(self, tensor):
        if random.random() < self.p:
            SNR = np.random.uniform(0.06, 0.1)
            mu_tensor = torch.mean(tensor)
            sig_noise = mu_tensor*SNR
            return tensor + torch.randn(tensor.size()) * sig_noise + self.mean
        else:
            return tensor

'''
Used in paper https://arxiv.org/pdf/2101.05224.pdf
'''
class BrightnessModulation(object):
    def __init__(self, p, brightness = 0.2):
        self.p = p
        self.brightness = brightness

    def __call__(self, tensor):
        if random.random() < self.p:
            BR = np.random.uniform(-self.brightness, self.brightness)
            tensor = tensor + BR
            return tensor.clamp_(min=0., max=1.) #The paper describes a clamping not a reescaling
        else:
            return tensor

'''
Used in paper https://arxiv.org/pdf/2101.05224.pdf. 
I think there is a typo in the paper, it describes an adjust from -0.2 to 0.2, but it does not make
sense to have a negative std
'''
class ContrastModulation(object):
    def __init__(self, p, contrast = 2):
        self.p = p
        self.contrast = contrast

    def __call__(self, tensor):
        if random.random() < self.p:
            CT = np.random.uniform(0 , self.contrast)
            mean_val = torch.mean(tensor)
            norm_tmp = tensor - mean_val
            std_mul = CT*torch.std(tensor)
            tensor = mean_val + norm_tmp * std_mul
            return tensor.clamp_(min=0., max=1.) 
        else:
            return tensor

'''
Adapted from https://github.com/facebookresearch/barlowtwins
'''
class Transform:
    def __init__(self, transform=None, transform_prime=None):
        '''

        :param transform: Transforms to be applied to first input
        :param transform_prime: transforms to be applied to second
        '''
        if transform == None:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
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
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        if transform_prime == None:

            self.transform_prime = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform_prime = transform_prime

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2