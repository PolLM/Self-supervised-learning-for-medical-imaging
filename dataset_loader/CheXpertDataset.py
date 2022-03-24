import torch
import numpy as np
import os
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd

class CheXpertDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.xraylist = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.xraylist)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.xraylist.iloc[idx, 1])
        image = Image.open(img_name)
        image = image.convert(mode="RGB")
        
        # label is 1 if desease
        # label is 0 if NO desease
        label = self.xraylist.iloc[idx, 6] != 1.0


        if self.transform:
            image = self.transform(image)

        return image,label