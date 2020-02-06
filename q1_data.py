import os
import torch
import scipy.misc

import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from PIL import Image
import numpy
import argparse
import parser


MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

class DATA(Dataset):
    def __init__(self, args):

        ''' set up basic parameters for dataset '''
        self.img_dir = args.data_dir    

        # Filenames of all files in img and seg directories, as a list
        img_names = sorted(os.listdir(self.img_dir))

        # self.data is a list with length equal to the number of images
        # Each element is the filepath to an image
        self.data = []
        for i in range(len(img_names)):
            self.data.append( os.path.join(self.img_dir, img_names[i]) ) 
            
        # Transform the image
        self.transform = transforms.Compose([
            transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
            transforms.Normalize(MEAN, STD)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        ''' get data '''
        img_path = self.data[idx]
        
        ''' read image '''
        img = Image.open(img_path).convert('RGB')

        return self.transform(img)