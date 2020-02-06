import os
import torch
import scipy.misc

import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image
import numpy

import argparse
import parser

import torchvision.transforms.functional as TF

import pandas as pd

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

class DATA(Dataset):
    def __init__(self, args):

        self.img_dir = os.path.join(args.data_dir, "train/")
        self.labels_dir = args.data_dir

        # Filenames of all files in img directory, as a list
        img_names = sorted(os.listdir(self.img_dir))

        # Read CSV of attribute labels. 'labels' is a list of 1/0 for the 'smiling' attribute for each image. 1 means that image is smiling, 0 means not smiling. Element 0 corresponds to 00000.png
        df = pd.read_csv( os.path.join(self.labels_dir, "train.csv"))
        labels = df["Smiling"].tolist()

        # self.data is a list with length equal to the number of images
        # Each element is another list, with first element being the filepath to an image, and second element being the label
        self.data = []
        for i in range(len(img_names)):
            self.data.append( [os.path.join(self.img_dir, img_names[i]), labels[i]] ) 
            
        # Transform the image
        self.transform = transforms.Compose([
            transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
            transforms.Normalize(MEAN, STD)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        ''' get data '''
        img_path, label = self.data[idx]
        
        ''' read image '''
        img = Image.open(img_path).convert('RGB')

        return self.transform(img), label