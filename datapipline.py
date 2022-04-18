import os
import glob
import yaml

import torch
import torch.nn
from torchvision import datasets, transforms as T, models

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Data_Pipeline():
    def __init__(self, config):
        self.mode = config['Data_Pipeline']['MODE']
        self.mean = config['Data_Pipeline']['TRANS_MEAN']
        self.std = config['Data_Pipeline']['TRANS_STD']

    def transformation(self):
        normalize = T.Normalize(mean=self.mean, std=self.std)
        if self.mode == 'TRAIN':
            transform = T.Compose([T.RandomRotation(30),
                                   T.Resize(255),
                                   T.CenterCrop(224),
                                   T.RandomHorizontalFlip(),
                                   T.ToTensor(),
                                   normalize])
        else:
            transform = T.Compose([T.Resize(255),
                                   T.CenterCrop(224),
                                   T.ToTensor(),
                                   normalize])
        return transform

    def get_data_loader(self):
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_transform = T.Compose([T.RandomRotation(30),
                                              T.Resize(224),
                                              T.CenterCrop(224),
                                              T.RandomHorizontalFlip(),
                                              normalize])



