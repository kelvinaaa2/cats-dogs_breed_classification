import os
import glob
import yaml

import torch
import torch.nn
from torchvision import datasets, models, transforms as T

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DataPipeline:
    def __init__(self, config):
        self.train_dir = config['Data_Pipeline']['TRAIN_DIR']
        self.valid_dir = config['Data_Pipeline']['VALID_DIR']
        self.mode = config['Data_Pipeline']['MODE']
        self.mean = config['Data_Pipeline']['TRANS_MEAN']
        self.std = config['Data_Pipeline']['TRANS_STD']
        self.batch_size = config['Data_Pipeline']['BATCH_SIZE']

    def transformation(self):
        normalize = T.Normalize(mean=self.mean, std=self.std)
        
        train_transform = T.Compose([T.RandomRotation(30),
                               T.Resize(255),
                               T.CenterCrop(224),
                               T.RandomHorizontalFlip(),
                               T.ToTensor(),
                               normalize])
        
        valid_transform = T.Compose([T.Resize(255),
                               T.CenterCrop(224),
                               T.ToTensor(),
                               normalize])
        
        return normalize, train_transform, valid_transform

    def get_data_loader(self):
        
        normalize, train_transform, valid_transform = self.transformation()

        train_data = datasets.ImageFolder(self.train_dir, transform=train_transform)
        valid_data = datasets.ImageFolder(self.valid_dir, transform=valid_transform)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size)
        valid_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size)

        return train_loader, valid_loader, train_data, valid_data


if __name__ == '__main__':
    pass 







