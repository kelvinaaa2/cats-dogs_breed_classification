from Model import BreedClassifier

import os
import torch
import torch.nn as nn 

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Solver:
    def __init__(self, config, data_loader, model):
        self.data_loader = data_loader
        self.model = model
        self.criterion = nn.NLLloss()
        # self.img_size = 
        
        self.epoch = config['TRAINING']['EPOCH']
        self.lr = config['TRAINING']['LR']
        
        # Freeze pretrained model param
        for param in self.model.features.parameters():
            param.require_grad = False
        
        self.optimizer = nn.optim.Adam(self.model.parameters()
                                       , lr=config['TRAINING']['LR'])
        # lr_scheduler = None
        self.model.to(device)
       
    def train(self):
        
        data_loader = self.data_loader
        
        for e in range(self.epoch):
            self.model.train()
            for image, labels in data_loader:
                
        
        
        
        
        
        
        
        