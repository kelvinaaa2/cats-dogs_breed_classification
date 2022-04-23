import torch
import torch.nn as nn
from torchvision import models


class BreedClassifier(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden = config['MODEL']['HIDDEN']

        # Call Pretrained Network
        self.pretrained = getattr(models, config['MODEL']['PRETRAINED'])(pretrained=True)
        n_in = next(self.pretrained.fc.modules()).in_features  # Raise except pretrained.classifier for densenet

        # Make Classifier
        self.hidden_layers = nn.ModuleList([nn.Linear(n_in, self.hidden)])

    def forward(self):
        pass


model = getattr(models, 'resnet50')(pretrained=True)
print(1)

