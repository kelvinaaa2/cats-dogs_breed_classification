import torch
import torch.nn as nn
from torchvision import models


class BreedClassifier(nn.Module):

    def __init__(self, config):
        super().__init__()

        # Call Pretrained Network
        self.pretrained = getattr(models, config['MODEL']['PRETRAINED'])(pretrained=True)
        try:
            n_in = next(self.pretrained.fc.modules()).in_features  # pretrained.fc for resnet
        except:
            n_in = next(self.pretrained.classifier.modules()).in_features  # pretrained.classifier for densenet

        # Make Classifier
        self.hidden_layers = nn.ModuleList([nn.Linear(n_in, self.hidden)])
        layer_sizes = zip(config['MODEL']['HIDDEN'][:-1], config['MODEL']['HIDDEN'][1:])

    def forward(self):
        pass

