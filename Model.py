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
        self.pretrained.classifier = NNClassifier(n_in, config)

    def forward(self, x):
        return self.pretrained(x)

    
class NNClassifier(nn.Module):
    def __init__(self, n_in, config):
        super().__init__()

        self.hidden_layers = nn.ModuleList([nn.Linear(n_in, self.hidden)])
        layer_sizes = zip(config['MODEL']['HIDDEN'][:-1], config['MODEL']['HIDDEN'][1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.dropout = nn.Dropout(p=config['MODEL']['DROPOUT'])
        self.output = nn.Linear(hidden_layers[-1], config['MODEL']['OUTPUTS'])

    def forward(self, x):
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        x = self.output(x)

        return F.log_softmax(x, dim=1)
        
        

