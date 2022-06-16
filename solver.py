from Model import BreedClassifier
from sklearn.metrics import confusion_matrix
import os
import torch
from torch import optim
import torch.nn as nn 

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Solver:
    def __init__(self, config, train_dataloader, test_dataloader, model):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.criterion = nn.NLLLoss()
        # self.img_size = 
        
        self.epoch = config['TRAINING']['EPOCH']
        self.lr = config['TRAINING']['LR']
        
        # Freeze pretrained model param
        try:
            for param in self.model.features.parameters():
                param.require_grad = False
        except:
            for param in self.model.pretrained.parameters():
                param.require_grad = False
        
        self.optimizer = optim.Adam(self.model.parameters()
                                       , lr=config['TRAINING']['LR'])
        # lr_scheduler = None
        self.model.to(device)
       
    def train(self):
        steps = 0 
        running_loss = 0
        model_loss = 0
        model_acc = 0
        
        data_loader = self.train_dataloader
        
        for e in range(self.epoch):
            # Train Mode on
            self.model.train()
            for images, labels in data_loader:
                steps += 1
                images, labels = images.to(device), labels.to(device)
                # Call optimizer
                self.optimizer.zero_grad()
                # Train the model
                output = self.model(images)
                # Calculating loss 
                loss = self.criterion(output, labels)
                loss.backward()    
                running_loss += loss.item()
                # Update optimizer
                self.optimizer.step()

            # Eval mode on
            self.model.eval()
            with torch.no_grad():
                test_loss, accuracy, confusion = self.validation()
            # Print the validation details
            print("Epoch: {}/{} - ".format(e + 1, self.epoch),
                  "Training Loss: {:.3f} - ".format(running_loss / steps + 1),
                  "Validation Loss: {:.3f} - ".format(test_loss / len(self.test_dataloader)),
                  "\nValidation Accuracy: {:.3f} -".format(accuracy / len(self.test_dataloader)),
                  "lr: {}".format(self.optimizer.param_groups[0]['lr']))
            print(f'\n{confusion}')
            # Save model based on validation performance
            if model_loss == 0:
                model_loss += test_loss
                model_acc += accuracy
                torch.save(self.model.state_dict(), f'model_{str(e)}.pt')
            else:
                if model_loss >= test_loss:
                    model_loss = test_loss
                    model_acc = accuracy
                    torch.save(self.model.state_dict(), f'model_{str(e)}.pt')

            running_loss = 0
            # Set model back to training mode
            self.model.train()

        return self.model

    def validation(self):
        test_loss = 0
        accuracy = 0

        pred = []
        label = []
        
        data_loader = self.test_dataloader
        
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            # Validate the model
            output = self.model(images)
            # Calculate test loss
            loss = self.criterion(output, labels)
            test_loss += loss.item()
            # remove log from nll_loss
            ps = torch.exp(output)
            # Find the highest probability output
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

            pred.append(ps.max(dim=1)[1])
            label.append(labels.data)

        pred = torch.cat(pred).tolist()
        label = torch.cat(label).tolist()

        confusion = confusion_matrix(label, pred)

        return test_loss, accuracy, confusion


if __name__ == '__main__':
    from data_pipeline import DataPipeline
    import argparse
    import yaml
    
    solver = Solver()
    res = solver.train

            
        
            
        
                
            
            
                
                
                
                
                
                

                
        
        
        
        
        
        
        
        