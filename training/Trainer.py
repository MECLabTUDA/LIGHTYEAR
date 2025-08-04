import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import os
import yaml
import torchvision.models as torch_classifiers
from networks import model_loading, optimizer_loading


class Trainer:

    def __init__(self, train_ldr, val_ldr, test_ldr, dataset):
        
        self.model     = model_loading(dataset)
        self.optimizer = optimizer_loading(dataset, self.model)

        self.train_ldr = train_ldr
        self.val_ldr   = val_ldr
        self.test_ldr  = test_ldr
        self.loss_fn   = nn.CrossEntropyLoss() 
        self.device    = 'cuda:0'

    def train_epoch(self):

        self.model.to(self.device)
        self.model.train()

        for img, label in self.train_ldr:
            img, label = img.to(self.device), label.to(self.device)
            pred       = self.model(img)
            loss       = self.loss_fn(pred, label)
            
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.save_model('local_model.pt')
        print('trained local epoch')

    def eval(self):
        self.model.eval()
        self.model.to(self.device)
        correct = 0
        total   = 0

        with torch.no_grad():
            for img, label in self.val_ldr:
                img, label = img.to(self.device), label.to(self.device)
                outputs  = self.model(img)
                _, preds = torch.max(outputs, dim=1)
                correct += preds.eq(label).sum().item()
                total   += label.size(0)

        acc = correct / total
        return acc
    
    def test(self):
        self.model.eval()
        self.model.to(self.device)
        correct = 0
        total   = 0

        with torch.no_grad():
            for img, label in self.test_ldr:
                img, label = img.to(self.device), label.to(self.device)
                outputs  = self.model(img)
                _, preds = torch.max(outputs, dim=1)
                correct += preds.eq(label).sum().item()
                total   += label.size(0)

        acc = correct / total
        return acc

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, state_dict):
        self.model.load_state_dict(state_dict)
