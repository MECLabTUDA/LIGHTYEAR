import torch
from torch import nn
from tqdm import tqdm
import os
import yaml
from networks import model_loading
from training.Trainer import Trainer

class DittoTrainer(Trainer):

    def __init__(self, train_ldr, val_ldr, test_ldr, dataset):
        
        super().__init__(train_ldr, val_ldr, test_ldr, dataset)
        self.pers_model   = model_loading(dataset)
        self.global_model = model_loading(dataset)
        self.lambda_      = 0.5
        self.optimizer.add_param_group({"params": self.pers_model.parameters()})
        self.loss_fn   = nn.CrossEntropyLoss() 

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
            

        self.pers_model.to(self.device)
        self.pers_model.train()
        self.global_model.to(self.device)
        self.global_model.train()

        for img, label in self.train_ldr:
            img, label = img.to(self.device), label.to(self.device)
            pred       = self.pers_model(img)
            loss       = self.loss_fn(pred, label)
            
            loss.backward()
            
            for pers_param, global_param in zip(
                    self.pers_model.parameters(), self.global_model.parameters()
                ):
                    if pers_param.requires_grad:
                        pers_param.grad.data += self.lambda_ * (
                            pers_param.data - global_param.data
                        )

            self.optimizer.step()
            self.optimizer.zero_grad()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.global_model.load_state_dict(state_dict)

