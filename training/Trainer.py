import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import os
import yaml
import torchvision.models as torch_classifiers
from networks import model_loading, optimizer_loading
from training.utils import loss_fn_loading, dice_score, acc_score

class Trainer:

    def __init__(self, train_ldr, val_ldr, test_ldr, dataset):
        
        self.model     = model_loading(dataset)
        self.optimizer = optimizer_loading(dataset, self.model)

        self.train_ldr = train_ldr
        self.val_ldr   = val_ldr
        self.test_ldr  = test_ldr
        self.loss_fn   = loss_fn_loading(dataset)
        if dataset in ['FeTS', 'prostate', 'polypgen', 'ultrasound', 'xray', 'mms']:
            self.score_fn = dice_score
        else:
            self.score_fn = acc_score
        self.device    = 'cuda:0'
        self.amp       = dataset == 'FeTS'
        if self.amp:
            print('IS AMP')
        self.scaler    = torch.cuda.amp.GradScaler()

    def train_epoch(self):

        self.model.to(self.device)
        self.model.train()


        for img, label in tqdm(self.train_ldr):
            img, label = img.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            if self.amp:
                with torch.cuda.amp.autocast():
                    pred = self.model(img)
                    loss = self.loss_fn(pred, label)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(img)
                loss = self.loss_fn(pred, label)
                loss.backward()
                self.optimizer.step()
        
        self.save_model('local_model.pt')
        print('trained local epoch')

    def eval(self):
        self.model.eval()
        self.model.to(self.device)
        total_score = 0
        n_batches   = 0

        with torch.no_grad():
            for img, label in self.val_ldr:
                img, label   = img.to(self.device), label.to(self.device)
                outputs      = self.model(img)
                score        = self.score_fn(outputs, label)
                total_score += score
                n_batches   += label.size(0)

        final_score = total_score / n_batches
        return final_score
    
    def test(self):
        self.model.eval()
        self.model.to(self.device)
        total_score = 0
        n_batches   = 0

        with torch.no_grad():
            for img, label in self.test_ldr:
                img, label   = img.to(self.device), label.to(self.device)
                outputs      = self.model(img)
                score        = self.score_fn(outputs, label)
                total_score += score
                n_batches   += label.size(0)

        final_score = total_score / n_batches
        return final_score
  
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, state_dict):
        self.model.load_state_dict(state_dict)
