import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import os
import yaml
import torchvision.models as torch_classifiers
from networks import model_loading, optimizer_loading
from training.Trainer import Trainer
from training.utils import dice_score

class SegTrainer(Trainer):

    def __init__(self, train_ldr, val_ldr, test_ldr, dataset):
        super().__init__(train_ldr, val_ldr, test_ldr, dataset)
        
    def eval(self):
        self.model.eval()
        self.model.to(self.device)
        dice_scores = []

        with torch.no_grad():
            for img, mask in self.val_ldr:
                img, mask = img.to(self.device), mask.to(self.device)
                outputs  = self.model(img)
                score = dice_score(outputs, mask)
                dice_scores.append(score)

        mean_dice = sum(dice_scores) / len(dice_scores)
        return mean_dice
    
    def test(self):
        self.model.eval()
        self.model.to(self.device)
        dice_scores = []

        with torch.no_grad():
            for img, mask in self.test_ldr:
                img, mask = img.to(self.device), mask.to(self.device)
                outputs  = self.model(img)
                score = dice_score(outputs, mask)
                dice_scores.append(score)

        mean_dice = sum(dice_scores) / len(dice_scores)
        return mean_dice
    

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, state_dict):
        self.model.load_state_dict(state_dict)
