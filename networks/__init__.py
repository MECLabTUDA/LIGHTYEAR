import torch
import random
import numpy as np

from networks.densenet_camelyon17 import DenseNetCamelyon17
from networks.cnn_femnist import CNNFEMNIST
from networks.vit_seg_modeling2D import VisionTransformer as ViT_seg 
from networks.efficientnet import EffNet



def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def model_loading(dataset):
    seed_everything(42)
    match dataset:
        case 'camelyon17':
            return DenseNetCamelyon17()
        case 'femnist':
            return CNNFEMNIST()
        case 'isic19':
            return EffNet()
        case 'ultrasound':
            return ViT_seg()
        case 'xray':
            return ViT_seg()

def optimizer_loading(dataset, model):
    match dataset:
        case 'camelyon17':
            return torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=5e-4, momentum=0.9) 
        case 'femnist':
            return torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        case 'isic19':
            return torch.optim.Adam(model.parameters(), 5e-4)
        case 'ultrasound':
            return torch.optim.Adam(model.parameters(), 0.001)
        case 'xray':
            return torch.optim.Adam(model.parameters(), 0.001)