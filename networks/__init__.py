import torch

from networks.densenet_camelyon17 import DenseNetCamelyon17
from networks.cnn_femnist import CNNFEMNIST

def model_loading(dataset):
    match dataset:
        case 'camelyon17':
            return DenseNetCamelyon17()
        case 'femnist':
            return CNNFEMNIST()

def optimizer_loading(dataset, model):
    match dataset:
        case 'camelyon17':
            return torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=5e-4, momentum=0.9) 
        case 'femnist':
            return torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
