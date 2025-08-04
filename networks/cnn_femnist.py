import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNFEMNIST(nn.Module):
    def __init__(self, num_classes=62):
            super(CNNFEMNIST, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
            self.pool1 = nn.MaxPool2d(2, 2)  # 14x14

            self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
            self.pool2 = nn.MaxPool2d(2, 2)  # 7x7

            self.fc1 = nn.Linear(64 * 7 * 7, 512)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))  # -> (32, 14, 14)
        x = self.pool2(F.relu(self.conv2(x)))  # -> (64, 7, 7)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
