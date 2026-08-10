import torch.nn as nn
import torchvision.models as torch_classifiers

class DenseNetCamelyon17(nn.Module):
    def __init__(self, num_classes=2):
        super(DenseNetCamelyon17, self).__init__()
        # Load the pre-trained DenseNet-121 model
        self.base_model = torch_classifiers.densenet121()
        
        # Get the number of input features for the classifier
        num_features = self.base_model.classifier.in_features
        
        # Replace the classifier with a custom fully connected layer
        self.base_model.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # Forward pass through the base model
        return self.base_model(x)
