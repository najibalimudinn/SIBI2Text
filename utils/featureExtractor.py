import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Load pretrained ResNet-18 model
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Replace the fully connected layer with identity
        self.model.fc = nn.Identity()
        self.model.eval()  # Set model to evaluation mode

    def forward(self, x):
        with torch.no_grad():  # Disable gradient computation
            return self.model(x)
