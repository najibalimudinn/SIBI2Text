import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

feature_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
feature_extractor.fc = nn.Identity()
feature_extractor.eval()