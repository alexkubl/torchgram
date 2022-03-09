import torch
from torch import nn
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models.resnet import ResNet, BasicBlock

num_classes = 28

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet101(pretrained=True)
        self.features = nn.ModuleList(resnet.children())[:-1]
        self.features = nn.Sequential(*self.features)
        in_features = resnet.fc.in_features
        self.fc = nn.Sequential(nn.Linear(in_features, num_classes), nn.Sigmoid())
    
    def forward(self, x):
        # change forward here
        out = self.features(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out # tensor (size = num_filters * batch_size) 
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet101(pretrained=True)
        self.features = nn.ModuleList(resnet.children())[:-1]
        self.features = nn.Sequential(*self.features)
        # Output layers
        in_features = resnet.fc.in_features
        self.adv_layer = nn.Sequential(nn.Linear(in_features, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(in_features, num_classes), nn.Sigmoid())

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label # 2 tensors: (size = 1 * batch_size, num_filters * batch_size)