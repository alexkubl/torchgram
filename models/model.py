import torch
from torch import nn
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models.resnet import ResNet, BasicBlock

num_classes = 1

seed = 1
torch.cuda.manual_seed(seed)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
#     elif classname.find('Linear'):
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)
    

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet101(pretrained=False)
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
        resnet = models.resnet101(pretrained=False)
        self.features = nn.ModuleList(resnet.children())[:-1]
        self.features = nn.Sequential(*self.features)
        # Output layers
        in_features = resnet.fc.in_features
        self.adv_layer = nn.Linear(in_features, 1)
#         self.aux_layer = nn.Sequential(nn.Linear(in_features, num_classes), nn.Sigmoid())

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
#         label = self.aux_layer(out)

        return validity #label # 2 tensors: (size = 1 * batch_size, num_filters * batch_size)