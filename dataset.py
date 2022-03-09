from typing import Type, Any, Callable, Union, List, Optional

import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

from torch.utils.data import Dataset
from torchvision import datasets

import numpy as np


from PIL import Image
from torchvision.io import read_image
import matplotlib.pyplot as plt

from glob import glob
import os
import math
import sys

input_size = 256
num_classes = 28

from filters import *

filters_list = open("/media/disk2/akublikova/GAN/filters.txt", "r").read().splitlines()


def gen_filter():
#     return torch.ones(28)
    f = torch.randn(num_classes - 4) - 1
    f = torch.cat((torch.rand(4), f), 0)
    f = torch.clamp(f, 0, 1)  #torch.zeros(24) #

    return f #torch.ones(1)/2


def apply_filters(img, f): # img, f - tensors
    new_img = img
#     print(img, f)
    for i in zip(filters_list, f):
        new_img = globals() [i[0]] (new_img, i[1].item())
    return new_img


def concat(img1, img2, mask): # concatenate background and foreground 
    return util.compose(img1, img2, mask)

def concat_batch(back, obj, mask):
    return [torch.clamp(concat(b, ob, m), 0, 1) for b, ob, m in zip(back, obj, mask)]
              
def apply_filters_batch(img, labels):
    return [apply_filters(image, label) for image, label in zip(img, labels)]


class DicsImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        super().__init__()
        self.img_dir = (f'{img_dir}/disc_train')
        image_names = []
        labels = []
        images = glob(f'{self.img_dir}/*')
        for img in images:
            name = os.path.basename(img)
            img_name = name #os.path.basename(name)
            image_names.append(img_name)
            label = gen_filter() #torch.clamp(torch.randn(num_classes), 0, 1)
            labels.append(label)
                
        self.data = image_names
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label, img = self.labels[idx], self.data[idx]
        
        img_path = os.path.join(self.img_dir, img) 
        image = Image.open(img_path)
        
        image = image.convert('RGB')
        if self.transform:
            image = self.transform['img'](image)

        return image, label
    
class GenBackImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        super().__init__()
        self.img_dir = img_dir
        back_names = []
        back = glob(f'{img_dir}/back/*')

        for img in back:
            name = os.path.basename(img) #             print(name)
            img_name = name #os.path.basename(name)
            back_names.append(img_name)
        
        self.data_back = back_names
        self.transform = transform

    def __len__(self):
        return len(self.data_back)
    
    def __getitem__(self, idx):
        img = self.data_back[idx]
        back_path = os.path.join(self.img_dir, f'back/{img}')
        
        back = Image.open(back_path)
        if self.transform:
            back = self.transform['img'](back)
        
        return back


class GenObjImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        super().__init__()
        self.img_dir = img_dir
        obj_names = []
        obj = glob(f'{img_dir}/objects/*')
        
        for img in obj:
            name = os.path.basename(img)
            img_name = name #os.path.basename(name)
            obj_names.append(img_name)
            
        self.data_obj = obj_names
        self.transform = transform
#         self.target_transform = target_transform

    def __len__(self):
        return len(self.data_obj)
    
    def __getitem__(self, idx):
        img = self.data_obj[idx]
        name = os.path.basename(img).split('.')[0]
        obj_path = os.path.join(self.img_dir, f'objects/{img}')
        mask_path = os.path.join(self.img_dir, f'masks/{name}.png')
        
        obj = Image.open(obj_path)
        obj = obj.convert('RGB')
        mask = Image.open(mask_path)
        mask = mask.convert('RGB')
#         obj = np.asarray(obj)/255
        
        mask = np.asarray(mask)
        
        if self.transform:
            obj = self.transform['object'](obj)
            mask = self.transform['mask'](mask)
        
#         print('obj:', obj)
        return obj, mask
#         return concat(back, obj, mask) # , label    
    
data_dir = "/media/disk2/akublikova/GAN/dataset" # masks + objects, back

data_transforms = {
    'object': transforms.Compose([ # masks + objects
        transforms.ToTensor(),
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    
    'mask': transforms.Compose([ # masks + objects
        transforms.ToTensor(),
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ConvertImageDtype(torch.float),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'img': transforms.Compose([
        transforms.ToTensor(),
#         transforms.CenterCrop(input_size),
        transforms.RandomResizedCrop(input_size),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

GenBackDS = GenBackImageDataset(data_dir, data_transforms)
GenObjDS = GenObjImageDataset(data_dir, data_transforms) 
DicsTrainDS = DicsImageDataset(data_dir, data_transforms)

# data_dir = "/media/disk2/akublikova/GAN/dataset"

# disc_data_transforms = transforms.Compose([  
#     transforms.ToTensor(),
#     transforms.ConvertImageDtype(torch.float),
#     transforms.RandomResizedCrop(input_size),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])    
# ])



