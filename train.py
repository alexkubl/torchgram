# initalization and train cycle

from typing import Type, Any, Callable, Union, List, Optional

import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

# from PIL import Image
from torchvision.io import read_image
import matplotlib.pyplot as plt

from glob import glob
import os
import math
import sys

import wandb
import argparse

# import torch.optim as optim
# from torchvision.utils import save_image
# from datetime import datetime
# from libs.compute import *
# from libs.constant import *
# from libs.model import *

from filters import * 

from models.dataset import * # GenBackDS, GenObjDS, DicsTrainDS
from models.model import * 


if __name__ == "__main__":

    wandb.init(project="gan", entity="alexkubl")
    
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    wandb.config = {
#       "learning_rate": 0.001,
#       "epochs": 100,
      "batch_size": 16
#       "input_size": 256
    }
#     parser.add_argument('--coeff', type=int)
    args = parser.parse_args()
    
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    
#     coeff = args.coeff
    
#     dataloaders = { }
    
    gen_obj_dataloader = torch.utils.data.DataLoader(GenObjDS, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    gen_back_dataloader = torch.utils.data.DataLoader(GenBackDS, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    disc_dataloader = torch.utils.data.DataLoader(DicsTrainDS, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Loss functions
    adversarial_loss = torch.nn.BCELoss() # cross entropy loss
    auxiliary_loss = torch.nn.MSELoss() # mean square error

    
    gen_path = "/media/disk2/akublikova/GAN/models/generator.pt"
    disc_path = "/media/disk2/akublikova/GAN/models/discriminator.pt"
    
    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    g_checkpoint = torch.load(gen_path)
    generator.load_state_dict(g_checkpoint['model_state_dict'])

    d_checkpoint = torch.load(disc_path)
    discriminator.load_state_dict(d_checkpoint['model_state_dict'])
    
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # lambda_gp = 10
    lr=0.0003

    # Setup Adam optimizers for both G and D
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    
    
    optimizer_G.load_state_dict(g_checkpoint['optimizer_state_dict'])
    epoch = g_checkpoint['epoch']
    g_loss = g_checkpoint['loss']
    
    
    optimizer_D.load_state_dict(d_checkpoint['optimizer_state_dict'])
#     epoch = checkpoint['epoch']
    d_loss = d_checkpoint['loss']
        

    G_loss = []
    D_loss = []
    
    real_val = torch.ones(batch_size).unsqueeze(1).to(device)
    fake_val = torch.zeros(batch_size).unsqueeze(1).to(device)

    for epoch in range(n_epochs):
        i = 0
        for back, pair in zip(gen_back_dataloader, gen_obj_dataloader):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
    #         optimizer_G.zero_grad()

            obj, mask = pair
            # loss for real images, all  real batch
            real_img, labels = next(iter(disc_dataloader))
            real_img = real_img.to(device)
            labels = labels.to(device)
            real_img = torch.stack(apply_filters_batch(real_img, labels)) # apply filters to batch

            real_pred, real_aux = discriminator(real_img)
            # loss(fake, real) / (input, target)
            d_real_aux = auxiliary_loss(real_aux, labels)
            d_real_loss = 0.5 * (adversarial_loss(real_pred, real_val) + d_real_aux)

            # adv_loss, aux_loss: tensors (size = 1 * batch_size, num_filters * batch_size)

            #####


            # loss for generated images 
            back = back.to(device)
            obj = obj.to(device)
            mask = mask.to(device)

            fake_img = torch.stack(concat_batch(back, obj, mask)) # for batch
            gen_labels = generator(fake_img) # labels from generator

            gen_obj = torch.stack(apply_filters_batch(obj, gen_labels))
            gen_images = torch.stack(concat_batch(back, gen_obj, mask)) # batch for discr

            fake_pred, fake_aux = discriminator(gen_images)
            d_fake_aux = auxiliary_loss(fake_aux, gen_labels)
            d_fake_loss = 0.5 * (adversarial_loss(fake_pred, fake_val) + d_fake_aux)

            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward() # retain_graph=True
            optimizer_D.step()
            D_loss.append(d_loss)

            # -----------------
            #  Train Generator
            # -----------------
    #         gen_labels = generator(fake_img)

    #         optimizer_D.zero_grad()
            optimizer_G.zero_grad()
            gen_labels = generator(fake_img)

            gen_obj = torch.stack(apply_filters_batch(obj, gen_labels))
            gen_images = torch.stack(concat_batch(back, gen_obj, mask))

    #         print(back)
            fake_pred, fake_aux = discriminator(gen_images) # using current batch of images for gen training
            g_loss = 0.5 * (adversarial_loss(fake_pred, fake_val) + auxiliary_loss(fake_aux, gen_labels))
    #         print(g_loss)
            g_loss.backward()
            optimizer_G.step()
            G_loss.append(g_loss)
            
            wandb.log({"d_loss": d_loss, 
                      "g_loss": g_loss, 
                      "d_real_aux": d_real_aux,
                      "d_fake_aux": d_fake_aux,
                  
                      })
                      
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, i, len(gen_obj_dataloader), d_loss.item(), g_loss.item())
            )
            i += 1

        # after one epoch saving the models with loss and checkpoints    
        torch.save({
            'epoch': epoch,
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': optimizer_G.state_dict(),
            'loss': g_loss }, gen_path)

        torch.save({
            'epoch': epoch,
            'model_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': optimizer_D.state_dict(),
            'loss': d_loss }, disc_path)