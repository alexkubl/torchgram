# initalization and train cycle

from typing import Type, Any, Callable, Union, List, Optional

import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import torch.autograd as autograd

# from PIL import Image
from torchvision.io import read_image
import matplotlib.pyplot as plt

from glob import glob
import os
import math
import sys

import wandb
import argparse

from filters import * 

from models.dataset import * # GenBackDS, GenObjDS, DicsTrainDS
from models.model import * 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Tensor = torch.cuda.FloatTensor

def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)

    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
    
    
if __name__ == "__main__":

    wandb.init(project="wgan_gp", entity="alexkubl") #, resume=True
    
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    wandb.config = {
      "batch_size": 16
    }

    args = parser.parse_args()
    
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    

    
    gen_obj_dataloader = torch.utils.data.DataLoader(GenObjDS, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    gen_back_dataloader = torch.utils.data.DataLoader(GenBackDS, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    disc_dataloader = torch.utils.data.DataLoader(DicsTrainDS, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    
    # Loss functions
    adversarial_loss = torch.nn.BCELoss() # cross entropy loss
    auxiliary_loss = torch.nn.MSELoss() # mean square error

    
    gen_path = "/media/disk2/akublikova/GAN/testing/models/generator.pt"
    disc_path = "/media/disk2/akublikova/GAN/testing/models/discriminator.pt"
    
    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()
    
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    discriminator = discriminator.to(device)
    generator = generator.to(device)
#     g_checkpoint = torch.load(gen_path)
#     generator.load_state_dict(g_checkpoint['model_state_dict'])

#     d_checkpoint = torch.load(disc_path)
#     discriminator.load_state_dict(d_checkpoint['model_state_dict'])

    lambda_gp = 0.1
    lr=0.0002
    n_critic = 5
    
    # Setup Adam optimizers for both G and D
    
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)

#     if wandb.run.resumed:

# g_checkpoint = torch.load(gen_path)
# generator.load_state_dict(g_checkpoint['model_state_dict'])

# d_checkpoint = torch.load(disc_path)
# discriminator.load_state_dict(d_checkpoint['model_state_dict']

#         checkpoint = torch.load(wandb.restore(CHECKPOINT_PATH))
#         model.load_state_dict(checkpoint['model_state_dict'])
    
#     optimizer_G.load_state_dict(g_checkpoint['optimizer_state_dict'])
#     epoch = g_checkpoint['epoch']
#     g_loss = g_checkpoint['loss']
    
# optimizer_D.load_state_dict(d_checkpoint['optimizer_state_dict'])
# optimizer_G.load_state_dict(g_checkpoint['optimizer_state_dict'])
# epoch = d_checkpoint['epoch']
# d_loss = d_checkpoint['loss']
# g_loss = g_checkpoint['loss']

#     optimizer_D.load_state_dict(d_checkpoint['optimizer_state_dict'])
# #     epoch = checkpoint['epoch']
#     d_loss = d_checkpoint['loss']
        
    
#     discriminator.train()
#     generator.train()
    
    for epoch in range(n_epochs):
        i = 0
        for i, (back, pair) in enumerate(zip(gen_back_dataloader, gen_obj_dataloader)):
#             real_val = torch.ones(batch_size).unsqueeze(1).to(device)
#             fake_val = torch.zeros(batch_size).unsqueeze(1).to(device)

            optimizer_D.zero_grad()
            obj, mask = pair
            back = back.to(device)
            obj = obj.to(device)
            mask = mask.to(device)
            gen_labels = generator(torch.stack(concat_batch(back, obj, mask))) # filter generation
            
            gen_obj = torch.stack(apply_filter_batch(obj, gen_labels)) # apply generated filters
            fake_imgs = torch.stack(concat_batch(back, gen_obj, mask))
            
            real_imgs, labels = next(iter(disc_dataloader))
            real_imgs = Variable(real_imgs.type(Tensor))
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            real_imgs = torch.stack(apply_filter_batch(real_imgs, labels)) # apply filters to real batch
            
            fake_validity = discriminator(fake_imgs) # , fake_aux
            real_validity = discriminator(real_imgs) # , real_aux
            
#             d_fake_aux = auxiliary_loss(fake_aux, gen_labels)
#             d_real_aux = auxiliary_loss(real_aux, labels)
#             d_loss_aux = d_fake_aux + d_real_aux
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty 
#             d_loss = d_loss_adv + d_loss_aux

            d_loss.backward()
            optimizer_D.step()
            
            
            optimizer_G.zero_grad()
            if i % n_critic == 0:
                obj, mask = pair
                back = back.to(device)
                obj = obj.to(device)
                mask = mask.to(device)
                gen_labels = generator(torch.stack(concat_batch(back, obj, mask))) # filter generation
            
                gen_obj = torch.stack(apply_filter_batch(obj, gen_labels)) # applying 
                fake_imgs = torch.stack(concat_batch(back, gen_obj, mask))
                
                fake_validity = discriminator(fake_imgs) #, fake_aux
                g_loss = -torch.mean(fake_validity)
#                 g_loss_aux = auxiliary_loss(gen_labels, fake_aux)
#                 g_loss = g_loss_adv + g_loss_aux
                g_loss.backward()
                optimizer_G.step()
                      
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, n_epochs, i, len(gen_obj_dataloader), d_loss.item(), g_loss.item())
                )

                i += 1
                wandb.log({"d_loss": d_loss, 
                    "g_loss": g_loss,            
                })
        
        # after one epoch saving the models with loss and checkpoints    
        torch.save({
            'epoch': epoch,
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': optimizer_G.state_dict()
        }, gen_path)

        torch.save({
            'epoch': epoch,
            'model_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': optimizer_D.state_dict()
        }, disc_path)