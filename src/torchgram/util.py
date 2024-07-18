import math
from functools import reduce
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

convert_to_tensor = transforms.ToTensor()
convert_to_PIL = transforms.ToPILImage()

# device = torch.device("cpu")
 # img - tensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def merge(r, g, b, a=None): # PIL merge
    if (a == None):
        return torch.stack((r, g, b), 0)
    return torch.stack((r, g, b, a), 0)

def split(img):
    return [band for band in img]

def split_alpha(img):
    if (img.size()[0] == 3):
        return img, None
    return torch.split(img, 3)

def add_alpha(img, a):
    sample = torch.unsqueeze(torch.ones(img[0].shape, device=device) * a, 0)
    return torch.cat((img, sample))

def fill(size, color):
    colors = list(set(color))

    sample = torch.ones(size, device=device) # 
    if len(color)==3:
        r, g, b = color
        return merge(sample*r/255, sample*g/255, sample*b/255)
    else:
        r, g, b, a = color 
        return merge(sample*r/255, sample*g/255, sample*b/255, sample*a)

# def clip == torch.clamp for clipping values

def invert(img):
    return torch.ones_like(img, device=device) - img
 
def linear_transformation(img, matrix):
    if (img.shape[0]) == 1:
        img = torch.stack((img[0], img[0], img[0]), dim=0)
#     print(img)
    img = torch.transpose(img, 0, 2)
    img = torch.transpose(img, 1, 2)
    
#     print(matrix)
#     matrix = matrix.to(device=device, dtype=torch.double)
#     img = img.to(device, float)
    
    img = torch.matmul(matrix, img.float())
    img = torch.clamp(img, 0, 1)
  
    img = torch.transpose(img, 0, 1)
    img = torch.transpose(img, 1, 2)
    return img

def linear_gradient_mask(size, start=0, end=1, is_horizontal=True): # size - tuple [w, h]
    if is_horizontal:
        h, w = size
    else:
        w, h = size

    output = torch.linspace((1-start)*255, 0+(1-end)*255, round(w), device=device) # 

    output = torch.round(output)
    output = torch.unsqueeze(output, 1)
    output = output.expand(-1, h)

    output = torch.stack((output, output, output), dim=0)
    if is_horizontal:
        output = torch.transpose(output, 1, 2)
    return output/255


def linear_gradient(size, start, end, is_horizontal=True):

    assert len(size) == 2
    assert len(start) == 3
    assert len(end) == 3

    im_start = fill(size, start)
    im_end = fill(size, end)
    mask = linear_gradient_mask(size, is_horizontal=is_horizontal)

    return composite(im_start, im_end, mask)

def radial_gradient_mask(size, length=0, scale=1, center=(.5, .5)):

    if length >= 1:
        return torch.transpose(torch.tensor(()).new_ones(size, device=device), 0, 1) #Image.new('L', size, 255)

    if scale <= 0:
        return torch.transpose(torch.tensor(()).new_zeros(size, device=device), 0, 1)

    w, h = size
    cx, cy = center

    rw_left = w * cx 
    rw_right = w * (1 - cx) 
    rh_top = h * cy 
    rh_bottom = h * (1 - cy) 

    x = torch.linspace(-rw_left, rw_right, w, device=device)
    y = torch.linspace(-rh_top, rh_bottom, h, device=device)[:, None]

    # r is a radius to the farthest-corner
    r = math.sqrt(max(rw_left, rw_right) ** 2 + max(rh_top, rh_bottom) ** 2)
    base = max(scale - length, 0.001)  # avoid a division by zero

    mask = torch.sqrt(x ** 2 + y ** 2) / r  # distance from center
    mask = (mask - length) / base  # adjust ending shape
    mask = invert(mask)  # invert: distance to center
    # print(mask)
    # mask *= 255
    mask = mask.clamp(0, 1)
    return torch.transpose(mask, 0, 1)

def radial_gradient(size, colors, positions=None, **kwargs):
    if positions is None:
        positions = torch.linspace(0, 1, len(colors))
    else:
        assert len(positions) >= 2
        assert len(colors) == len(positions)

    colors = [fill(size, color) for color in colors]
    def compose(x, y):
        kwargs_ = kwargs.copy()
        kwargs_['length'] = x[1]
        kwargs_['scale'] = y[1]

        mask = radial_gradient_mask(size, **kwargs_)
        return (composite(x[0], y[0], mask), y[1])

    return reduce(compose, zip(colors, positions))[0]

def composite(img1, img2, mask):
    # print(img1.shape, img2.shape, mask.shape)
    return torch.mul(img1, mask) + torch.mul(invert(mask), img2)

def add(img1, img2): #test
    return torch.clamp(img1 + img2, 0, 255)

def subtract(img1, img2): #test
    return torch.clamp(img1 - img2, 0, 255)

def compose(img1, img2, mask):
    # print(img1.shape, img2.shape, mask.shape)
    return torch.mul(img2, mask) + torch.mul(invert(mask), img1)

# def normalize(img, mean, std, inplace=0): # mean - sequence of mean for each channel, std - sequence of standard deviations for each channel
# return transforms.functional.normalize(img, mean, std, inplace)