import os
import cv2
import torch
from torchvision import transforms

from PIL import Image
import numpy as np
import css
from css import *
import util
from util import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
convert_to_tensor = transforms.ToTensor()
convert_to_PIL = transforms.ToPILImage()

def aden(img, amount=1): # input image: img
    
        
    cb = torch.cat([img[:3]])  # RGB mode
    if (amount == 0):
        return cb
#     print(cb)
    cs = util.fill(cb[0].shape, [66, 10, 14, amount])
    cs = css.darken(cb, cs)

    mask = amount * linear_gradient_mask(cb[0].shape, start=0.8)
    cr = util.composite(cs, cb, mask)

    cr = css.hue_rotate(cr, -20/360 * amount)
    cr = css.contrast(cr, 1 - 0.1 * amount)
    cr = css.saturate(cr, 1 - 0.15 * amount)
    cr = css.brightness(cr, 1 + 0.2 * amount)
    return cr

def brannan(img, amount=1):
    cb = torch.cat([img[:3]])
    if (amount == 0):
        return cb
#     print(cb)
    cs = util.fill(cb[0].shape, [161, 44, 199, .31 * amount])
    cr = css.lighten(cb, cs)

    cr = css.sepia(cr, .5 * amount)
    cr = css.contrast(cr, 1 + amount * .4)

    return cr


def brooklyn(img, amount=1):
    cb = torch.cat([img[:3]]) 
    if (amount == 0):
        return cb
#     print(cb)
    cs1 = util.fill(cb[0].shape, [168, 223, 193, .4 * amount])
    cm1 = css.overlay(cb, cs1)

    cs2 = util.fill(cb[0].shape, [196, 183, 200, amount])
    cm2 = css.overlay(cb, cs2)

    gradient_mask = util.radial_gradient_mask(cb[0].size(), length=.7)
    
    cr = util.composite(cm1, cm2, gradient_mask)

    cr = css.contrast(cr, 1 - amount * .1)
    cr = css.brightness(cr, 1 + amount * .1)

    return cr

def clarendon(img, amount=1):
    cb = torch.cat([img[:3]]) 
    if (amount == 0):
        return cb
#     print(cb)
    cs = util.fill(cb[0].shape, [127, 187, 227, .2 * amount])
    cr = css.overlay(cb, cs)

    cr = css.contrast(cr, 1 + amount * .2)
    cr = css.saturate(cr, 1 + amount * .35)

    return cr

def earlybird(img, amount=1):
    cb = torch.cat([img[:3]]) 
    if (amount == 0):
        return cb
#     print(cb)
    cs = util.radial_gradient(cb[0].shape,
            [(208, 186, 142), (54, 3, 9), (29, 2, 16)],
            [.2, .85, 1])
    cs = util.add_alpha(cs, amount)
    cr = css.overlay(cb, cs)

    cr = css.contrast(cr, 1 - amount * .1)
    cr = css.sepia(cr, amount * .2)

    return cr

def f1977(img, amount=1):
#     return img
#     print(img)
    cb = torch.cat([img[:3]])
    if (amount == 0):
        return cb
#     print(cb)
    cs = util.fill(cb[0].shape, [243, 106, 188, .3 * amount])

    cr = css.screen(cb, cs)

    cr = css.contrast(cr, 1 + .1 * amount)
    cr = css.brightness(cr, 1 + .1 * amount)
    cr = css.saturate(cr, 1 + .3 * amount)

    return cr

def gingham(img, amount=1):
    cb = torch.cat([img[:3]]) 
    if (amount == 0):
        return cb
    cs = util.fill(cb[0].shape, [230, 230, 250, amount])
    
    cr = css.soft_light(cb, cs)

    cr = css.brightness(cr, 1 + amount * .05)
    cr = css.hue_rotate(cr, -10/360 * amount)

    return cr

def hudson(img, amount=1):
    cb = torch.cat([img[:3]]) 
    if (amount == 0):
        return cb
    cs = util.radial_gradient(cb[0].shape,
        [(166, 177, 255), (52, 33, 52)], [.5, 1])
    cs = css.multiply(cb, cs)
    cr = css.blend(cb, cs, .5 * amount)  # opacity

    cr = css.brightness(cr, 1 + amount * .2)
    cr = css.contrast(cr, 1 - amount * .1)
    cr = css.saturate(cr, 1 + amount * .1)

    return cr

def inkwell(img, amount=1):
    cb = torch.cat([img[:3]]) #util.or_convert(im, 'RGB')
#     print(cb)
    if (amount == 0):
        return cb
    cr = css.sepia(cb, 0.3 * amount)
    cr = css.contrast(cr, 1 + 0.1 * amount)
    cr = css.brightness(cr, 1 + 0.1 * amount)
    cs = css.grayscale(cr)
    cr = css.blend(cr, cs, amount)
    return cr

def kelvin(img, amount=1):
    cb = torch.cat([img[:3]])
    if (amount == 0):
        return cb
    cs1 = util.fill(cb[0].shape, [56, 44, 52, amount])
    cs = css.color_dodge(cb, cs1)

    cs2 = util.fill(cb[0].shape, [183, 125, 33, amount])
    cr = css.overlay(cs, cs2)

    return cr

def lark(img, amount=1):
    cb = torch.cat([img[:3]])
    if (amount == 0):
        return cb
    cs1 = util.fill(cb[0].shape, [34, 37, 63, amount])
    cm1 = css.color_dodge(cb, cs1)

    cs2 = util.fill(cb[0].shape, [242, 242, 242, .8 * amount])
    cr = css.darken(cm1, cs2)

    cr = css.contrast(cr, 1 - .1 * amount)

    return cr

def lofi(img, amount=1):   # ready to test
    cb = torch.cat([img[:3]])
    if (amount == 0):
        return cb
    cs = util.fill(cb[0].size(), [34, 34, 34, amount])
    cs = css.multiply(cb, cs)

    mask = util.radial_gradient_mask(cb[0].size(), length=.7, scale=1.5)
    cr = util.composite(cb, cs, mask)

    cr = css.saturate(cr, 1 + amount * .1)
    cr = css.contrast(cr, 1 + amount * .5)

    return cr

# def maven(img, amount=1): # дописать 
#     cb = torch.cat([img[:3]]) #util.or_convert(im, 'RGB')

#     cs = util.fill(cb[0].size(), [3, 230, 26]) # alpha = 0.2
#     cr = css.hue(cb, cs) # hue blending

#     cr = sepia(cr, .25)
#     cr = brightness(cr, .95)
#     cr = contrast(cr, .95)
#     cr = saturate(cr, 1.5)

#     return cr

def mayfair(img, amount=1):
    cb = torch.cat([img[:3]])
    if (amount == 0):
        return cb
    
    size = cb[0].shape
    pos = (.4, .4)

    cs1 = util.fill(size, [255, 255, 255, .8 * amount])
    cm1 = css.overlay(cb, cs1)

    cs2 = util.fill(size, [255, 200, 200, .6 * amount])
    cm2 = css.overlay(cb, cs2) 

    cs3 = util.fill(size, [17, 17, 17, amount])
    cm3 = css.overlay(cb, cs3)

    mask1 = util.radial_gradient_mask(size, scale=.3, center=pos)
    cs = util.composite(cm1, cm2, mask1)

    mask2 = util.radial_gradient_mask(size, length=.3, scale=.6, center=pos)
    cs = util.composite(cs, cm3, mask2)
    cr = css.blend(cb, cs, .4 * amount)  # opacity

    cr = css.contrast(cr, 1 + amount * .1)
    cr = css.saturate(cr, 1 + amount * .1)

    return cr

def moon(img, amount=1):
    cb = torch.cat([img[:3]])
    if (amount == 0):
        return cb

    cs1 = util.fill(cb[1].shape, [160, 160, 160, amount])
    cs = css.soft_light(cb, cs1)

    cs2 = util.fill(cb[1].shape, [56, 56, 56, amount])
    cr = css.lighten(cs, cs2)

    cs = css.grayscale(cr)
    cr = css.blend(cr, cs, amount)
    cr = css.contrast(cr, 1 + amount * .1)
    cr = css.brightness(cr, 1 + amount * .1)

    return cr

def nashville(img, amount=1):
    cb = torch.cat([img[:3]]) 
    if (amount == 0):
        return cb
    
    cs1 = fill(cb[0].shape, [247, 176, 153, 0.56 * amount]) #  alpha = 0.56
    cm1 = darken(cb, cs1)

    cs2 = fill(cb[0].shape, [0, 70, 150, 0.4 * amount])  # alpha = 0.4
    cr = lighten(cm1, cs2)

    cr = css.sepia(cr, 0.2 * amount)
    cr = css.contrast(cr, 1 + 0.2 * amount)
    cr = css.brightness(cr, 1 + 0.05 * amount)
    cr = css.saturate(cr, 1 + 0.2 * amount)

    return cr

def perpetua(img, amount=1):
    cb = torch.cat([img[:3]]) 
    if (amount == 0):
        return cb
    cs = util.linear_gradient(cb[0].shape, [0, 91, 154], [230, 193, 61], False)
    cs = css.soft_light(cb, cs)
    cr = css.blend(cb, cs, .5 * amount)  # opacity

    return cr

def reyes(img, amount=1):
    cb = torch.cat([img[:3]]) 
    if (amount == 0):
        return cb
    cs = util.fill(img[0].shape, [239, 205, 173, amount])
    cs = css.soft_light(cb, cs)
    cr = css.blend(cb, cs, .5 * amount)  # opacity

    cr = css.sepia(cr, .22 * amount)
    cr = css.brightness(cr, 1 + amount * .1)
    cr = css.contrast(cr, 1 - .15 * amount)
    cr = css.saturate(cr, 1 - .25 * amount)

    return cr

def rise(img, amount=1): 
    cb = torch.cat([img[:3]]) 
    if (amount == 0):
        return cb
    cs1 = util.fill(cb[0].shape, [236, 205, 169, .15 * amount])
    cm1 = css.multiply(cb, cs1)

    cs2 = util.fill(cb[0].shape, [50, 30, 7, .4 * amount])
    cm2 = css.multiply(cb, cs2)

    gradient_mask1 = util.radial_gradient_mask(cb[0].shape, length=.55)
    cm = util.composite(cm1, cm2, gradient_mask1)

    cs3 = util.fill(img[0].shape, [232, 197, 152, .8 * amount])
    cm3 = css.overlay(cm, cs3)

    gradient_mask2 = util.radial_gradient_mask(cb[0].shape, scale=.9)
    cm_ = util.composite(cm3, cm, gradient_mask2)
    cr = css.blend(cm, cm_, .6 * amount)  # opacity

    cr = css.brightness(cr, 1 + amount * .05)
    cr = css.sepia(cr, .2 * amount)
    cr = css.contrast(cr, 1 - amount * .1)
    cr = css.saturate(cr, 1 - amount * .1)

    return cr

def slumber(img, amount=1):
    cb = torch.cat([img[:3]]) 
    if (amount == 0):
        return cb
    cs1 = util.fill(img[0].shape, [69, 41, 12, .4 * amount])
    cm = css.lighten(cb, cs1)

    cs2 = util.fill(img[0].shape, [125, 105, 24, .5 * amount])
    cr = css.soft_light(cm, cs2)

    cr = css.saturate(cr, 1 - amount * .34)
    cr = css.brightness(cr, 1 + amount * .05)

    return cr

def stinson(img, amount=1):
    cb = torch.cat([img[:3]]) 
    if (amount == 0):
        return cb
    cs = util.fill(img[0].shape, [240, 149, 128, .2 * amount])
    cr = css.soft_light(cb, cs)

    cr = css.contrast(cr, 1 - amount * .25)
    cr = css.saturate(cr, 1 - amount * .15)
    cr = css.brightness(cr, 1 + amount * .15)

    return cr

def toaster(img, amount=1):
    cb = torch.cat([img[:3]]) 
    if (amount == 0):
        return cb
    cs = util.radial_gradient(img[0].shape, [(128, 78, 15), (59, 0, 59)])
    cs = util.add_alpha(cs, amount)
    cr = css.screen(cb, cs)

    cr = css.contrast(cr, 1 + amount * .5)
    cr = css.brightness(cr, 1 - amount * .1)

    return cr


def valencia(img, amount=1):
    cb = torch.cat([img[:3]]) 
    if (amount == 0):
        return cb
    cs = util.fill(cb[0].shape, [58, 3, 57])
    cs = css.exclusion(cb, cs)
    cr = css.blend(cb, cs, .5 * amount)  # opacity

    cr = css.contrast(cr, 1 + amount * .08)
    cr = css.brightness(cr, 1 + amount * .08)
    cr = css.sepia(cr, amount * .08)

    return cr

def walden(img, amount=1):
    cb = torch.cat([img[:3]]) 
    if (amount == 0):
        return cb
    cs = util.fill(cb[0].shape, [0, 68, 204])
    cs = css.screen(cb, cs)
    cr = css.blend(cb, cs, .3 * amount)  # opacity

    cr = css.brightness(cr, 1 + .1 * amount)
    cr = css.hue_rotate(cr, -10/360 * amount)
    cr = css.sepia(cr, .3 * amount)
    cr = css.saturate(cr, 1 + .6 * amount)

    return cr

# def willow(img, amount=1): # color blending 
#     cb = torch.cat([img[:3]]) 

#     cs1 = util.radial_gradient(
#             cb[0].size(),
#             [(212, 169, 175), (0, 0, 0)],
#             [.55, 1.5])
#     cm1 = css.overlay(cb, cs1)

#     cs2 = util.fill(cb.size, [216, 205, 203])
#     cr = css.color(cm1, cs2) 

#     cr = css.grayscale(cr, .5)
#     cr = css.contrast(cr, .95)
#     cr = css.brightness(cr, .9)

#     return cr

def xpro2(img, amount=1):

    cb = torch.cat([img[:3]]) 
    if (amount == 0):
        return cb
    cs1 = util.fill(cb[0].shape, [230, 231, 224])
    cs2 = util.fill(cb[0].shape, [43, 42, 161])
    cs2 = css.blend(cb, cs2, .6)
    
    gradient_mask = util.radial_gradient_mask(cb[0].shape, length=.4, scale=1.1)
    cs = util.composite(cs1, cs2, gradient_mask)
    cs = util.add_alpha(cs, amount)
    cm1 = css.color_burn(cb, cs) 
    # cm2 = cm1.detach().clone() # ?????
    cm2 = cm1
    cm2 = css.blend(cb, cm2, .6 * amount)
    cr = util.composite(cm1, cm2, gradient_mask)

    cr = css.sepia(cr, .3 * amount)

    return cr


def brighten(img, amount=0.5):
    return transforms.functional.adjust_brightness(img, amount * 2)
#     return css.brightness(img, amount * 2)

def contrast(img, amount=0.5):
    return css.contrast(img, amount * 2)

def hue(img, amount=0.5):
    return css.hue_rotate(img, amount - .5)

def saturate(img, amount=0.5):
    return css.saturate(img, amount * 2)

    
# img2 = Image.open('cat.png')
# img2 = convert_to_tensor(img2)


# convert_to_PIL(util.composite(img1, img2, mask))

# convert_to_PIL(earlybird(img2, 0.8)).show()

