import math
from functools import reduce
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms as T
import util

# from pilgram.util import fill, invert

# device = torch.device("cpu")
convert_to_tensor = transforms.ToTensor()
convert_to_PIL = transforms.ToPILImage()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def brightness(img, amount=1):  # [0, 2]  (1.0 - original) 
    if (img.shape[0] == 4):
        rgb_img = torch.stack((img[0], img[1], img[2]), dim=0)
        rgb_img = transforms.functional.adjust_brightness(rgb_img, amount)
        return torch.stack((rgb_img[0], rgb_img[1], rgb_img[2], img[3]), dim=0)
    else:
        return transforms.functional.adjust_brightness(img, amount)

def contrast(img, amount=1):  # [0, 2]  (1.0 - original)
    if (img.shape[0] == 4):
        rgb_img = torch.stack((img[0], img[1], img[2]), dim=0)
        rgb_img = transforms.functional.adjust_contrast(rgb_img, amount)
        return torch.stack((rgb_img[0], rgb_img[1], rgb_img[2], img[3]), dim=0)
    else:
        return transforms.functional.adjust_contrast(img, amount)

def hue_rotate(img, amount=0): #[-0.5,0.5] (0.0 - original)
    if (img.shape[0] == 4):
        rgb_img = torch.stack((img[0], img[1], img[2]), dim=0)
        rgb_img = transforms.functional.adjust_hue(rgb_img, amount)
        return torch.stack((rgb_img[0], rgb_img[1], rgb_img[2], img[3]), dim=0)
    else:
        return transforms.functional.adjust_hue(img, amount)

def saturate(img, amount): #[0,2] (1.0 - original)
    return transforms.functional.adjust_saturation(img, amount)

def grayscale(img, amount=3): # amount - output channels (1, 2, 3)
    return transforms.functional.rgb_to_grayscale(img, amount)

def sepia(img, amount): # 0 <= amount < 1
    amount = 1 - amount
    matrix = torch.tensor([[0.393 + 0.607 * amount, 0.769 * (1 - amount), 0.189 * (1 - amount)], 
        [0.349 * (1 - amount), 0.686 + 0.314 * amount, 0.168 * (1 - amount)], 
        [0.272 * (1 - amount), 0.534 * (1 - amount), 0.131 + 0.869 * amount]], dtype=torch.float, device=device)
    return util.linear_transformation(img, matrix)

def gamma(img, gamma, gain=1): # gamma correction: gamma>=0
    return transforms.functional.adjust_gamma(img, gamma, gain)

def sharpness(img, amount): #[0,2]
    return transforms.functional.adjust_sharpness(img, amount)

# def equalize(img): 
#     convert_uint8 = transforms.ConvertImageDtype(torch.torch.uint8)
#     img = convert_uint8(img)
#     return transforms.functional.equalize(img)

def blend(img1, img2, a):
    return img1*(1 - a) + img2*a

def _darken(img1, img2):
    return torch.minimum(img1, img2)

def _lighten(img1, img2):
    return torch.maximum(img1, img2)

def lighten(img1, img2):
    return alpha_blend(img1, img2, _lighten)

def darken(img1, img2):
    return alpha_blend(img1, img2, _darken)

def _screen(img1, img2):
    return torch.clamp((1 - torch.mul(util.invert(img1), util.invert(img2))/1), 0, 1)
    
def screen(img1, img2):
    return alpha_blend(img1, img2, _screen)
    

def _color_dodge(img1, img2):
    r, g, b = [((cb != 0) * (cs_inv == 0) + (cb / cs_inv)) for cb, cs_inv in zip(util.split(img1), util.split(util.invert(img2)))]
    return util.merge(r, g, b)


def color_dodge(img1, img2):
    return alpha_blend(img1, img2, _color_dodge)

def color_burn(img1, img2):
    return alpha_blend(img1, img2, _color_burn)

def _color_burn(img1, img2):
    r, g, b = [((cb == 1) + (cb < 1) * (cs > 0) * (1 - ((1 - cb) / cs))) for cb, cs in zip(util.split(img1), util.split(img2))]
    return util.merge(r, g, b)


'''

def lum_im(im):
    """Returns luminosity as image.
    The formula is defined as:
        Lum(C) = 0.3 x Cred + 0.59 x Cgreen + 0.11 x Cblue

    See: https://www.w3.org/TR/compositing-1/#blendingnonseparable
    Arguments:
        im: An input image.
    Returns:
        The luminosity image.
    """
    return im.convert('L') # convert to L = R * 299/1000 + G * 587/1000 + B * 114/1000

def _color_image_math(cs, lum_cb, lum_cs):
    """Returns ImageMath operands for color blend mode"""
    cs = [_float(c) for c in cs]
    lum_cb = _float(lum_cb)
    lum_cs = _float(lum_cs)

    return set_lum_im(cs, lum_cb, lum_cs)


def _color(im1, im2):
    """The color blend mode.
    Arguments:
        im1: A backdrop image (RGB).
        im2: A source image (RGB).
    Returns:
        The output image.
    """

    r, g, b = im2.split()  # Cs
    lum_cb = lum_im(im1)   # Lum(Cb)
    lum_cs = lum_im(im2)   # Lum(C) in SetLum

    bands = ImageMath.eval(
        'f((r, g, b), lum_cb, lum_cs)',
        f=_color_image_math, r=r, g=g, b=b, lum_cb=lum_cb, lum_cs=lum_cs)
    bands = [_convert(band, 'L').im for band in bands]

    return Image.merge('RGB', bands)

'''

def _d_cb(img):
    """Returns D(Cb) - Cb"""

    d = torch.mul((torch.mul((torch.mul(img, 16) - 12 * torch.ones_like(img)), img) + 4 * torch.ones_like(img)), img)
    d = torch.where(img <= .25, d, torch.sqrt(img))
    return d - img


def _soft_light(img1, img2):
    _1_2_x_cs = torch.clamp((torch.ones_like(img2) - torch.mul(img2, 2)), 0, 1)

    cb_x_1_cb = torch.clamp(torch.mul(img1, util.invert(img1)), 0, 1)


    c1 = util.subtract(img1, multiply(_1_2_x_cs, cb_x_1_cb))

    _2_x_cs_1 = torch.clamp(torch.mul(img2, 2) - torch.ones_like(img2), 0, 1)

    d_cb = _d_cb(img1)
    c2 = add(img1, multiply(_2_x_cs_1, d_cb))

    cm = torch.where(img2 < .5, c1, c2)
    return cm


def soft_light(im1, im2):
    return alpha_blend(im1, im2, _soft_light)

def _hard_light(img1, img2):
    img2_multiply = torch.clamp(torch.mul(img2, 2), 0, 1)
    mult = multiply(img1, img2_multiply)

    img2_screen = torch.clamp(torch.mul(img2, 2) - torch.ones_like(img2), 0, 1)
    sc = screen(img1, img2_screen)

    cm = torch.where(img2 < .5, mult, sc)
    return cm

def hard_light(img1, img2):
    return alpha_blend(img1, img2, _hard_light)

def overlay(img1, img2):
    return hard_light(img2, img1)

def _exclusion(img1, img2):
    sc = screen(img1, img2)
    mult = multiply(img1, img2)
    return sc - mult

def exclusion(img1, img2):
    return alpha_blend(img1, img2, _exclusion)

def color_burn(img1, img2):
    return alpha_blend(img1, img2, _color_burn)

def color(img1, img2):
    return alpha_blend(img1, img2, _color)


def _multiply(img1, img2):
    return torch.clamp(torch.mul(img1 * 255, img2 * 255)/255**2, 0, 1)

def multiply(img1, img2):
    return alpha_blend(img1, img2, _multiply)

def substract(img1, img2):
#     print(img1 - img2)
    return torch.clamp(img1 - img2, 0, 1)

def alpha_blending(img1, img2, a):
    cs = img1 + img2 * (1 - a)
    return torch.clamp(cs, 0, 255)

def add(img1, img2):
    return torch.clamp(img1 + img2, 0, 1)

def add3(img1, img2, img3):
    return torch.clamp(torch.stack((img1, img2, img3), 0) , 0, 255)

def alpha_to_rgb(img):
    a = img[0]
    return torch.stack((a, a, a), 0)

def alpha_blend(img1, img2, blending):
    im1, a1 = util.split_alpha(img1)
    im2, a2 = util.split_alpha(img2)

    im_blended = blending(im1, im2)
    if a1 is not None and a2 is not None:
        im_blended_alpha = multiply(a1, a2)
        im1_alpha = subtract(a1, im_blended_alpha)
        im2_alpha = subtract(a2, im_blended_alpha)
        c1 = multiply(alpha_to_rgb(im2_alpha), im2)
        c2 = multiply(alpha_to_rgb(im_blended_alpha), im_blended)
        c3 = multiply(alpha_to_rgb(im1_alpha), im1)
        im_blended = add3(c1, c2, c3)
    elif a1 is not None:
        a1_rgb = alpha_to_rgb(a1)
        a1_invert_rgb = alpha_to_rgb(util.invert(a1))
        im_blended = add(
            multiply(a1_rgb, im_blended),
            multiply(a1_invert_rgb, im2))
    elif a2 is not None:
        a2_rgb = alpha_to_rgb(a2)
        a2_invert_rgb = alpha_to_rgb(util.invert(a2))
        im_blended = add(
            multiply(a2_rgb, im_blended),
            multiply(a2_invert_rgb, im1))
    return im_blended
