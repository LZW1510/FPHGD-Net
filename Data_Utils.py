import os
from os import listdir
from os.path import join
import platform

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, \
    RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, Grayscale

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import DataLoader


import random
import glob
from time import time

import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt

import cv2





def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.bmp', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, block_size):
    return crop_size - (crop_size % block_size)


def train_hr_transform(crop_size):
    seed = random.randint(0, 1)
    return Compose([
        RandomCrop(crop_size),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        # transforms.RandomApply(RandomRotation(degrees=(-90, -90)), p=0.5),
        # transforms.RandomApply(RandomRotation(degrees=(90, 90)), p=0.5),
        Grayscale(),
        ToTensor(),
    ])


class TrainDatasetFromFolder(Dataset):

    def __init__(self, data_dir, crop_size, block_size):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(data_dir, x) for i in range(200) for x in listdir(data_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, block_size)
        self.hr_transform = train_hr_transform(crop_size)

    def __getitem__(self, item):
        try:
            hr_image = self.hr_transform(Image.open(self.image_filenames[item]))
            return hr_image
        except:
            hr_image = self.hr_transform(Image.open(self.image_filenames[item + 1]))
            return hr_image

    def __len__(self):
        return len(self.image_filenames)


def imread_CS_py(Iorg, block_size):
    block_size = block_size
    [row, col] = Iorg.shape
    row_pad = 0
    col_pad = 0
    if np.mod(row, block_size)!=0:
        row_pad = block_size - np.mod(row, block_size)
    if np.mod(col, block_size) != 0:
        col_pad = block_size - np.mod(col, block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col + col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]


def imread_CS_py2(Iorg, block_size):
    block_size = block_size
    [row, col] = Iorg.shape
    if (np.mod(row, block_size==0)):
        row_pad = 0
    else:
        row_pad = block_size - np.mod(row, block_size)
    if (np.mod(col, block_size)==0):
        col_pad = 0
    else:
        col_pad = block_size - np.mod(col, block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col + col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]

# 图像的每个块元素按列排列
def img2col_py(Ipad, block_size):
    [row, col] = Ipad.shape
    row_block = row / block_size
    col_block = col / block_size
    block_num = int(row_block * col_block)
    img_col = np.zeros([block_size ** 2, block_num])
    count = 0
    for x in range(0, row - block_size + 1, block_size):
        for y in range(0, col - block_size + 1, block_size):
            img_col[:, count] = Ipad[x:x + block_size, y:y + block_size].reshape([-1])
            count = count + 1
    return img_col


# 将列排列的元素恢复为图像按块的排列
def col2im_CS_py(X_col, row, col, row_new, col_new, block_size):
    block_size = block_size
    X0_rec = np.zeros([row_new, col_new])
    count = 0
    for x in range(0, row_new - block_size + 1, block_size):
        for y in range(0, col_new - block_size + 1, block_size):
            X0_rec[x:x + block_size, y:y + block_size] = X_col[:, count].reshape([block_size, block_size])
            count = count + 1
    x_rec = X0_rec[:row, :col]

    return x_rec


def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def Fourier_Transform(x_input):
    f = torch.fft.fft2(x_input, dim=(-2, -1))
    f = torch.abs(f)

    return f


def Fourier_Inverse_Transform(x_input):
    f = torch.fft.ifftshift(x_input, dim=(-2, -1))
    img = torch.fft.ifft2(f, dim=(-2, -1))
    img = torch.abs(img)

    return img


def HighPassfilter(x_input):
    row, col = x_input.size(-2), x_input.size(-1)
    mask = torch.ones((row, col)).to(device='cuda:0')
    center_x, center_y = int(col / 2), int(row / 2)
    mask[center_x - 5:center_x + 5, center_y - 5:center_y + 5] = 0
    f = Fourier_Transform(x_input)
    Highf = torch.abs(f) * mask
    img = Fourier_Inverse_Transform(Highf)

    return img


def PhiTPhifun(PhiT, Phi, x):
    """
    :param PhiT:transpose of measurement matrix
    :param Phi: measurement matrix
    :param x: image
    :return: PhiTPhix
    """
    x = F.conv2d(x, Phi, stride=32, bias=None)
    x = F.conv2d(x, PhiT, stride=1, bias=None)

    return nn.PixelShuffle(32)(x)


def Channel_2_Batch(x_input):
    B, C, H, W = x_input.shape
    x = x_input.view(-1, 1, H, W)

    return x


def Batch_2_Channel(x_input, input_channel):
    H, W = x_input.size(2), x_input.size(3)
    x = x_input.view(-1, input_channel, H, W)

    return x

def edge_extractor(x_input):

    x_input = x_input.squeeze(1)
    x = x_input.cpu().data.numpy()
    x = 255 * np.clip(x, 0, 1).astype(np.uint8)
    for i in range(64):
        x[i, :, :] = cv2.Canny(x[i, :, :], 50, 150)

    return x

def setup_seed(seed=1):
    """
    fix random seed
    :param seed: random seed
    """
    torch.manual_seed(seed) #为CPU设置随机种子
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)    #为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)    #为所有GPU设置随机种子
