# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:38:32 2021

@author: Not Your Computer
"""

import numpy as np
import pandas as pd
from scipy.ndimage.filters import convolve
from skimage.transform import resize, rescale
import matplotlib.pyplot as plt
from PIL import Image


def Invert(img):
    img = img + 1
    img[np.where(img == 2)] = 0
    return img


def RemoveStrayPixels(img, k = 3):
    h,w = np.shape(img)
    rows = []
    cols = []
    for i in range(k,h-k):
        for j in range(k,w-k):
            if img[i,j] == 1:
                if np.sum(img[i-k:i+k,j-k:j+k]) == 1:
                    rows.append(i)
                    cols.append(j)
    img[rows,cols] = 0
    return img


# Calculate bounds occupied by pixels
def PixelBounds(img, buffer = 10, k = 3):
    h,w = np.shape(img)
    img = RemoveStrayPixels(img, k = k)
    xmin = np.where(img.sum(axis = 0) > 0)[0][0] - buffer
    ymin = np.where(img.sum(axis = 1) > 0)[0][0] - buffer
    xmax = np.where(img.sum(axis = 0) > 0)[0][-1] + buffer
    ymax = np.where(img.sum(axis = 1) > 0)[0][-1] + buffer
    return max(0,xmin), max(0,ymin), min(xmax,w), min(ymax,h)

