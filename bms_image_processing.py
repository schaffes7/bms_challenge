# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:38:32 2021

@author: Not Your Computer
"""

import os
import numpy as np
import pandas as pd
import random
import itertools
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


def IOU(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


# Save all logged annotations as image chips in separate folder
def SaveChips(entry, outdir = r'D:\DataStuff\bms\chips'):
    entry = entry.split(',')
    path = entry[0]
    fname = os.path.basename(os.path.dirname(path))
    n_entries = int(len(entry[3:]) / 9)
    img = plt.imread(path)
    for i in range(n_entries):
        x,y,w,h,label,pxmin,pymin,pxmax,pymax = entry[3+9*i:3+9*(i+1)]
        x = int(float(x)); y = int(float(y))
        w = int(float(w)/2); h = int(float(h)/2)
        
        chip = img[y-h:y+h, x-w:x+w]
        plt.imshow(chip, cmap = 'gray')
        plt.show()
        outfile = '{}\\{}_{}_{}.png'.format(outdir, fname, label, i)
        plt.imsave(outfile, chip, cmap = 'gray')
        
        
# Create a new image w/ image chips
def AlphabetSoup(labels,
                 srcdir = r'D:\DataStuff\bms\chips',
                 target_shape = (640,640),
                 output_type = 'yolo',
                 size_thresh = 100,
                 preprocess = True,
                 min_size = 10,
                 buffer = 100,
                 S = (10,10),
                 B = 1,
                 C = 0):
    
    
    flist = os.listdir(srcdir)
    sf = random.random() * 1.5 + 1
    img = np.zeros(target_shape)
    boxes = []
    
    
    if output_type == 'yolo':
        annotations = np.zeros((S[0], S[1], (5*B+C)))
        grid_h = target_shape[0] / S[0]
        grid_w = target_shape[1] / S[1]
    
    if output_type == 'unet':
        mask = np.zeros(target_shape)
    
    
    for fname in random.sample(flist, random.randint(1,15)):
        fpath = os.path.join(srcdir, fname)
        label = fpath.split('.')[0].split('_')[-2]
        chip = Invert(plt.imread(fpath)[:,:,0])
        
        x = int(random.random() * (target_shape[1] - 2*buffer)) + buffer
        y = int(random.random() * (target_shape[0] - 2*buffer)) + buffer
        
        
        # Convolve the image to flesh out broken lines
        if preprocess:
            chip = convolve(chip, [[1,1,1],[1,1,1],[1,1,1]])
            chip = chip / np.max(chip)
            chip = convolve(chip, [[1,1,1],[1,1,1],[1,1,1]])
            chip = chip / np.max(chip)
            
        chip = rescale(chip, sf)
        h,w = np.shape(chip)
        chip_box = [x,y,x+w,y+h]
        
        
        no_overlap = True
        for box in boxes:
            if IOU(chip_box, box) > 0.10:
                no_overlap = False
                break
                
            
        if no_overlap:
            img[y:y+h, x:x+w] = chip
            boxes.append(chip_box)

            
            if output_type == 'yolo':
                row = int(np.floor(y / grid_h))
                col = int(np.floor(x / grid_w))
                
                x = x - col*grid_w
                y = y - row*grid_h
                
                x = x / grid_w
                y = y / grid_h
                
                h = min(h/(size_thresh-min_size),1)
                w = min(w/(size_thresh-min_size),1)
                annotations[row,col,:] = [x,y,w,h,1]
            
            if output_type == 'unet':
#                mask[y:y+h,x:x+w] = labels.index(label)
                mask[y:y+h,x:x+w] = 1
            
    if output_type == 'yolo':
        return img, annotations
    
    if output_type == 'unet':
        return img, mask

#
## Create a new image w/ image chips
#def AlphabetSoup(srcdir = r'D:\DataStuff\bms\chips', target_shape = (640,640), size_thresh = 100, buffer = 100, S = (10,10), B = 1, C = 0, preprocess = True, min_size = 10):
#    flist = os.listdir(srcdir)
#    fnames = random.sample(flist, random.randint(1,15))
#    sf = random.random() * 1 + 1
#    img = np.zeros(target_shape)
#    boxes = []
#    annotations = np.zeros((S[0], S[1], (5*B+C)))
#    grid_h = target_shape[0] / S[0]
#    grid_w = target_shape[1] / S[1]
#    
#    for fname in fnames:
#        fpath = os.path.join(srcdir, fname)
#        label = fpath.split('.')[0].split('_')[-2]
#        chip = Invert(plt.imread(fpath)[:,:,0])
#        
#        x = int(random.random() * (target_shape[1] - 2*buffer)) + buffer
#        y = int(random.random() * (target_shape[0] - 2*buffer)) + buffer
#        
#        # Convolve the image to flesh out broken lines
#        if preprocess:
#            chip = convolve(chip, [[1,1,1],[1,1,1],[1,1,1]])
#            chip = chip / np.max(chip)
#            chip = convolve(chip, [[1,1,1],[1,1,1],[1,1,1]])
#            chip = chip / np.max(chip)
#            
#        chip = rescale(chip, sf)
#        h,w = np.shape(chip)
#        
#        chip_box = [x,y,x+w,y+h]
#        
#        no_overlap = True
#        for box in boxes:
#            if IOU(chip_box, box) > 0.10:
#                no_overlap = False
#                break
#                
#        if no_overlap:
#            img[y:y+h, x:x+w] = chip
#            boxes.append(chip_box)
#            row = int(np.floor(y / grid_h))
#            col = int(np.floor(x / grid_w))
#            
#            x = x - col*grid_w
#            y = y - row*grid_h
#            
#            x = x / grid_w
#            y = y / grid_h
#            
#            h = min(h/(size_thresh-min_size),1)
#            w = min(w/(size_thresh-min_size),1)
#            
#            annotations[row,col,:] = [x,y,w,h,1]
#            
#    return img, annotations