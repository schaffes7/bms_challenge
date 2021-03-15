# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.ndimage.filters import convolve
import random
from random import shuffle
from skimage.transform import resize, rescale
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Input, ZeroPadding2D, BatchNormalization
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from bms_image_processing import Invert, PixelBounds


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def RecSearch(search_dir, ext, outfile = None, verbose = False):
    fpaths = []
    for f in os.listdir(search_dir):
        fpath = os.path.join(search_dir, f)
        if os.path.isdir(fpath):
            fpaths += RecSearch(fpath, ext, outfile = outfile)
        else:
            if f.split('.')[-1].lower() == ext:
                fpaths.append(fpath)
                if outfile != None:
                    if os.path.exists(outfile):
                        with open(outfile,'a') as f:
                            outstr = fpath + '\n'
                            f.write(outstr)
                    else:
                        with open(outfile,'w') as f:
                            outstr = fpath + '\n'
                            f.write(outstr)
    if verbose:
        print('\n{} {} files found in {}'.format(len(fpaths), ext, search_dir))
    return fpaths


def LoadDF(path = r"D:\DataStuff\bms\train_labels.csv"):
    df = pd.read_csv(path)
    df = df.InChI.str.split('/', expand = True)
    return


# =============================================================================
# IMAGE FUNCTIONS
# =============================================================================

def SampleImages(srcdir = 'D:\\DataStuff\\bms\\train', N = 1000):
    opts = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
    paths = []
    while len(paths) < N:
        subdir = os.path.join(srcdir, random.choice(opts), random.choice(opts), random.choice(opts))
        flist = os.listdir(subdir)
        img_path = os.path.join(subdir, random.choice(flist))
        paths.append(img_path)
    return paths


def LoadImg(path, target_shape = (416,416), invert = True, preprocess = True):
    # Read in image
    img = plt.imread(path)
    h,w = np.shape(img)
    H,W = target_shape
    
    if h > H or w > W:
        img = Image.open(path)
        
        if h > H:
            img = img.resize((w, target_shape[0]))
            h = target_shape[0]
        
        if w > W:
            img = img.resize((target_shape[1], h))
            w = target_shape[1]
        
        img = np.array(img).astype('float32')
        img[np.where(img == 255)] = 1
    
    # Invert image if needed
    if invert:
        img = Invert(img)
    
    # Convolve the image to flesh out broken lines
    if preprocess:
        img = convolve(img, [[1,1,1],[1,1,1],[1,1,1]])
        img = img / np.max(img)
    
    # Center the image in the target-sized field
    out_img = np.zeros(target_shape)
    if h <= target_shape[0] and w <= target_shape[1]:
        h_buff = int((target_shape[0] - h)/2)
        w_buff = int((target_shape[1] - w)/2)
        out_img[h_buff:h_buff+h, w_buff:w_buff+w] = img
    
    return out_img


def LoadImg2(path, target_shape = (416,416), invert = True, preprocess = True, crop = True, scale_images = True, buffer = 10, k = 3):
    # Read in image
    img = plt.imread(path)
    
    # Invert image if needed
    if invert:
        img = Invert(img)
    
    # Crop Image
    if crop:
        xmin,ymin,xmax,ymax = PixelBounds(img, buffer = buffer, k = k)
        h = ymax - ymin
        w = xmax - xmin
        img = img[ymin:ymax,xmin:xmax]

    # Convolve the image to flesh out broken lines
    if preprocess:
        img = convolve(img, [[1,1,1],[1,1,1],[1,1,1]])
        img = img / np.max(img)
        img = convolve(img, [[1,1,1],[1,1,1],[1,1,1]])
        img = img / np.max(img)
        
    # Scale Image
    if scale_images or max(h,w) >= max(target_shape):
        sf = target_shape[np.argmax([h,w])] / max(h,w)
        img = rescale(img, sf)
    
    
    
    # Center the image in the target-sized field
    out_img = np.zeros(target_shape)
    h,w = np.shape(img)
#    h_buff = int((target_shape[0] - h)/2)
#    w_buff = int((target_shape[1] - w)/2)
#    out_img[h_buff:h_buff+h, w_buff:w_buff+w] = img
    out_img[0:h,0:w] = img
    
    return out_img


# =============================================================================
# LABELME ANNOTATION FUNCTIONS
# =============================================================================

# PRINT OUT LIST OF LABELME COMMANDS TO GENERATE ANNOTATION FILES FROM JSONS
def LabelMeCommands(folder = 'D:\\DataStuff\\bms\\annotations', outfile = None):
    command_list = []
    flist = os.listdir(folder)
    for f in flist:
        if f.split('.')[-1].lower() == 'json':
            fpath = '{}\\{}'.format(folder, f)
            command = 'labelme_json_to_dataset "{}" -o "D:\\DataStuff\\bms\\annotations\\{}"'.format(fpath, fpath.split('\\')[-1].split('.')[0])
            img_path = fpath.replace('.json','\\img.png')
            if not os.path.exists(img_path):
                print(command)
            command_list.append(command)
    if outfile != None:
        with open(outfile, 'w') as f:
            for com in command_list:
                outstr = '{}\n'.format()
                f.write(outstr)
        print('\n[LabelMeCommands]: Commands written to file: ', outfile)


# RETURN ANNOTATED IMAGE FILEPATHS
def ImageFiles(srcdir = 'D:\\DataStuff\\bms\\annotations'):
    paths = []
    png_paths = RecSearch(srcdir, 'png')
    for path in png_paths:
        if os.path.basename(path) == 'img.png':
            paths.append(path)
    return paths


# RETURN ANNOTATED MASK FILEPATHS
def MaskFiles(srcdir = 'D:\\DataStuff\\bms\\annotations'):
    paths = []
    png_paths = RecSearch(srcdir, 'png')
    for path in png_paths:
        if os.path.basename(path) == 'label.png':
            paths.append(path)
    return paths


# RETURN ANNOTATED LABEL FILEPATHS
def LabelFiles(srcdir = 'D:\\DataStuff\\bms\\annotations'):
    paths = []
    txt_paths = RecSearch(srcdir, 'txt')
    for path in txt_paths:
        if os.path.basename(path) == 'label_names.txt':
            paths.append(path)
    return paths


def ImageLabels(path):
    ext = path.split('.')[-1].lower()
    if ext == 'png':
        label_path = '{}\\label_names.txt'.format(os.path.dirname(path))
    else:
        label_path = path
    with open(label_path, 'r') as f:
        labels = f.read()
    labels = labels.split('\n')[:-1]
    for i in range(len(labels)):
        labels[i] = labels[i].lower()
    return labels


def Annotations(path, keep_labels = []):
    entries = []
    if len(keep_labels) > 0:
        # Use only if specified
        labels = keep_labels
    else:
        # Default set of labels is everything
        labels = Labels()[1:]
    fname = os.path.basename(path).split('.')[0]
    img_path = os.path.join(os.path.dirname(path), fname, 'img.png')
    
    with open(path,'r') as f:
        content = json.load(f)
    W = int(content['imageWidth'])
    H = int(content['imageHeight'])
    entries = [img_path, W, H]
    
    # For each entry
    for entry in content['shapes']:
        label = entry['label']
        
        x1 = float(entry['points'][0][0])
        y1 = float(entry['points'][0][1])
        x2 = float(entry['points'][1][0])
        y2 = float(entry['points'][1][1])
        x = (x2 + x1) / 2
        y = (y2 + y1) / 2
        
        w = abs(x2-x1)
        h = abs(y2-y1)
        
        # If we want to keep this label...
        if label in labels:
            entries += [x, y, w, h, labels.index(label)]
        
    return entries



def Annotations2(path, keep_labels = [], buffer = 10, k = 3):
    entries = []
    if len(keep_labels) > 0:
        # Use only if specified
        labels = keep_labels
    else:
        # Default set of labels is everything
        labels = Labels()[1:]
    fname = os.path.basename(path).split('.')[0]
    img_path = os.path.join(os.path.dirname(path), fname, 'img.png')
    
    with open(path,'r') as f:
        content = json.load(f)
    W = int(content['imageWidth'])
    H = int(content['imageHeight'])
    entries = [img_path, W, H]
    
    # For each entry
    for entry in content['shapes']:
        label = entry['label']
        
        x1 = float(entry['points'][0][0])
        y1 = float(entry['points'][0][1])
        x2 = float(entry['points'][1][0])
        y2 = float(entry['points'][1][1])
        x = (x2 + x1) / 2
        y = (y2 + y1) / 2
        
        w = abs(x2-x1)
        h = abs(y2-y1)
        
        # If we want to keep this label...
        if label in labels:
            img = Invert(plt.imread(img_path))
            pxmin, pymin, pxmax, pymax = PixelBounds(img, buffer = buffer, k = k)
            entries += [x, y, w, h, label, pxmin, pymin, pxmax, pymax]
        
    return entries



def AnnoDF(srcdir = 'D:\\DataStuff\\bms\\annotations'):
    json_paths = RecSearch(srcdir, 'json')
    df = []
    for path in json_paths:
        annos = Annotations(path)
        path,W,H = annos[0:3]
        n_entries = int(len(annos[3:]) / 5)
        for i in range(n_entries):
            x,y,w,h,c = annos[3+5*i:3+5*i+5]
            row = [path,W,H,x,y,w,h,c]
            df.append(row)
    df = pd.DataFrame(df, columns = ['path','W','H','x','y','w','h','c'])
    df.to_csv('D:\\DataStuff\\bms\\annotations.csv', index = False)
    return df


# RETURN A LIST OF UNIQUE ANNOTATION LABELS
def Labels(srcdir = 'D:\\DataStuff\\bms\\annotations'):
    all_labs = []
    img_paths = RecSearch(srcdir, 'png')
    for path in img_paths:
        fpath = os.path.join(os.path.dirname(path), 'label_names.txt')
        with open(fpath, 'r') as f:
            labs = f.read()
        labs = labs.split('\n')[:-1]
        for i in range(len(labs)):
            labs[i] = labs[i].lower()
        all_labs += labs
    all_labs = list(set(all_labs))
    all_labs.sort()
    return all_labs


# CONVERT LABELME ANNOTATIONS TO YOLO FORMAT
# format: [img_path, x_min, y_min, x_max, y_max, class, x_min, y_min, x_max, y_max, class...]
def ConvertAnnotations(srcdir = 'D:\\DataStuff\\bms\\annotations',
                       outpath = 'D:\\DataStuff\\bms\\annotations.txt',
                       target_shape = (416,416),
                       keep_labels = []):
    
    if os.path.exists(outpath):
        print('\nRemoving old annotations file...')
        os.remove(outpath)
    print('\nFinding JSONs...')
    paths = RecSearch(srcdir, ext = 'json')
    entries = []
    for path in paths:
        
        # Get annotations for image
        annos = Annotations(path = path, keep_labels = keep_labels)
        n_entries = int(len(annos[3:])/5)
        W,H = annos[1:3]
        annos[1] = target_shape[1]
        annos[2] = target_shape[0]
        
        # Apply correction to coords
        for i in range(n_entries):
            
            if H > target_shape[0]:
                annos[4+5*i] = annos[4+5*i] / H
                annos[6+5*i] = (target_shape[0] / H) * annos[6+5*i]
        
            else:
                h_buff = int((target_shape[0] - H) / 2)
                annos[4+5*i] = annos[4+5*i] + h_buff
                annos[4+5*i] = annos[4+5*i] / target_shape[0]
                
            if W > target_shape[1]:
                annos[3+5*i] = annos[3+5*i] / W
                annos[5+5*i] = (target_shape[1] / W) * annos[5+5*i]
        
            else:
                w_buff = int((target_shape[1] - W) / 2)
                annos[3+5*i] = annos[3+5*i] + w_buff
                annos[3+5*i] = annos[3+5*i] / target_shape[1]
                
        # Convert to string and add to list
        annos = np.array(annos).astype(str)
        annos = ','.join(annos)
        entries.append(annos)
        
    outstr = '\n'.join(entries)
    print('\nWriting annotations to file...')
    with open(outpath,'w') as f:
        f.write(outstr)


def ExtractAnnotations(srcdir = 'D:\\DataStuff\\bms\\annotations',
                       outpath = 'D:\\DataStuff\\bms\\annotations.txt',
                       keep_labels = [],
                       buffer = 10,
                       k = 3):
    
    # Remove old annotations file
    if os.path.exists(outpath):
        print('\nRemoving old annotations file...')
        os.remove(outpath)
        
    # Find JSON filepaths
    print('\nFinding JSONs...')
    paths = RecSearch(srcdir, ext = 'json')
    entries = []
    
    # For each JSON path...
    print('\nReading JSONs...')
    i = 0
    for path in paths:

        # Get annotations for image
        annos = Annotations2(path = path, keep_labels = keep_labels, buffer = buffer, k = k)
        
        # Convert to string and add to list
        annos = np.array(annos).astype(str)
        annos = ','.join(annos)
        entries.append(annos)
        
        if i%50 == 0:
            print('Percent Complete: ', round(100*i/len(paths), 2))
        i += 1
    
    # Write to file
    outstr = '\n'.join(entries)
    print('\nWriting annotations to file...')
    with open(outpath,'w') as f:
        f.write(outstr)


# =============================================================================
# ANCHOR GENERATION
# =============================================================================

def GenerateAnchors(target_shape = (416,416), B = 3, n = 1000, outfile = 'D:\\DataStuff\\bms\\anchors.csv'):
    df = pd.read_csv('D:\\DataStuff\\bms\\annotations.csv')
    df['sf'] = target_shape[1] / df[['h','w']].max(axis = 1)
    df['normx'] = df.x / df.W
    df['normy'] = df.y / df.H
    df['normw'] = df.w / df.W
    df['normh'] = df.h / df.H
    X = np.array(df[['normx','normy','normw','normh']])
    model = KMeans(n_clusters = B).fit(X)
    df['cluster'] = model.labels_
    print('\nSaving anchors to file...')
    df.to_csv(outfile, index = False)
    return df