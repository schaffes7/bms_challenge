# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 16:32:36 2021

@author: Not Your Computer
"""

import numpy as np
import pandas as pd
import os
import sys
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import tensorflow as tf
from tensorflow.keras import backend as K

from bms_utils import Labels, LoadImg, ConvertAnnotations, PixelBounds
from bms_modeling import YOLO, yolo_cls_loss


# Make a SxSx(5B+C) YOLO output brick
# Input: annotation (str) - a line of converted annotation text
# Output: data brick (np.array)
def YOLOBrick(entry, S = (16,16), target_shape = (416,416), B = 1, C = 10):
    entry = entry.split(',')
    path = entry[0]
    W = int(entry[1])
    H = int(entry[2])
    brick = np.zeros((S[0],S[1],(5*B+C)))
    n_entries = int(len(entry[3:]) / 5)
    
    # For each annotation entry...
    for i in range(n_entries):
        x,y,w,h,lab = entry[3+5*i:3+5*(i+1)]
        x = float(x)
        y = float(y)
        
        # Identify responsible grid cell
        cell = (int(y*(S[0]-1)), int(x*(S[1]-1)))
        
        # Normalize X,Y pixel coords wrt the grid cell
        x = ((x * W) - (W / S[1]) * cell[1]) / (W / S[1])
        y = ((y * H) - (H / S[0]) * cell[0]) / (H / S[0])
        
        # Normalize height & width
#        w = float(w) / W
#        h = float(h) / H
        w = min(float(w) / 45, 1)
        h = min(float(h) / 45, 1)
        
        anno = [x,y,w,h,1]
        
        if C > 0:
            lab = int(lab)
            class_onehot = list(np.zeros((C), dtype = int))
            class_onehot[int(lab)] = 1
            anno = anno + class_onehot
                
        # Add entry data to brick
        brick[cell[0],cell[1],:] = anno
    
    return brick


# GENERATE TRAINING ITEMS FOR YOLO MODEL
def XY(entry, target_shape = (416,416), S = (16,16), C = 10, add_shift = False):
    x = LoadImg(entry.split(',')[0], target_shape = target_shape, invert = True)
    
    if add_shift:
        blank = np.zeros(target_shape)
        bounds = PixelBounds(x)
        chip = x[bounds[1]:bounds[3], bounds[0]:bounds[2]]
        h,w = np.shape(chip)
        h_off = random.randint(0,target_shape[0]-h-1)
        w_off = random.randint(0,target_shape[1]-w-1)
        blank[h_off:h_off+h, w_off:w_off+w] = chip
        x = blank
    else:
        h_off = 0
        w_off = 0
    
    x = np.reshape(x, (target_shape[0], target_shape[1], 1))
    y = YOLOBrick(entry, S = S, C = C, target_shape = target_shape)
    
    return x, y


def PlotYOLO(p, path, target_shape = (416,416), S = (16,16), labels = [], obj_conf = 0.50, cls_conf = 0.50):
    df = []
    colors = []
    for color in itertools.permutations([0.9,0,0.9,0.9,0.6,0.6,0.6,0.4,0,0], 3):
        if color != (0,0,0):
            colors.append(color)
    colors = np.unique(colors, axis = 0)
    if len(labels) == 0:
        labels = Labels()[1:]
    for i in range(p.shape[1]):
        for j in range(p.shape[2]):
            row = p[0,i,j,:]
            row = list(row) + [i,j]
            df.append(row)
    cols = ['x','y','width','height','obj'] + labels + ['R','C']
    df = pd.DataFrame(df, columns = cols)
    
    img = LoadImg(path, target_shape = target_shape)
    plt.figure(figsize = (8,8))
    plt.imshow(img)
    
    for idx,row in df.iterrows():
        if row['obj'] >= df.obj.quantile(0.95):
            ax = plt.gca()
#            w = int(row['width'] * target_shape[1])
#            h = int(row['height'] * target_shape[0])
            w = int(row['width'] * 45)
            h = int(row['height'] * 45)
            x = int(row['x'] * (target_shape[1] / S[1])) + int(row['C'] * (target_shape[1] / S[1]))
            y = int(row['y'] * (target_shape[0] / S[0])) + int(row['R'] * (target_shape[0] / S[0]))
            rect = Rectangle((x-int(w/2), y-int(h/2)), w, h, linewidth = 1, edgecolor = colors[labels.index(row[labels][row[labels] == np.max(row[labels])].index[0])], facecolor = 'none')
            ax.add_patch(rect)
    plt.show()
    return df


#%%
if __name__ == '__main__':
    
    all_labels = Labels()[1:]
    keep_labels = ['b','br','c','cl','f','h','i','n','nh','nh2','o','oh','p','s','sh','si']
    target_shape = (640,640)
    S = (12,12)
    C = len(keep_labels)
    B = 1
    batch_size = 1
    epochs = 1
    lr = 1e-5
    dr = 0.0
    n_valid = 0
    
    K.clear_session()
    
    #%%
    # Generate annotation data from files
#    if not os.path.exists(r"D:\DataStuff\bms\annotations.txt"):
    ConvertAnnotations(target_shape = target_shape, keep_labels = keep_labels)
    
    #%%
    # Initialize YOLO model
    print('\nCompiling model...')
    model = YOLO(input_shape = (target_shape[0],target_shape[1],1), S = S, B = B, C = C, dr = dr)
    
    
    # Compile model
#    opt = tf.keras.optimizers.Adam(lr = lr, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01)
    opt = tf.keras.optimizers.RMSprop(learning_rate = lr)
    model.compile(loss = yolo_cls_loss, optimizer = opt)
    print(model.summary())    
    
    
    # Read annotations.txt file
    print('\nReading annotations file...')
    with open(r"D:\DataStuff\bms\annotations.txt",'r') as f:
        entries = f.read()
    entries = entries.split('\n')
    
    
    # For each annotation entry...
    print('\nFormatting data...')
    i = 0; x_train = []; y_train = []
    for entry in entries:
        if os.path.exists(entry.split(',')[0]):
            # Create input / output training items
            x, y = XY(entry, target_shape = target_shape, S = S, C = C, add_shift = False)
            x_train.append(x)
            y_train.append(y)
        if i % 50 == 0:
            print('Percent Complete: ', round(100 * (i / len(entries)), 2))
        i += 1
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    x_test, y_test = XY(entries[0], target_shape = target_shape, S = S, C = C)
    
    
#    # Visuallize training on single image
#    for i in range(len(entries)):
#        print(i)
#        model.fit(x_train[i:i+1],y_train[i:i+1])
#        p = model.predict(np.reshape(x_test, (1,target_shape[0],target_shape[1],1)))
#        df = PlotDetYOLO(p, entries[0].split(',')[0],
#                                  target_shape = target_shape,
#                                  S = S,
#                                  obj_conf = 0.50)
    
    
    # Fit model on training data
    model.fit(x_train[:-n_valid], y_train[:-n_valid],
              validation_data = (x_train[len(x_train)-n_valid:],y_train[len(y_train)-n_valid:]),
              batch_size = batch_size,
              epochs = epochs)
#%%
    obj_conf = 0.35
    for i in range(15):
        # Make predictions
        randent = random.choice(entries)
        if os.path.exists(randent.split(',')[0]):
            x_test, y_test = XY(randent, target_shape = target_shape, S = S, C = C)
            p = model.predict(np.reshape(x_test, (1,target_shape[0],target_shape[1],1)))
            df = PlotYOLO(p, randent.split(',')[0],
                          target_shape = target_shape,
                          S = S,
                          labels = keep_labels,
                          obj_conf = obj_conf,
                          cls_conf = 0.10)