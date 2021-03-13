# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 16:32:36 2021

@author: Not Your Computer
"""

import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import tensorflow as tf
from tensorflow.keras import backend as K

from bms_utils import Labels, LoadImg2, ExtractAnnotations
from bms_modeling import YOLO, yolo_det_loss
from bms_image_processing import AlphabetSoup


class AlphaGen(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, params, n_imgs = 100):
        self.params = params
        self.n_imgs = n_imgs
        self.batch_size = params['batch_size']
        self.target_shape = params['target_shape']
        self.keep_labels = params['keep_labels']
        self.S = params['S']
        self.B = params['B']
        self.C = params['C']
        self.size_thresh = params['size_thresh']
        
    def __len__(self):
        return self.n_imgs // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        X, Y = AlphabetSoup(target_shape = self.target_shape, size_thresh = self.size_thresh, S = self.S, B = self.B, C = self.C, buffer = 120)
        X = np.reshape(X, (self.batch_size, self.target_shape[0], self.target_shape[1], 1))
        return X, Y


class DataGen(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, params, entries):
        self.entries = entries
        self.params = params
        self.batch_size = params['batch_size']
        self.target_shape = params['target_shape']
        self.keep_labels = params['keep_labels']
        self.S = params['S']
        self.B = params['B']
        self.C = params['C']
        self.size_thresh = params['size_thresh']
        self.scale_images = params['scale_images']
        
    def __len__(self):
        return len(self.entries) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        entry = self.entries[idx]
        path = entry.split(',')[0]
        
        x = LoadImg2(path, target_shape = self.target_shape, scale_images = self.scale_images)
        x = np.reshape(x, (self.batch_size, self.target_shape[0], self.target_shape[1], 1))
        y = YOLOBrick(entry, S = self.S, target_shape = self.target_shape, size_thresh = self.size_thresh, scale_images = self.scale_images)
        return x, y


# Make a SxSx(5B+C) YOLO output brick
# Input: annotation (str) - a line of converted annotation text
# Output: data brick (np.array)
def YOLOBrick(entry, S = (16,16), target_shape = (416,416), B = 1, size_thresh = 100, scale_images = True):
    entry = entry.split(',')
    path = entry[0]
    img_w = int(entry[1])
    img_h = int(entry[2])
    brick = np.zeros((S[0],S[1],5*B))
    n_entries = int(len(entry[3:]) / 9)
    grid_h = target_shape[0] / S[0]
    grid_w = target_shape[1] / S[1]
    
    # For each annotation entry...
    for i in range(n_entries):
        
        # Read entry data
        x,y,w,h,label,pxmin,pymin,pxmax,pymax = entry[3+9*i:3+9*(i+1)]
        x = float(x); y = float(y)
        w = float(w); h = float(h)
        pxmin = int(pxmin); pymin = int(pymin)
        pxmax = int(pxmax); pymax = int(pymax)

        x = x - pxmin
        y = y - pymin

        # Transform coordinates
        if scale_images:
            sf = target_shape[np.argmax([pymax-pymin, pxmax-pxmin])] / max(pymax-pymin, pxmax-pxmin)
        else:
            sf = 1
        
        x = sf * x
        y = sf * y

        h = sf * h
        w = sf * w

        # Identify responsible grid cell
        cell = (int(np.floor(y/target_shape[0] * S[0])), int(np.floor(x/target_shape[1] * S[1])))
        
        # Normalize X,Y pixel coords wrt the grid cell
        x = (x - cell[1]*grid_w) / grid_w
        y = (y - cell[0]*grid_h) / grid_h

        # Normalize height & width
        w = min(float(w) / size_thresh, 1)
        h = min(float(h) / size_thresh, 1)
        
        anno = [x,y,w,h,1]
        
        # Add entry data to brick
        brick[cell[0],cell[1],:] = anno
    
    return brick


# GENERATE TRAINING ITEMS FOR YOLO MODEL
def XY(entry, target_shape = (416,416), S = (16,16), scale_images = True):
    x = LoadImg2(entry.split(',')[0], target_shape = target_shape, scale_images = scale_images)
    x = np.reshape(x, (target_shape[0], target_shape[1], 1))
    y = YOLOBrick(entry, S = S, target_shape = target_shape, scale_images = scale_images)
    return x, y


def PlotXY(img, brick, conf_thresh = 0, size_thresh = 100):
    S = brick.shape[0:2]
    img_h, img_w, img_d = img.shape
    grid_h = img_h / S[0]
    grid_w = img_w / S[1]
    
    # Plot image
    plt.imshow(img[:,:,0])
    
    for i in range(S[0]):
        
        for j in range(S[1]):
            
            # Add boxes if populated
            if np.sum(brick[i,j,:]) > 0:
                ax = plt.gca()
                x,y,w,h,conf = brick[i,j,:5]
                w *= size_thresh
                h *= size_thresh
                x = x*grid_w + j*grid_w
                y = y*grid_h + i*grid_h
                
                if conf > 0:
                    if conf >= conf_thresh:
                        rect = Rectangle((x-int(w/2), y-int(h/2)), w, h, linewidth = 1, edgecolor = 'r', facecolor = 'none')
                        ax.add_patch(rect)
                else:     
                    rect = Rectangle((x-int(w/2), y-int(h/2)), w, h, linewidth = 1, edgecolor = 'r', facecolor = 'none')
                    ax.add_patch(rect)
    plt.show()


#%%
if __name__ == '__main__':
    
    params = {'labels':Labels()[1:],
              'keep_labels':['b','br','c','cl','f','h','i','n','nh','nh2','o','oh','p','s','sh','si'],
              'target_shape':(640,640),
              'scale_images':True,
              'size_thresh':60,
              'S':(11,11),
              'B':1,
              'C':0,
              'batch_size':1,
              'epochs':1,
              'lr':1e-5,
              'dr':0.0,
              'n_train':1130}
    
    K.clear_session()
    
    
    #%%
    # Generate annotation data from files
#    ExtractAnnotations(keep_labels = params['keep_labels'])
    
    
    #%%
    # Initialize YOLO model 
    print('\nCompiling model...')
    model = YOLO(input_shape = (params['target_shape'][0],params['target_shape'][1],1),
                 S = params['S'],
                 B = params['B'],
                 dr = params['dr'])
    
    
#%%
    # Compile model
    opt = tf.keras.optimizers.Adam(lr = params['lr'], beta_1 = 0.9, beta_2 = 0.999, decay = 0.01)
#    opt = tf.keras.optimizers.RMSprop(learning_rate = params['lr'])
#    opt = tf.keras.optimizers.Adagrad(learning_rate = params['lr'])
#    opt = tf.keras.optimizers.Adadelta(learning_rate = params['lr'])
    model.compile(loss = yolo_det_loss, optimizer = opt)
    print(model.summary())
   
    
#%%
    # Read annotations.txt file
    print('\nReading annotations file...')
    with open(r"D:\DataStuff\bms\annotations.txt",'r') as f:
        entries = f.read()
    entries = entries.split('\n')
    print('{} entries found.'.format(len(entries)))
    
    
#%%
    print('\nPre-Training model...')
    train_gen = AlphaGen(params, n_imgs = 3000)
    valid_gen = AlphaGen(params, n_imgs = 1000)
    
    model.fit(train_gen,
              validation_data = valid_gen,
              epochs = params['epochs'])

#%%
    print('\nTraining model...')
    train_gen = DataGen(params, entries[0:params['n_train']])
    valid_gen = DataGen(params, entries[params['n_train']:])
    
    model.fit(train_gen,
              validation_data = valid_gen,
              epochs = params['epochs'])


#%%
    obj_conf = 0.01
    for i in range(10):
        # Make predictions
        randent = random.choice(entries)
        if os.path.exists(randent.split(',')[0]):
            x_test, y_test = XY(randent, target_shape = params['target_shape'], S = params['S'], scale_images = params['scale_images'])
            p = model.predict(np.reshape(x_test, (1,params['target_shape'][0],params['target_shape'][1],1)))
            PlotXY(x_test, p[0], conf_thresh = obj_conf, size_thresh = params['size_thresh'])
            