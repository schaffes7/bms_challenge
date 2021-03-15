# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 21:30:38 2021

@author: Not Your Computer
"""
from skimage.transform import rescale

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K

from bms_image_processing import AlphabetSoup
from bms_modeling import UNet
from bms_utils import Labels, LoadImg2, ExtractAnnotations, Invert


#%%

class AlphaGen(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, params, n_imgs = 100):
        self.params = params
        self.n_imgs = n_imgs
        self.batch_size = params['batch_size']
        self.target_shape = params['target_shape']
        self.keep_labels = params['keep_labels']
        self.size_thresh = params['size_thresh']
        self.preprocess = params['preprocess']
        
    def __len__(self):
        return self.n_imgs // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        X, Y = AlphabetSoup(labels = self.keep_labels, output_type = 'unet', target_shape = self.target_shape, size_thresh = self.size_thresh, buffer = 120, preprocess = self.preprocess)
        X = np.reshape(X, (self.batch_size, self.target_shape[0], self.target_shape[1], 1))
        Y = np.reshape(Y, (Y.shape[0], Y.shape[1], 1))
        return X, Y


class DataGen(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, params, entries):
        self.entries = entries
        self.params = params
        self.batch_size = params['batch_size']
        self.target_shape = params['target_shape']
        self.keep_labels = params['keep_labels']
        self.size_thresh = params['size_thresh']
        self.scale_images = params['scale_images']
        self.preprocess = params['preprocess']
        
    def __len__(self):
        return len(self.entries) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        entry = self.entries[idx]
        entry = entry.split(',')
        n_entries = int(len(entry[3:]) / 9)
        path,W,H = entry[0:3]
        W = int(W)
        H = int(H)
        sf = self.target_shape[np.argmax([H,W])] / max(H,W)
        
        X = LoadImg2(path, target_shape = self.target_shape, scale_images = self.scale_images, preprocess = self.preprocess)
        X = np.reshape(X, (self.batch_size, self.target_shape[0], self.target_shape[1], 1))
        
        Y = np.zeros(target_shape)
        for i in range(n_entries):
            x,y,w,h = entry[3+9*i:3+9*i+4]
            x = sf * float(x)
            y = sf * float(y)
            w = sf * float(w)
            h = sf * float(h)
            Y[int(y-h/2):int(y+h/2),int(x-w/2):int(x+w/2)] = 1
        
        return X, Y


def XY(entry, target_shape = (640,640), scale_images = True, preprocess = True):
    entry = entry.split(',')
    n_entries = int(len(entry[3:]) / 9)
    path,W,H = entry[0:3]
    W = int(W)
    H = int(H)
    sf = target_shape[np.argmax([H,W])] / max(H,W)
    
    X = LoadImg2(path, target_shape = target_shape, scale_images = scale_images, preprocess = preprocess)
    X = np.reshape(X, (1, target_shape[0], target_shape[1], 1))
    
    Y = np.zeros(target_shape)
    for i in range(n_entries):
        x,y,w,h = entry[3+9*i:3+9*i+4]
        x = sf * float(x)
        y = sf * float(y)
        w = sf * float(w)
        h = sf * float(h)
        Y[int(y-h/2):int(y+h/2),int(x-w/2):int(x+w/2)] = 1
    
    return X,Y


def LoadMask(path, target_shape = (640,640)):
    mask = plt.imread(path)
    h,w,d = np.shape(mask)
    sf = target_shape[np.argmax([h,w])] / max(h,w)
    mask = rescale(mask, sf)
    out_img = np.zeros((target_shape[0],target_shape[1],3))
    h,w,d = np.shape(mask)
    out_img[0:h,0:w,:] = mask
    return out_img


#%%
params = {'labels':Labels()[1:],
          'keep_labels':['b','br','c','cl','f','h','i','n','nh','nh2','o','oh','p','s','sh','si'],
          'target_shape':(704,704),
          'scale_images':True,
          'preprocess':True,
          'min_size':10,
          'size_thresh':64,
          'batch_size':1,
          'epochs':1,
          'lr':1e-4,
          'dr':0.0,
          'n_train':1000}

K.clear_session()


#%%
target_shape = params['target_shape']
model = UNet(img_shape = (target_shape[0], target_shape[1], 1), final_act = 'sigmoid')
opt = tf.keras.optimizers.Adam(lr = params['lr'], beta_1 = 0.9, beta_2 = 0.999, decay = 0.01)
model.compile(loss = 'mean_absolute_error', optimizer = opt)
print(model.summary())


#%%
# Read annotations.txt file
print('\nReading annotations file...')
with open(r"D:\DataStuff\bms\annotations.txt",'r') as f:
    entries = f.read()
entries = entries.split('\n')


#%%
print('\nPre-Training model...')
train_gen = AlphaGen(params, n_imgs = 500)
valid_gen = AlphaGen(params, n_imgs = 500)

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
X,Y = XY(entries[random.randint(0,100)], target_shape = target_shape)
p = model.predict(X)
plt.imshow(X[0,:,:,0])
plt.show()
plt.imshow(Y[:,:])
plt.show()
plt.imshow(p[0,:,:,0])
plt.show()