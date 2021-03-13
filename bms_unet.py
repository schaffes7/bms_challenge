# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 21:30:38 2021

@author: Not Your Computer
"""
from skimage.transform import resize, rescale

from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import Image, ImageOps, ImageEnhance
from tensorflow import keras
import numpy as np
import pandas as pd
import os
import cv2
from tensorflow.keras import layers
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.transform import resize
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import concatenate, Dense, Activation, Dropout, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, UpSampling2D, Input, ZeroPadding2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import MobileNet
from sklearn.model_selection import train_test_split
import imageio

from bms_utils import Labels, LoadImg2, ExtractAnnotations, Invert, PixelBounds


#%%
def CustomUNet(img_shape = (640,640,1), net_layers = [32,64,128], act = 'relu', pool_size = (2,2), final_pool = (1,1), dropout = 0.50, final_act = 'softmax'):
    # INPUT
    img_input = Input(shape = img_shape)
    
    # ENCODING LAYERS
    fwd_lyrs = []
    i = 0
    for lyrs in net_layers:
        print('Encode: ', lyrs)
        if i == 0: x = Conv2D(lyrs, (3,3), activation = act, padding = 'same')(img_input)
        else: x = Conv2D(lyrs, (3,3), activation = act, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        print('Conv: ', list(x.get_shape()))
        fwd_lyrs.append(x)
        x = AveragePooling2D(pool_size)(x)
        print('Pool: ', list(x.get_shape()))
        i += 1
    
    # MIDDLE LAYER
    act = 'relu'
    x = Conv2D(1, (3,3), activation = act, padding = 'same')(x)
    print('Conv: ', list(x.get_shape()))
    net_layers.reverse()
    fwd_lyrs.reverse()
    
    # DECODING LAYERS
    i = 0
    for lyrs in net_layers:
        print('Decode: ', lyrs)
        x = Conv2D(lyrs, (3,3), activation = act, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        print('Conv: ', list(x.get_shape()))
        x = concatenate([UpSampling2D(pool_size)(x), fwd_lyrs[i]])
        print('Up2D: ', list(x.get_shape()))
        i += 1
    
    # OUTPUT
    out = Conv2D(1, final_pool, activation = final_act, padding = 'same')(x)
    print('Conv: ', list(out.get_shape()))
    model = Model(img_input, out)
    return model


def LoadMask(path, target_shape = (640,640)):
    mask = plt.imread(path)
    h,w,d = np.shape(mask)
    sf = target_shape[np.argmax([h,w])] / max(h,w)
    mask = rescale(mask, sf)
    out_img = np.zeros((target_shape[0],target_shape[1],3))
    h,w,d = np.shape(mask)
    out_img[0:h,0:w,:] = mask
    return out_img


def UNetXY(entry, target_shape = (640,640)):
    path = entry.split(',')[0]
    x = LoadImg2(path, target_shape = target_shape)
    h,w = np.shape(x)
    x = np.reshape(x, (target_shape[0],target_shape[1],1))
    y_path = path.replace('img.png','label.png')
    y = LoadMask(y_path, target_shape = target_shape)
    return x,y


#%%
lr = 1e-5
n_train = 100
target_shape = (640,640)
model = CustomUNet(img_shape = (target_shape[0], target_shape[1], 1))
opt = tf.keras.optimizers.Adam(lr = lr, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01)
#    opt = tf.keras.optimizers.RMSprop(learning_rate = lr)
model.compile(loss = 'mean_squared_error', optimizer = opt)
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
        x, y = UNetXY(entry, target_shape = target_shape)
        x_train.append(x)
        y_train.append(y)
    if i % 50 == 0:
        print('Percent Complete: ', round(100 * (i / len(entries)), 2))
    i += 1
x_train = np.array(x_train)
y_train = np.array(y_train)
#%%

# Fit model on training data
model.fit(x_train[0:10], y_train[0:10],
          batch_size = 1,
          epochs = 1)

#%%
x_test, y_test = UNetXY(entries[0], target_shape = target_shape)
model.predict(x_test)