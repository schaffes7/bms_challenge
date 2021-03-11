# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:44:21 2021

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
from tensorflow.keras.layers import Conv2D, Input, Reshape, Dense, Flatten, Add, ZeroPadding2D, Dropout, Concatenate, MaxPooling2D, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model



def YOLO(input_shape = (416,416,1), S = (16,16), B = 1, C = 0, dr = 0.10):

    i = Input(shape = input_shape)
    
    x = Conv2D(8, (1,1), padding = 'same')(i)
    x = Conv2D(16, (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha = 0.3)(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    
    x = Conv2D(16, (1,1), padding = 'same')(x)
    x = Conv2D(32, (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha = 0.3)(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    
    x = Conv2D(32, (1,1), padding = 'same')(x)
    x = Conv2D(64, (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha = 0.3)(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    
    x = Conv2D(64, (1,1), padding = 'same')(x)
    x = Conv2D(128, (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha = 0.3)(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    
    x = Conv2D(128, (1,1), padding = 'same')(x)
    x = Conv2D(256, (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha = 0.3)(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    
    x = Conv2D(256, (1,1), padding = 'same')(x)
    x = Conv2D(512, (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha = 0.3)(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    
    x = Flatten()(x)
    
    # Fully connected
    x = Dense(4096, activation = 'relu')(x)
    x = Dropout(dr)(x)
    
    # Fully connected
    x = Dense(2048, activation = 'relu')(x)
    x = Dropout(dr)(x)
    
    # Fully connected
    x = Dense(S[0] * S[1] * (5*B + C), activation = 'sigmoid')(x)
    x = Reshape((S[0], S[1], (5*B + C)))(x)
    
    model = Model(i, x)
    return model



def yolo_det_loss(y, p):
    
    # X/Y values
    y_xy = y[...,0:2]
    p_xy = p[...,0:2]
    
    # Width/Height values
    y_wh = y[...,2:4]
    p_wh = p[...,2:4]
    
    # Object confidence
    y_conf = y[...,4]
    p_conf = p[...,4]
   
    # Intersection over Union
    intersect_wh = K.maximum(K.zeros_like(p_wh), (p_wh + y_wh)/2 - K.square(p_xy - y_xy))
    I = intersect_wh[...,0] * intersect_wh[...,1]
    true_area = y_wh[...,0] * y_wh[...,1]
    pred_area = p_wh[...,0] * p_wh[...,1]
    U = pred_area + true_area - I
    iou = I / U
    
    # Calculate individual errors
    e_xy = K.sum(K.sum(K.square(y_xy - p_xy), axis = -1) * y_conf, axis = -1)
    e_wh = K.sum(K.sum(K.square(K.sqrt(y_wh) - K.sqrt(p_wh)), axis = -1) * y_conf, axis = -1)
    e_conf = K.sum(K.square(y_conf * iou - p_conf), axis = -1)
    
    # Sum all errors
    e = e_xy + e_wh + e_conf
    return e


def yolo_cls_loss(y, p):
    
    # Class Values
    y_class = y[...,5:]
    p_class = p[...,5:] 
    
    # X/Y values
    y_xy = y[...,0:2]
    p_xy = p[...,0:2]
    
    # Width/Height values
    y_wh = y[...,2:4]
    p_wh = p[...,2:4]
    
    # Object confidence
    y_conf = y[...,4]
    p_conf = p[...,4]
   
    # Intersection over Union
    intersect_wh = K.maximum(K.zeros_like(p_wh), (p_wh + y_wh)/2 - K.square(p_xy - y_xy))
    I = intersect_wh[...,0] * intersect_wh[...,1]
    true_area = y_wh[...,0] * y_wh[...,1]
    pred_area = p_wh[...,0] * p_wh[...,1]
    U = pred_area + true_area - I
    iou = I / U
    
    # Calculate individual errors
    e_xy = K.sum(K.sum(K.square(y_xy - p_xy), axis = -1) * y_conf, axis = -1)
    e_wh = K.sum(K.sum(K.square(K.sqrt(y_wh) - K.sqrt(p_wh)), axis = -1) * y_conf, axis = -1)
    e_conf = K.sum(K.square(y_conf * iou - p_conf), axis = -1)
    e_clss = K.sum(K.square(y_class - p_class), axis = -1)
    
    # Sum all errors
    e = e_xy + e_wh + 10*e_conf + 0.5*e_clss
    return e

