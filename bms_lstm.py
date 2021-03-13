# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 22:29:12 2021

@author: Not Your Computer
"""

import os; import joblib; import numpy as np
import pandas as pd; import random; import time
import pickle; import sys; import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk; from random import shuffle
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout, Input, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.text import one_hot, text_to_word_sequence, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from wordcloud import WordCloud, STOPWORDS

from bms_utils import Invert


def LoadImg(path):
    img = plt.imread(path)
    img = Invert(img)
    return img


def XY(path):
    X = LoadImg(path)
    X = np.reshape(X, (1, X.shape[0], X.shape[1], 1))
    return X, Y


train_path = r"D:\DataStuff\bms\train_labels.csv"
df = pd.read_csv(train_path)


n_epochs = 100

# ASSEMBLE LSTM MODEL
model = Sequential()
model.add(LSTM(32, input_shape = np.shape(X)[1:], return_sequences = True))
model.add(BatchNormalization())
model.add(Dropout(0.20))
model.add(LSTM(32))
model.add(BatchNormalization())
model.add(Dropout(0.20))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.20))
model.add(Dense(Y.shape[-1], activation = 'relu'))

opt_type = keras.optimizers.Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0)
model.compile(loss = 'mean_squared_error', optimizer = opt_type, metrics = ['accuracy'])
print(model.summary())

# TRAIN THE MODEL
history = model.fit(X, Y, batch_size = 128, epochs = n_epochs)

# PLOT TRAINING HISTORY
if n_epochs > 1:
    plt.plot(history.history['accuracy'], label = 'train acc')
    plt.title('LSTM Training History', fontsize = 16)
    plt.ylabel('Accuracy (%)', fontsize = 12)
    plt.xlabel('Epoch', fontsize = 12)
    plt.legend(loc = "upper left")
    plt.show()
    

X, y = XY(path)