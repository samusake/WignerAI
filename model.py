#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import model_from_json

#%%
class simpleDeepNN:
    def __init__(self, nxs, nphi):
        self.model=tf.keras.Sequential()
        
        self.model.add(layers.Dense(nphi*nxs, activation='relu'))
        # Add another:
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(256, activation='relu'))
        # Add a softmax layer with 10 output units:
        self.model.add(layers.Dense(nxs*nxs, activation='linear'))
#%%
class simpleConv:
    def __init__(self, nxs, nphi):
        self.model = tf.keras.Sequential()
        self.model.add(layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(nphi,nxs,1), strides=1))
        self.model.add(layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', strides=1))
        
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dropout(0.25))
        
        self.model.add(layers.Dense(254, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        
        self.model.add(layers.Dense(nxs*nxs, activation='linear'))
#%%
class simpleDeepNN2:
    def __init__(self, nxs, nphi):
        self.model=tf.keras.Sequential()
        
        self.model.add(layers.Dense(nphi*nxs, activation='relu'))
        # Add another:
        self.model.add(layers.Dense(512, activation='relu'))
        self.model.add(layers.Dense(512, activation='relu'))
        self.model.add(layers.Dense(512, activation='relu'))
        # Add a softmax layer with 10 output units:
        self.model.add(layers.Dense(nxs*nxs, activation='linear'))
#%%
class smallDeepNN1:
    def __init__(self, nxs, nphi):
        self.model=tf.keras.Sequential()
        
        self.model.add(layers.Dense(nphi*nxs, activation='relu'))
        # Add another:
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(128, activation='relu'))
        # Add a softmax layer with 10 output units:
        self.model.add(layers.Dense(nxs*nxs, activation='linear'))

#%%
class DensityDeepNN:
    def __init__(self, N, nxs, nphi):
        self.model=tf.keras.Sequential()
        
        self.model.add(layers.Dense(nphi*nxs, activation='relu'))
        # Add another:
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dense(128, activation='relu'))
        # Add a softmax layer with 10 output units:
        self.model.add(layers.Dense(2*N*N, activation='linear'))













