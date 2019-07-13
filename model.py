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
class smallDeepNN:
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
        self.model.add(layers.Dense(400, activation='relu'))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(400, activation='relu'))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(400, activation='relu'))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(400, activation='relu'))
        self.model.add(layers.Dropout(0.2))
        # Add a softmax layer with 10 output units:
        self.model.add(layers.Dense(2*N*N, activation='linear'))
#%%
class smallDensityDeepNN:
    def __init__(self, N, nxs, nphi):
        self.model=tf.keras.Sequential()
        
        self.model.add(layers.Dense(nphi*nxs, activation='relu'))
        # Add another:
        self.model.add(layers.Dense(50, activation='relu'))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(10, activation='relu'))
        self.model.add(layers.Dropout(0.2))
        # Add a softmax layer with 10 output units:
        self.model.add(layers.Dense(2*N*N, activation='linear'))

#%%
class pureDataNN:
    def __init__(self, nxs, Ndata):
        self.model=tf.keras.Sequential()
        
        self.model.add(layers.Dense(Ndata, activation='relu'))
        # Add another:
        self.model.add(layers.Dense(10000, activation='relu'))
        self.model.add(layers.Dense(5000, activation='relu'))
        self.model.add(layers.Dense(2500, activation='relu'))
        self.model.add(layers.Dense(256, activation='relu'))
        # Add a softmax layer with 10 output units:
        self.model.add(layers.Dense(nxs*nxs, activation='linear'))

#%%
class pureDataNN_P:
    def __init__(self, nphi, nxs, Ndata):
        self.model=tf.keras.Sequential()
        
        self.model.add(layers.Dense(Ndata, activation='relu'))
        # Add another:
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(256, activation='relu'))
        # Add a softmax layer with 10 output units:
        self.model.add(layers.Dense(nphi*nxs, activation='linear'))

#%%
class imageDeepNN:
    def __init__(self, nxs, nxs_P, nphi):
        self.model=tf.keras.Sequential()
        
        self.model.add(layers.Dense(nphi*nxs_p, activation='relu'))
        # Add another:
        self.model.add(layers.Dense(2048, activation='relu'))
        self.model.add(layers.Dense(2048, activation='relu'))
        # Add a softmax layer with 10 output units:
        self.model.add(layers.Dense(nxs*nxs, activation='linear'))



