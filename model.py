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
    def train(self, eps, inputV, outputV, verb=0):
        self.model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='mean_squared_error',
              metrics=['accuracy', 'mean_squared_error'])
        
    def evaluate(self, testIn, testOut):
        return(self.model.evaluate(testIn, testOut))
    def predict(self, testIn):
        return(self.model.predict(testIn))
#%%
class simpleConv:
    def __init__(self, nxs, nphi):
        self.model=tf.keras.Sequential()
        print('help0')
        self.model.add(layers.Conv2D(32, kernel_size=(5,5), strides=(1,1),activation='linear', input_shape=(nphi,nxs)))
        print('help1')
        self.model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        self.model.add(layers.Conv2D(64, (5, 5), activation='linear'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        #self.model.add(layers.Flatten())
        #self.model.add(layers.Dense(1000, activation='linear'))
        #self.model.add(layers.Dense(nxs*, activation='linear'))
        self.model.add(layers.Conv2D(1, kernel_size(nxs,nxs),strides=(1,1),
                                       activation='linear',))
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
    def train(self, eps, inputV, outputV, verb=0):
        self.model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='mean_squared_error',
              metrics=['accuracy', 'mean_squared_error'])
        
    def evaluate(self, testIn, testOut):
        return(self.model.evaluate(testIn, testOut))
    def predict(self, testIn):
        return(self.model.predict(testIn))
#%%
class smallDeepNN:
    def __init__(self, nxs, nphi):
        self.model=tf.keras.Sequential()
        
        self.model.add(layers.Dense(nphi*nxs, activation='relu'))
        # Add another:
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dense(128, activation='relu'))
        # Add a softmax layer with 10 output units:
        self.model.add(layers.Dense(nxs*nxs, activation='linear'))
    def train(self, eps, inputV, outputV, verb=0):
        self.model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='mean_squared_error',
              metrics=['accuracy', 'mean_squared_error'])
        
    def evaluate(self, testIn, testOut):
        return(self.model.evaluate(testIn, testOut))
    def predict(self, testIn):
        return(self.model.predict(testIn))
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
    def train(self, eps, inputV, outputV, verb=0):
        self.model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='mean_squared_error',
              metrics=['accuracy', 'mean_squared_error'])
        
    def evaluate(self, testIn, testOut):
        return(self.model.evaluate(testIn, testOut))
    def predict(self, testIn):
        return(self.model.predict(testIn))














