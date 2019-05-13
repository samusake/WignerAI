#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 19:34:57 2019

@author: herbert

"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from keras.utils import to_categorical

#%%
class simpleDeepNN:
    def __init__(self, nxs, nphi):
        self.model=tf.keras.Sequential()
        
        self.model.add(layers.Dense(nphi*nxs, activation='linear'))
        # Add another:
        self.model.add(layers.Dense(256, activation='linear'))
        self.model.add(layers.Dense(256, activation='linear'))
        self.model.add(layers.Dense(256, activation='linear'))
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
        self.model.add(layers.Conv2D(32, kernel_size=(5,5), strides=(1,1),
                                       activation='linear', input_shape=(nphi,nxs)))
        self.model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        self.model.add(layers.Conv2D(64, (5, 5), activation='linear'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        #self.model.add(layers.Flatten())
        #self.model.add(layers.Dense(1000, activation='linear'))
        #self.model.add(layers.Dense(nxs*, activation='linear'))
        self.model.add(layers.Conv2D(1, kernel_size(nxs,nxs),strides=(1,1),
                                       activation='linear',))