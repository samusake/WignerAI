#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from qutip import *

from skimage.transform import iradon, iradon_sart
from skimage.transform import radon, rescale
from scipy.special import erf

import numpy as np
from scipy.special import factorial
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as sk

import time
import json

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint

import autokeras as ak

import keras.backend as K
dtype='float16'
K.set_floatx(dtype)
#%%
s=20000#number of samples
nphi=42#45 #number of angleSteps

nxs=42
xmax=5
lxs=np.linspace(-xmax, xmax, nxs)
[xs, ys]=np.meshgrid(lxs,lxs)

phispace=np.linspace(0,180,nphi, endpoint=False)
[px, py]=np.meshgrid(lxs,phispace)

Nmax=20
#%%
'''
W=np.zeros((s,nxs,nxs))
P=np.zeros((s,nphi,nxs))
N=np.zeros((s,Nmax))
for i in range(0,s):        
    if i%100==0:
        k=i/s*100
        print("{0} %".format(k))
    N_current=np.random.randint(Nmax)+1
    rho=rand_dm(N_current)
    
    N[i,N_current-1]=1
    W[i]=wigner(rho,lxs,lxs)
    P[i]=radon(W[i],theta=phispace, circle=True)
np.save('data/P42_N', P)
np.save('data/W42_N', W)
np.save('data/N42_N', N)
'''
#%%
P=np.load('data/P42_N.npy')
W=np.load('data/W42_N.npy')
N=np.load('data/N42_N.npy')
#%%
inputV=np.zeros((s,nxs*nphi))
for i in range(0, len(P)):
    inputV[i]=P[i].flatten()
#%%
clf=ak.ImageClassifier()
clf.fit(P[i],N[i])
#%%
model=tf.keras.Sequential()
model.add(layers.Dense(nphi*nxs, activation='relu'))
#model.add(layers.Dense(3528, activation='relu'))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(200, activation='relu'))

model.add(layers.Dense(Nmax, activation='softmax'))
#%%
model.compile(optimizer=keras.optimizers.Adam(0.001),#,decay=0.0001),#tf.train.GradientDescentOptimizer(0.005),#optimizer=tf.train.AdamOptimizer(0.001),
    loss='binary_crossentropy', metrics=['categorical_accuracy'])

checkpoint = ModelCheckpoint('ai_checkpoint.h5', monitor='val_binary_crossentropy', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history=model.fit(inputV, N, epochs=1, batch_size=32, verbose=1, validation_split=0.1, callbacks=callbacks_list)
#%%
plt.figure(1)
plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])

plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.figure(2)
plt.semilogy(history.history['categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.show()

#%%
Nai=model.predict(inputV)
