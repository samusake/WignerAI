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

import keras.backend as K
dtype='float16'
K.set_floatx(dtype)
#%%
s=100000#number of samples
nphi=42#45 #number of angleSteps

nxs=42
xmax=5
lxs=np.linspace(-xmax, xmax, nxs)
[xs, ys]=np.meshgrid(lxs,lxs)

phispace=np.linspace(0,180,nphi, endpoint=False)
[px, py]=np.meshgrid(lxs,phispace)

#%%
'''
W=np.zeros((s,nxs,nxs))
P=np.zeros((s,nphi,nxs))
for i in range(0,s):        
    if i%100==0:
        k=i/s*100
        print("{0} %".format(k))
    
    rho=rand_dm(np.random.randint(5)+1)
    W[i]=wigner(rho,lxs,lxs)
    P[i]=radon(W[i],theta=phispace, circle=True)
np.save('data/P42', P)
np.save('data/W42', W)
'''
#%%
P=np.load('data/P42.npy')
W=np.load('data/W42.npy')
#%%
inputV=np.zeros((s,nxs*nphi))
outputV=np.zeros((s,nxs*nxs))
for i in range(0, len(P)):
    inputV[i]=P[i].flatten()
    outputV[i]=W[i].flatten()
#%%
model=tf.keras.Sequential()
model.add(layers.Dense(nphi*nxs, activation='relu'))
model.add(layers.Dense(3528, activation='relu'))
model.add(layers.Dense(1764, activation='relu'))
model.add(layers.Dense(1764, activation='relu'))

model.add(layers.Dense(nxs*nxs, activation='linear'))
#%%
model.compile(optimizer=keras.optimizers.Adam(0.001),#,decay=0.0001),#tf.train.GradientDescentOptimizer(0.005),#optimizer=tf.train.AdamOptimizer(0.001),
    loss='mean_squared_error')

checkpoint = ModelCheckpoint('ai_checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history=model.fit(inputV, outputV, epochs=300, batch_size=32, verbose=1, validation_split=0.1, callbacks=callbacks_list)
#%%
with open('models/randomWigner_Nsmaller5/model.json', 'w') as json_file:
    json_file.write(model.to_json())
model.save_weights('models/randomWigner_Nsmaller5/weights.h5')
with open('models/randomWigner_Nsmaller5/history.json', 'w') as json_file:
    json.dump(history.history, json_file)
#%%
plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])

plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#%%
Wai=model.predict(inputV)
inputV=None
outputV=None
Wai=np.concatenate(Wai)
Wai=np.reshape(Wai, (s,nxs,nxs))
#%%
fig, axs = plt.subplots(3, 6)

for i in range(0,6):
    axs[0,i].contourf(P[i],levels=15)
    axs[0,i].set_ylabel('phi')
    #axs[0,i].axis('equal')
   
    axs[1,i].contourf(xs,ys,W[i]/np.mean(W[i]),levels=15)
    axs[1,i].set_ylabel('Y')
    #axs[1,i].axis('equal')

    axs[2,i].contourf(xs,ys,Wai[i]/np.mean(W[i]),levels=15)
    axs[2,i].set_xlabel('X')
    axs[2,i].set_ylabel('Y')
    #axs[2,i].axis('equal')
axs[0,3].set_title('Probability distributions')
axs[1,3].set_title('Theoretical wigner distribution')
axs[2,3].set_title('AI prediction of wigner distribution')

plt.show()