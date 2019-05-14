#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import factorial
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as sk

from numba import jit
import time
import pickle

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint

from genSample import *
from model import *
#%%

N=-1 #dimension of rho
s=50000 #number of samples
nphi=20#45 #number of angleSteps

nxs=40
xmax=5
lxs=np.linspace(-xmax, xmax, nxs)
[xs, ys]=np.meshgrid(lxs,lxs);

phispace=np.linspace(0,180,nphi)
[px, py]=np.meshgrid(lxs,phispace)

'''
P, W=generateDataset(N,s,nphi,lxs)
np.save('data/P50000_20_40', P)
np.save('data/W50000_20_40', W)
'''
#%%
P=np.load('data/P50000_20_40.npy')
W=np.load('data/W50000_20_40.npy')
#%%
fig=plt.figure(1)
ax=fig.add_subplot(111)

ax.contourf(px,py,P[3],levels=15)
ax.set_xlabel('u')
ax.set_ylabel('phi')

fig=plt.figure(2)
ax=fig.add_subplot(111)
ax.contourf(xs,ys,np.real(W[3]),levels=15)
ax.set_xlabel('X')
ax.set_ylabel('Y')
#%%
stest=30
Ptest, Wtest=generateDataset(-1,stest,nphi,lxs)

inputV=np.zeros((s,nxs*nphi))
outputV=np.zeros((s,nxs*nxs))
for i in range(0, len(P)):
    inputV[i]=P[i].flatten()
    outputV[i]=W[i].flatten()
   
testIn=np.zeros((stest,nxs*nphi))
testOut=np.zeros((stest,nxs*nxs))
for i in range(0,len(Ptest)):
    testIn[i]=Ptest[i].flatten()
    testOut[i]=Wtest[i].flatten()
#%%
'''
ai=simpleConv(nxs,nphi)
#%%
ai.model.compile(optimizer=tf.train.AdamOptimizer(0.001),#tf.train.GradientDescentOptimizer(0.005),#optimizer=tf.train.AdamOptimizer(0.001),
    loss='mean_squared_error')
history=ai.model.fit(P, W, epochs=3, batch_size=256, verbose=1, validation_data=(Ptest, Wtest))
'''
#%%

ai=simpleDeepNN2(nxs,nphi)
ai.model.compile(optimizer=keras.optimizers.Adam(0.001),#tf.train.GradientDescentOptimizer(0.005),#optimizer=tf.train.AdamOptimizer(0.001),
    loss='mean_squared_error')

checkpoint = ModelCheckpoint('models/ai_checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history=ai.model.fit(inputV, outputV, epochs=15, batch_size=256, verbose=1, validation_split=0.1, callbacks=callbacks_list)
#%%
ai.model.load_weights('models/ai_checkpoint.h5')
#%%
'''
with open('models/ai_model.json', 'w') as json_file:
    json_file.write(ai.model.to_json())
ai.model.save_weights('models/ai_weights.h5')
'''
#save AI
#%%
plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#%%
Wai_orig=ai.model.predict(testIn)
Wai=np.concatenate(Wai_orig)
Wai=np.reshape(Wai, (30,nxs,nxs))

fig, axs = plt.subplots(3, 6, sharex='col', sharey='row')

for i in range(0,6):
    axs[0,i].contourf(px,py,Ptest[i],levels=15)
    axs[0,i].set_ylabel('phi')
    #axs[0,i].axis('equal')
   
    axs[1,i].contourf(xs,ys,Wtest[i],levels=15)
    axs[1,i].set_ylabel('Y')
    #axs[1,i].axis('equal')

    axs[2,i].contourf(xs,ys,Wai[i],levels=15)
    axs[2,i].set_xlabel('X')
    axs[2,i].set_ylabel('Y')
    #axs[2,i].axis('equal')
axs[0,3].set_title('Probability distributions')
axs[1,3].set_title('Theoretical wigner distribution')
axs[2,3].set_title('AI prediction of wigner distribution')

plt.show()


#%%















