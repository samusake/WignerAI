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

import time
import json

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
s=10000 #number of samples
nphi=10#45 #number of angleSteps

nxs=10
xmax=5
lxs=np.linspace(-xmax, xmax, nxs)
[xs, ys]=np.meshgrid(lxs,lxs);

phispace=np.linspace(0,180,nphi, endpoint=False)
[px, py]=np.meshgrid(lxs,phispace)


'''
P, W=generateDatasetWithShiftAndSqueezed(N,s,phispace,lxs)
np.save('data/P10000_10_10_shift_squeezed', P)
np.save('data/W10000_10_10_shift_squeezed', W)
'''
#%%

P=np.load('data/P10000_10_10_shift_squeezed.npy')
W=np.load('data/W10000_10_10_shift_squeezed.npy')

#%%
fig=plt.figure(1)
ax=fig.add_subplot(111)

ax.contourf(px,py,P[2],levels=15)
ax.set_xlabel('u')
ax.set_ylabel('phi')

fig=plt.figure(2)
ax=fig.add_subplot(111)
ax.contourf(xs,ys,np.real(W[2]),levels=15)
ax.set_xlabel('X')
ax.set_ylabel('Y')

#%%
stest=30
Ptest, Wtest=generateDatasetWithShiftAndSqueezed(-1,stest,phispace,lxs)

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
P_input=P.reshape(s,nphi,nxs,1)
ai=simpleConv(nxs,nphi)
ai.model.compile(optimizer=keras.optimizers.RMSprop(0.001),#tf.train.GradientDescentOptimizer(0.005),#optimizer=tf.train.AdamOptimizer(0.001),
    loss='mean_squared_error')

checkpoint = ModelCheckpoint('models/ai_checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history=ai.model.fit(P_input, outputV, epochs=1, batch_size=32, verbose=1, validation_split=0.1, callbacks=callbacks_list)
ai.model.count_params()
'''
#%%

ai=smallDeepNN(nxs,nphi)

ai.model.compile(optimizer=keras.optimizers.RMSprop(0.001),#tf.train.GradientDescentOptimizer(0.005),#optimizer=tf.train.AdamOptimizer(0.001),
    loss='mean_squared_error')

checkpoint = ModelCheckpoint('models/ai_checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history=ai.model.fit(inputV, outputV, epochs=1, batch_size=32, verbose=1, validation_split=0.1, callbacks=callbacks_list)
for i in range(1,5):
    temphist=ai.model.fit(inputV, outputV, epochs=1, batch_size=32, verbose=1, validation_split=0.1, callbacks=callbacks_list)
    history.history['loss'].append(temphist.history['loss'][0])
    history.history['val_loss'].append(temphist.history['val_loss'][0])
ai.model.count_params()
#%%
ai.model.load_weights('models/ai_checkpoint.h5')
#%%

with open('models/ai_model.json', 'w') as json_file:
    json_file.write(ai.model.to_json())
ai.model.save_weights('models/ai_weights.h5')
with open('models/ai_history.json', 'w') as json_file:
    json.dump(history.history, json_file)
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

contour=np.linspace(-0.5,1,50)
for i in range(0,6):
    axs[0,i].contourf(px,py,Ptest[i],contour)
    axs[0,i].set_ylabel('phi')
    #axs[0,i].axis('equal')
   
    axs[1,i].contourf(xs,ys,Wtest[i],contour)
    axs[1,i].set_ylabel('Y')
    #axs[1,i].axis('equal')

    axs[2,i].contourf(xs,ys,Wai[i],contour)
    axs[2,i].set_xlabel('X')
    axs[2,i].set_ylabel('Y')
    #axs[2,i].axis('equal')
axs[0,3].set_title('Probability distributions')
axs[1,3].set_title('Theoretical wigner distribution')
axs[2,3].set_title('AI prediction of wigner distribution')

plt.show()


#%%












