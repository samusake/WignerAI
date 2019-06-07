#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:24:39 2019

@author: herbert
"""

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

P=np.load('data/P10000_10_10_shift_squeezed.npy')
W=np.load('data/W10000_10_10_shift_squeezed.npy')

inputV=np.zeros((s,nxs*nphi))
outputV=np.zeros((s,nxs*nxs))
for i in range(0, len(P)):
    inputV[i]=P[i].flatten()
    outputV[i]=W[i].flatten()
#%%
ai=smallDeepNN1(nxs,nphi)

ai.model.compile(optimizer=keras.optimizers.Adam(0.01),#tf.train.GradientDescentOptimizer(0.005),#optimizer=tf.train.AdamOptimizer(0.001),
    loss='mean_squared_error')

checkpoint = ModelCheckpoint('models/ai_checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history1=ai.model.fit(inputV, outputV, epochs=40, batch_size=32, verbose=1, validation_split=0.1, callbacks=callbacks_list)
c1=ai.model.count_params()
#%%
ai=smallDeepNN1(nxs,nphi)

ai.model.compile(optimizer=keras.optimizers.Adam(0.001),#tf.train.GradientDescentOptimizer(0.005),#optimizer=tf.train.AdamOptimizer(0.001),
    loss='mean_squared_error')

checkpoint = ModelCheckpoint('models/ai_checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history2=ai.model.fit(inputV, outputV, epochs=40, batch_size=32, verbose=1, validation_split=0.1, callbacks=callbacks_list)
c2=ai.model.count_params()
#%%
ai=smallDeepNN1(nxs,nphi)

ai.model.compile(optimizer=keras.optimizers.Adam(0.0005),#tf.train.GradientDescentOptimizer(0.005),#optimizer=tf.train.AdamOptimizer(0.001),
    loss='mean_squared_error')

checkpoint = ModelCheckpoint('models/ai_checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history3=ai.model.fit(inputV, outputV, epochs=40, batch_size=32, verbose=1, validation_split=0.1, callbacks=callbacks_list)
c3=ai.model.count_params()
#%%
ai=smallDeepNN1(nxs,nphi)

ai.model.compile(optimizer=keras.optimizers.Adam(0.0001),#tf.train.GradientDescentOptimizer(0.005),#optimizer=tf.train.AdamOptimizer(0.001),
    loss='mean_squared_error')

checkpoint = ModelCheckpoint('models/ai_checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history4=ai.model.fit(inputV, outputV, epochs=40, batch_size=32, verbose=1, validation_split=0.1, callbacks=callbacks_list)
c4=ai.model.count_params()
#%%
ai.model.load_weights('models/ai_checkpoint.h5')
#%%

with open('models/ai_model.json', 'w') as json_file:
    json_file.write(ai.model.to_json())
ai.model.save_weights('models/ai_weights.h5')

#save AI
#%%
plt.semilogy(history1.history['val_loss'])
plt.semilogy(history2.history['val_loss'])
plt.semilogy(history3.history['val_loss'])
plt.semilogy(history4.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Adam, lr=0,01', 'Adam, lr=0,001', 'Adam, lr=0,0005', 'Adam, lr=0,0001'], loc='right')
plt.show()

print("#Param1=", c1)
print("#Param2=", c2)
print("#Param3=", c3)
print("#Param4=", c4)














