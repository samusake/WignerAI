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
s=200000#number of samples
nphi=84#45 #number of angleSteps

nxs=84
xmax=5
lxs=np.linspace(-xmax, xmax, nxs)
[xs, ys]=np.meshgrid(lxs,lxs)

phispace=np.linspace(0,180,nphi, endpoint=False)
[px, py]=np.meshgrid(lxs,phispace)

N=20 #number of qubits: n, N=2^n
#%%
'''
W=np.zeros((s,nxs,nxs))
P=np.zeros((s,nphi,nxs))
rho=np.zeros((s,N,N),dtype=complex)#,Nmax))
for i in range(0,s):        
    if i%100==0:
        k=i/s*100
        print("{0} %".format(k))
    rho_current=rand_dm(N)
    rho[i]=rho_current.full()
    W=wigner(rho_current,lxs,lxs)
    P[i]=radon(W,theta=phispace, circle=True)

np.save('data/P84_rho20', P)
np.save('data/rho84_rho20', rho)
'''
#%%

P=np.load('data/P84_rho20.npy')
rho=np.load('data/rho84_rho20.npy')

#%%
rhosplit=np.stack((rho.real, rho.imag), -1)
rho=None
outputV=np.zeros((s,2*N*N))
inputV=np.zeros((s,nxs*nphi))
for i in range(0, len(P)):
    inputV[i]=P[i].flatten()
    outputV[i]=rhosplit[i].flatten()
P=None
rhosplit=None
#%%
model=tf.keras.Sequential()
        
model.add(layers.Dense(nphi*nxs, activation='relu'))
model.add(layers.Dense(7056, activation='relu'))
model.add(layers.Dense(7056, activation='relu'))
model.add(layers.Dense(2*N*N, activation='linear'))

model.compile(optimizer=keras.optimizers.Adam(0.001),#decay=0.0001),#tf.train.GradientDescentOptimizer(0.005),#optimizer=tf.train.AdamOptimizer(0.001),
    loss='mean_squared_error')

checkpoint = ModelCheckpoint('models/ai_checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
#%%
history=model.fit(inputV, outputV, epochs=42, batch_size=512, verbose=1, validation_split=0.2, callbacks=callbacks_list)
#%%
model.load_weights('models/checkpoint.h5')
#%%

with open('models/density_20_2/20_model.json', 'w') as json_file:
    json_file.write(model.to_json())
model.save_weights('models/density_20_2/20_weights.h5')
with open('models/density_20_2/20_history.json', 'w') as json_file:
    json.dump(history.history, json_file)

#%%