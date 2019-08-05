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

N=10 #number of qubits: n, N=2^n
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
    W[i]=wigner(rho_current,lxs,lxs)
    P[i]=radon(W[i],theta=phispace, circle=True)

np.save('data/P42_rho10', P)
np.save('data/rho42_rho10', rho)
np.save('data/W42_rho10', W)
'''
#%%

P=np.load('data/P42_rho10.npy')
rho=np.load('data/rho42_rho10.npy')

#%%
rhosplit=np.stack((rho.real, rho.imag), -1)
outputV=np.zeros((s,2*N*N))
inputV=np.zeros((s,nxs*nphi))
for i in range(0, len(P)):
    inputV[i]=P[i].flatten()
    outputV[i]=rhosplit[i].flatten()
#%%
model=tf.keras.Sequential()
        
model.add(layers.Dense(nphi*nxs, activation='relu'))
model.add(layers.Dense(1600, activation='relu'))
model.add(layers.Dense(1600, activation='relu'))
model.add(layers.Dense(2*N*N, activation='linear'))

model.compile(optimizer=keras.optimizers.Adam(0.001),#decay=0.0001),#tf.train.GradientDescentOptimizer(0.005),#optimizer=tf.train.AdamOptimizer(0.001),
    loss='mean_squared_error')

checkpoint = ModelCheckpoint('models/ai_checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
#%%
history=model.fit(inputV, outputV, epochs=400, batch_size=256, verbose=1, validation_split=0.1, callbacks=callbacks_list)
#%%
model.load_weights('models/ai_checkpoint.h5')
#%%
'''
with open('models/density_10/10_model.json', 'w') as json_file:
    json_file.write(model.to_json())
model.save_weights('models/density_10/10_weights.h5')
with open('models/density_10/10_history.json', 'w') as json_file:
    json.dump(history.history, json_file)
'''
#%%
plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])

plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
#%%
rhoai_orig=model.predict(inputV[int(s-0.1*s):s])
rhoai_orig=np.concatenate(rhoai_orig)
rhoai_real=np.zeros(int(len(rhoai_orig)/2))
rhoai_imag=np.zeros(int(len(rhoai_orig)/2))

for i in range(0, len(rhoai_orig)):
    if i%2==0:
        rhoai_real[int(i/2)]=rhoai_orig[i]
        rhoai_imag[int(i/2)]=rhoai_orig[i+1]
rhoai=rhoai_real+1j*rhoai_imag
rhoai=np.reshape(rhoai, (int(0.1*s),N,N))  

Ptest=P[int(s-0.1*s):s]
rhotest=rho[int(s-0.1*s):s]
#rhoai=np.concatenate(rhoai_orig)
#rhoai=np.reshape(rhoai, (30,N,N))
#%%
fig, axs = plt.subplots(5, 6, sharex='col', sharey='row')

contour=np.linspace(-0.5,1,50)
for i in range(0,6):
    axs[0,i].contourf(px,py,Ptest[i])
    #axs[0,i].axis('equal')
   
    axs[1,i].imshow(rhotest.real[i])
    #axs[1,i].axis('equal')

    axs[2,i].imshow(rhoai.real[i])
    #axs[2,i].axis('equal')
    
    axs[3,i].imshow(rhotest.imag[i])
    
    #axs[1,i].axis('equal')
    axs[4,i].imshow(rhoai.imag[i])
    #axs[2,i].axis('equal')
axs[0,0].set_ylabel('θ')
axs[0,3].set_title('Sinogram')
axs[1,3].set_title('Theoretical density matrix - real values')
axs[2,3].set_title('AI prediction of density matrix - real values')
axs[3,3].set_title('Theoretical density matrix - complex values')
axs[4,3].set_title('AI prediction of density matrix - complex values')

plt.show()