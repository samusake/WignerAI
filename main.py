#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 19:34:08 2019

@author: herbert
"""

import numpy as np
from scipy.special import factorial
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as sk

from numba import jit
import time
import pickle

from genSample import *
from model import *
#%%
N=4 #dimension of rho
s=5000 #number of samples
nphi=12 #number of angleSteps

nxs=21
xmax=5
lxs=np.linspace(-xmax, xmax, nxs)
[xs, ys]=np.meshgrid(lxs,lxs);

phispace=np.linspace(0,180,nphi)
[px, py]=np.meshgrid(lxs,phispace)

P, W=generateDataset(N,s,nphi,lxs)

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
Ptest, Wtest=generateDataset(N,stest,nphi,lxs)

inputV=np.zeros((s,nxs*nphi))
outputV=np.zeros((s,nxs*nxs))
for i in range(0, len(P)):
    inputV[i]=P[i].flatten()
    outputV[i]=W[i].flatten()
   
testIn=np.zeros((stest,nxs*nphi))
testOut=np.zeros((stest,nxs*nxs))
for i in range(0,len(Ptest)):
    testIn[i]=P[i].flatten()
    testOut[i]=W[i].flatten()
    
#%%
ai=simpleDeepNN(nxs,nphi)
#%%
ai.model.compile(optimizer=tf.train.AdamOptimizer(0.001),#tf.train.GradientDescentOptimizer(0.005),#optimizer=tf.train.AdamOptimizer(0.001),
    loss='mean_squared_error')
history=ai.model.fit(inputV, outputV, epochs=200, batch_size=32, verbose=1, validation_data=(testIn, testOut))

#%%
plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
#%%

fig=plt.figure(5)
ax=fig.add_subplot(111)
ax.contourf(xs,ys,np.real(Wtest[9]),levels=15)
ax.set_xlabel('X')
ax.set_ylabel('Y')


Wai=ai.model.predict(testIn)
Wai=np.reshape(Wai,(30,nxs,nxs))
fig=plt.figure(6)
ax=fig.add_subplot(111)
ax.contourf(xs,ys,Wai[9],levels=15)
ax.set_xlabel('X')
ax.set_ylabel('Y')
