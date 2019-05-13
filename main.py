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
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as sk

from numba import jit
import time
import pickle

from genSample import *
from model import *
#%%

N=2 #dimension of rho
s=200000 #number of samples
nphi=12 #number of angleSteps

nxs=17
xmax=5
lxs=np.linspace(-xmax, xmax, nxs)
[xs, ys]=np.meshgrid(lxs,lxs);

phispace=np.linspace(0,180,nphi)
[px, py]=np.meshgrid(lxs,phispace)

'''
P, W=generateDataset(N,s,nphi,lxs)
np.save('data/P200000', P)
np.save('data/W200000', W)
'''
#%%
P=np.load('data/P200000.npy')
W=np.load('data/W200000.npy')
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
    testIn[i]=Ptest[i].flatten()
    testOut[i]=Wtest[i].flatten()

P2=np.concatenate(testIn)
P2=np.reshape(P2,(30,nphi,nxs))
print(mean_squared_error(Ptest[1],P2[1]))

#%%
'''
ai=simpleConv(nxs,nphi)
#%%
ai.model.compile(optimizer=tf.train.AdamOptimizer(0.001),#tf.train.GradientDescentOptimizer(0.005),#optimizer=tf.train.AdamOptimizer(0.001),
    loss='mean_squared_error')
history=ai.model.fit(P, W, epochs=3, batch_size=256, verbose=1, validation_data=(Ptest, Wtest))
'''
#%%
ai=simpleDeepNN(nxs,nphi)
ai.model.compile(optimizer=tf.train.AdamOptimizer(0.001),#tf.train.GradientDescentOptimizer(0.005),#optimizer=tf.train.AdamOptimizer(0.001),
    loss='mean_squared_error')
history=ai.model.fit(inputV, outputV, epochs=3, batch_size=256, verbose=1, validation_data=(testIn, testOut))
#%%
#%%
#save AI
#%%
plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
score=ai.model.evaluate(testIn, testOut, batch_size=1)
print(score)
t1=ai.model.predict(testIn)
print(mean_squared_error(t1,testOut))
#%%
k=4

Wai_orig=ai.model.predict(testIn)
Wai=np.concatenate(Wai_orig)
Wai=np.reshape(Wai, (30,nxs,nxs))

for i in range(k,k+1):
    fig=plt.figure(i)
    ax=fig.add_subplot(111)
    ax.contourf(xs,ys,np.real(Wtest[i]),levels=15)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    fig=plt.figure(6)
    ax=fig.add_subplot(111)
    ax.contourf(xs,ys,Wai[i],levels=15)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
print(mean_squared_error(Wai[k],Wtest[k]))