#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:40:56 2019

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
def randomWignerAndDensityMatrix(N, xaxis):
    nxs=len(xaxis)
    rho=randomDensityMatrix(N)
    w=np.zeros((nxs,nxs))
    for i in range(0,N):
        for j in range(0,N):
            w=w+rho[i,j]*wnm(i,j,xaxis)
    w=np.real(w)
    return(w,rho)
    
def generateDatasetDensityMatrix(N,s,phispace,xaxis): #N-Dimension of rho, s-Number of samples, nphi-angular stepsize
    nxs=len(xaxis)   
    nphi=len(phispace)
    W=np.zeros((s,nxs,nxs))
    P=np.zeros((s,nphi,nxs))
    rho=np.zeros((s,N,N),dtype=complex)
    for i in range(0, s):
        if i%100==0:
            k=i/s*100
        W[i],rho[i]=randomWignerAndDensityMatrix(N,xaxis)
        P[i]=generatePofw(W[i],xaxis,phispace)
    return((P,W,rho))
#%%
N=2 #dimension of rho
s=100000 #number of samples
nphi=20#45 #number of angleSteps

nxs=20
xmax=5
lxs=np.linspace(-xmax, xmax, nxs)
[xs, ys]=np.meshgrid(lxs,lxs);

phispace=np.linspace(0,180,nphi, endpoint=False)
[px, py]=np.meshgrid(lxs,phispace)


'''
Pr,Wr,rho=generateDatasetDensityMatrix(N,s,phispace,lxs)
np.save('data/100000_20_20_density_N2/P', Pr)
np.save('data/100000_20_20_density_N2/W', Wr)
np.save('data/100000_20_20_density_N2/rho', rho)
'''
#%%

P=np.load('data/100000_20_20_density_N2/P.npy')
W=np.load('data/100000_20_20_density_N2/W.npy')
rho=np.load('data/100000_20_20_density_N2/rho.npy')

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
Ptest, Wtest, rhotest=generateDatasetDensityMatrix(N,stest,phispace,lxs)

#split real and complex
rhotestsplit=np.stack((rhotest.real, rhotest.imag), -1)
#rhoback=rhosplit[:,:,:,0]+1j*rhosplit[:,:,:,1]
#%%
rhosplit=np.stack((rho.real, rho.imag), -1)

inputV=np.zeros((s,nxs*nphi))
outputV=np.zeros((s,2*N*N))
for i in range(0, len(P)):
    inputV[i]=P[i].flatten()
    outputV[i]=rhosplit[i].flatten()
   
testIn=np.zeros((stest,nxs*nphi))
testOut=np.zeros((stest,2*N*N))
for i in range(0,len(Ptest)):
    testIn[i]=Ptest[i].flatten()
    testOut[i]=rhotestsplit[i].flatten()
#%%

#%%

ai=DensityDeepNN(N, nxs,nphi)
ai.model.compile(optimizer=keras.optimizers.RMSprop(0.001),#tf.train.GradientDescentOptimizer(0.005),#optimizer=tf.train.AdamOptimizer(0.001),
    loss='mean_squared_error')

checkpoint = ModelCheckpoint('models/ai_checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history=ai.model.fit(inputV, outputV, epochs=30, batch_size=32, verbose=1, validation_split=0.1, callbacks=callbacks_list)

#%%
ai.model.load_weights('models/ai_checkpoint.h5')
#%%

with open('models/ai_model.json', 'w') as json_file:
    json_file.write(ai.model.to_json())
ai.model.save_weights('models/ai_weights.h5')

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
json_file = open('models/ai_100000_12_20.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
ai = keras.models.model_from_json(loaded_model_json)
# load weights into new model
ai.load_weights("models/ai_100000_12_20.h5")
print("Loaded model from disk")
#%%
rhoai_orig=ai.predict(testIn)
rhoai=np.concatenate(rhoai_orig)
rhoai=np.reshape(rhoai, (30,N,N))

fig, axs = plt.subplots(3, 6, sharex='col', sharey='row')

contour=np.linspace(-0.5,1,50)
for i in range(0,6):
    axs[0,i].contourf(px,py,Ptest[i],contour)
    axs[0,i].set_ylabel('phi')
    #axs[0,i].axis('equal')
   
    axs[1,i].imshow(rhotest[i])
    axs[1,i].set_ylabel('Y')
    #axs[1,i].axis('equal')

    axs[2,i].imshow(rhoai[i])
    axs[2,i].set_xlabel('X')
    axs[2,i].set_ylabel('Y')
    #axs[2,i].axis('equal')
axs[0,3].set_title('Probability distributions')
axs[1,3].set_title('Theoretical wigner distribution')
axs[2,3].set_title('AI prediction of wigner distribution')

plt.show()
