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

from WigToDensity import *
from genSample import *
from model import *

import keras.backend as K
dtype='float16'
K.set_floatx(dtype)
#%%
#https://arxiv.org/pdf/1811.06654.pdf
def randomWignerAndDensityMatrix(N, xaxis):
    nxs=len(xaxis)
    fs=randomDensityMatrix(N)
    w=np.zeros((nxs,nxs))
    for i in range(0,N):
        for j in range(0,N):
            w=w+rho[i,j]*wnm(i,j,xaxis)
    #w=np.real(w)
    return(w,fs)
    
def generateDatasetDensityMatrix(N,s,phispace,xaxis): #N-Dimension of rho, s-Number of samples, nphi-angular stepsize
    nxs=len(xaxis)   
    nphi=len(phispace)
    W=np.zeros((s,nxs,nxs))
    P=np.zeros((s,nphi,nxs))
    rho=np.zeros((s,N,N),dtype=complex)
    for i in range(0, s):
        if i%100==0:
            k=i/s*100
            print("{0} %".format(k))
        W[i],fs[i]=randomWignerAndDensityMatrix(N,xaxis)
        P[i]=generatePofw(W[i],xaxis,phispace)
        rho[i]=rho1modeps(W[i],phispace, xaxis, N-1)
    return((P,W,rho))
def fidelity(rho, sigma):
    return(np.square(np.trace(np.sqrt(np.sqrt(rho).dot(sigma).dot(np.sqrt(rho))))))
#%%
N=5 #dimension of rho
s=20000 #number of samples
nphi=20#45 #number of angleSteps

nxs=20
xmax=5
lxs=np.linspace(-xmax, xmax, nxs)
[xs, ys]=np.meshgrid(lxs,lxs);

phispace=np.linspace(0,180,nphi, endpoint=False)
[px, py]=np.meshgrid(lxs,phispace)



P,W,rho=generateDatasetDensityMatrix(N,s,phispace,lxs)

np.save('data/100000_20_20_density_N5/20000P', P)
np.save('data/100000_20_20_density_N5/20000W', W)
np.save('data/100000_20_20_density_N5/20000rho', rho)

#%%
'''
P=np.load('data/100000_20_20_density_N5/P.npy')
W=np.load('data/100000_20_20_density_N5/W.npy')
rho=np.load('data/100000_20_20_density_N5/rho.npy')
'''
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
for i in range(0, s):
    inputV[i]=P[i].flatten()
    outputV[i]=rhosplit[i].flatten()
   
testIn=np.zeros((stest,nxs*nphi))
testOut=np.zeros((stest,2*N*N))
for i in range(0,stest):
    testIn[i]=Ptest[i].flatten()
    testOut[i]=rhotestsplit[i].flatten()
#%%

#%%

ai=smallDensityDeepNN(N, nxs,nphi)
ai.model.compile(optimizer=keras.optimizers.Adam(0.001),#decay=0.0001),#tf.train.GradientDescentOptimizer(0.005),#optimizer=tf.train.AdamOptimizer(0.001),
    loss='mean_squared_error')

checkpoint = ModelCheckpoint('models/ai_checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history=ai.model.fit(inputV, outputV, epochs=200, batch_size=32, verbose=1, validation_split=0.1, callbacks=callbacks_list)

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
'''
json_file = open('models/ai_100000_12_20.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
ai = keras.models.model_from_json(loaded_model_json)
# load weights into new model
ai.load_weights("models/ai_100000_12_20.h5")
print("Loaded model from disk")
'''
#%%
rhoai_orig=ai.model.predict(testIn)
rhoai_orig=np.concatenate(rhoai_orig)
rhoai_real=np.zeros(int(len(rhoai_orig)/2))
rhoai_imag=np.zeros(int(len(rhoai_orig)/2))

for i in range(0, len(rhoai_orig)):
    if i%2==0:
        rhoai_real[int(i/2)]=rhoai_orig[i]
        rhoai_imag[int(i/2)]=rhoai_orig[i+1]
rhoai=rhoai_real+1j*rhoai_imag
rhoai=np.reshape(rhoai, (30,N,N))  

#rhoai=np.concatenate(rhoai_orig)
#rhoai=np.reshape(rhoai, (30,N,N))

fig, axs = plt.subplots(5, 6)

contour=np.linspace(-0.5,1,50)
for i in range(0,6):
    axs[0,i].contourf(px,py,Ptest[i])
    axs[0,i].set_ylabel('phi')
    axs[0,i].set_xlabel('x')
    axs[0,i].invert_yaxis()
    #axs[0,i].axis('equal')
   
    axs[1,i].imshow(rhotest.real[i])
    axs[1,i].invert_yaxis()
    #axs[1,i].axis('equal')

    axs[2,i].imshow(rhoai.real[i])
    axs[2,i].invert_yaxis()
    #axs[2,i].axis('equal')
    
    axs[3,i].imshow(rhotest.imag[i])
    axs[3,i].invert_yaxis()
    #axs[1,i].axis('equal')
    #axs[1,i].axis('equal')

    axs[2,i].imshow(rhoai.real[i])
    axs[2,i].invert_yaxis()
    #axs[2,i].axis('equal')
    
    axs[3,i].imshow(rhotest.imag[i])
    axs[3,i].invert_yaxis()
    #axs[1,i].axis('equal')
    axs[4,i].imshow(rhoai.imag[i])
    axs[4,i].invert_yaxis()
    #axs[2,i].axis('equal')
axs[0,3].set_title('Probability distributions')
axs[1,3].set_title('Theoretical density matrix - real values')
axs[2,3].set_title('AI prediction of density matrix - real values')
axs[3,3].set_title('Theoretical density matrix - complex values')
axs[4,3].set_title('AI prediction of density matrix - complex values')

plt.show()
#%%
i=0
print(fidelity(rhotest[i],rhoai[i]))