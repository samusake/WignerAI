#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:25:37 2019

@author: herbert
"""
import numpy as np
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

from skimage.transform import iradon, iradon_sart
from skimage.transform import radon, rescale
from scipy.special import erf
#%%
N=-1 #dimension of rho
stest=1000 #number of samples
nphi=100#45 #number of angleSteps

nxs=100
xmax=5
lxs=np.linspace(-xmax, xmax, nxs)
[xs, ys]=np.meshgrid(lxs,lxs);

phispace=np.linspace(0,180,nphi, endpoint=False)
[px, py]=np.meshgrid(lxs,phispace)
#%%
#stest=10
#P, W=generateDatasetWithShiftAndSqueezed(-1,stest,phispace,lxs)
P=np.load('data/Ptest100000_100_100_shift_squeezed.npy')
W=np.load('data/Wtest100000_100_100_shift_squeezed.npy')

#%%
P_radon=np.zeros((stest,nphi,nxs))
for i in range(0,stest):
    P_radon[i]=generateP_radonofw(W[i],lxs,phispace)
#%%
json_file = open('models/100/ai_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
ai = keras.models.model_from_json(loaded_model_json)
# load weights into new model
ai.load_weights("models/100/ai_checkpoint.h5")
print("Loaded model from disk")

#%%
reconstruction_fbp=np.zeros((stest, nxs,nphi))
start=time.time()
for i in range(0,stest):
    reconstruction_fbp[i] = iradon(P_radon[i], theta=phispace)
end=time.time()
error = reconstruction_fbp - W
print('FBP reconstruction error (1000 samples): %.3g' % np.sqrt(np.mean(error**2)))
print('FBP calculation duration (1000 samples): %.3g' % (end-start))
print('FBP mean calculation duration: %.3g' % ((end-start)/stest))
#sinogram=radon(W[i], theta=phispace, circle=True)

inputV=np.zeros((stest,nxs*nphi))
for i in range(0, len(P)):
    inputV[i]=P[i].flatten()


#predict for all
start=time.time()
wai_orig=ai.predict(inputV)
end=time.time()
wai=np.concatenate(wai_orig)
wai=np.reshape(wai, (stest,nxs,nxs))

error = wai - W


print('AI reconstruction error (1000 samples): %.3g' % np.sqrt(np.mean(error**2)))
print('AI prediction duration (1000 samples): %.3g' % (end-start))

start=time.time()
for i in range(0,stest):
    wai_orig=ai.predict(np.array([inputV[i]]))
end=time.time()
print('AI mean prediction duration: %.3g' % ((end-start)/stest))


#%%
#contour1=erf(np.linspace(-0.7,0.7,100)/np.sqrt(2))
#contour2=erf(np.linspace(-0.5,1,100)/np.sqrt(2*0.4))
fig, axs = plt.subplots(4,4, sharex=True)
for i in range(0,4):
    axs[0,i].contourf(px,py,P[i]/np.mean(P[i]),levels=15,extend='both')
    axs[1,i].contourf(xs,ys,W[i]/np.mean(W[i]),levels=15,extend='both')
    axs[1,i].axis('equal')  
    axs[2,i].contourf(xs,ys, wai[i]/np.mean(W[i]),levels=15,extend='both')
    axs[2,i].axis('equal')    
    axs[3,i].contourf(xs,ys, reconstruction_fbp[i]/np.mean(W[i]),levels=15,extend='both')
    axs[3,i].axis('equal')


axs[0,0].set_ylabel('phi')
axs[1,0].set_ylabel('Y')
axs[2,0].set_ylabel('Y')
axs[3,0].set_ylabel('Y')

axs[3,0].set_xlabel('X')
axs[3,1].set_xlabel('X')
axs[3,2].set_xlabel('X')
axs[3,3].set_xlabel('X')

axs[0,1].title.set_text('Sinogram')
axs[1,1].title.set_text('Theoretical wigner distribution')
axs[2,1].title.set_text('AI prediction')
axs[3,1].title.set_text('FBP (skimage implementation)')

plt.show()