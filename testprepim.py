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
import json

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
stest=3060 #number of samples
nphi=42#45 #number of angleSteps

nxs=42
xmax=5
lxs=np.linspace(-xmax, xmax, nxs)
[xs, ys]=np.meshgrid(lxs,lxs);

phispace=np.linspace(0,180,nphi, endpoint=False)
[px, py]=np.meshgrid(lxs,phispace)
#%%
#stest=10

P=np.load('data/image_learn/P30608_images.npy')[(30608-3060):]
W=np.load('data/image_learn/W30608_images.npy')[(30608-3060):]

#%%
P_radon=np.zeros((stest, 60,nphi))
for i in range(0,stest):
    #P_radon[i]=generateP_radonofw(W[i],lxs,phispace)
    P_radon[i]=radon(W[i],theta=phispace, circle=False)
#%%
json_file = open('models/images42/ai_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
ai = keras.models.model_from_json(loaded_model_json)
# load weights into new model
ai.load_weights("models/images42/ai_checkpoint.h5")
print("Loaded model from disk")
#%%
with open('models/images42/ai_history.json') as json_file:
    history=json.load(json_file)

plt.semilogy(history['loss'][3:499])
plt.semilogy(history['val_loss'][3:499])

plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
#%%
reconstruction_fbp=np.zeros((stest, nxs,nphi))
start=time.time()
for i in range(0,stest):
    reconstruction_fbp[i] = iradon(P_radon[i], theta=phispace)
end=time.time()
error = reconstruction_fbp - W
print('FBP reconstruction error (3060 samples): %.3g' % np.sqrt(np.mean(error**2)))
print('FBP calculation duration (3060 samples): %.3g' % (end-start))
print('FBP mean calculation duration: %.3g' % ((end-start)/stest))
#sinogram=radon(W[i], theta=phispace, circle=True)

inputV=np.zeros((stest,60*nphi))
for i in range(0, len(P)):
    inputV[i]=P[i].flatten()


#predict for all
start=time.time()
wai_orig=ai.predict(inputV)
end=time.time()
wai=np.concatenate(wai_orig)
wai=np.reshape(wai, (stest,nxs,nxs))

error = wai - W


print('AI reconstruction error (3060 samples): %.3g' % np.sqrt(np.mean(error**2)))
print('AI prediction duration (3060 samples): %.3g' % (end-start))

start=time.time()
for i in range(0,stest):
    wai_orig=ai.predict(np.array([inputV[i]]))
end=time.time()
print('AI mean prediction duration: %.3g' % ((end-start)/stest))


#%%
#contour1=erf(np.linspace(-0.7,0.7,100)/np.sqrt(2))
#contour2=erf(np.linspace(-0.5,1,100)/np.sqrt(2*0.4))
k=55
fig, axs = plt.subplots(4,4, sharex=True)
for i in range(0,4):
    axs[0,i].contourf(P[i+k]/np.mean(P[i+k]),levels=15,extend='both')
    axs[1,i].imshow(W[i+k], cmap='gray')
    #axs[1,i].axis('equal')  
    axs[2,i].imshow(wai[i+k], cmap='gray')
    #axs[2,i].axis('equal')    
    axs[3,i].imshow(reconstruction_fbp[i+k], cmap='gray')
    #axs[3,i].axis('equal')


axs[0,0].set_ylabel('phi')
axs[1,0].set_ylabel('Y')
axs[2,0].set_ylabel('Y')
axs[3,0].set_ylabel('Y')

axs[3,0].set_xlabel('X')
axs[3,1].set_xlabel('X')
axs[3,2].set_xlabel('X')
axs[3,3].set_xlabel('X')

axs[0,1].title.set_text('Sinogram')
axs[1,1].title.set_text('Ground truth')
axs[2,1].title.set_text('AI prediction')
axs[3,1].title.set_text('FBP (skimage implementation)')

plt.show()