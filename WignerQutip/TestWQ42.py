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

import matplotlib
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
#%%#
P=np.load('data/P42.npy')
W=np.load('data/W42.npy')
#%%
inputV=np.zeros((s,nxs*nphi))
outputV=np.zeros((s,nxs*nxs))
for i in range(0, len(P)):
    inputV[i]=P[i].flatten()
    outputV[i]=W[i].flatten()
#%%
json_file = open('models/randomWigner_Nsmaller5/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
ai = keras.models.model_from_json(loaded_model_json)
# load weights into new model
ai.load_weights("models/randomWigner_Nsmaller5/weights.h5")
print("Loaded model from disk")

f = open('models/randomWigner_Nsmaller5/history.json', 'r')
history=json.load(f)
f.close()
#%%
plt.semilogy(history['loss'])
plt.semilogy(history['val_loss'])

#plt.title('Model loss')1121
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()
#%%
'''
reconstruction_fbp=np.zeros((s, nxs,nphi))
start=time.time()
for i in range(0,s):
    reconstruction_fbp[i] = iradon(P[i], theta=phispace, circle=True)
end=time.time()
error = reconstruction_fbp - W
#%%
print('FBP reconstruction RMSD (100000 samples): %.3g' % np.sqrt(np.mean(error**2)))
print('FBP calculation duration (100000 samples): %.3g' % (end-start))
print('FBP mean calculation duration: %.3g' % ((end-start)/s))
#sinogram=radon(W[i], theta=phispace, circle=True)
#%%
#predict for all
start=time.time()
Wai=ai.predict(inputV)
end=time.time()
Wai=np.concatenate(Wai)
Wai=np.reshape(Wai, (s,nxs,nxs))
#%%
error = Wai - W


print('AI reconstruction RMSD (100000 samples): %.3g' % np.sqrt(np.mean(error**2)))
print('AI prediction duration (100000 samples): %.3g' % (end-start))
#%%
start=time.time()
for i in range(0,s):
    wai_orig=ai.predict(np.array([inputV[i]]))
end=time.time()
print('AI mean prediction duration: %.3g' % ((end-start)/s))
'''

#%%

Wai=ai.predict(inputV)
inputV=None
outputV=None
Wai=np.concatenate(Wai)
Wai=np.reshape(Wai, (s,nxs,nxs))

#%%fig, axs = plt.subplots(3, 6)
k=23
font={'size'   : '14'}
matplotlib.rc('font', **font)


fig, axs = plt.subplots(4, 6)

for i in range(0,6):
    axs[0,i].contourf(px, py,P[i+k],levels=100)
    #axs[0,i].axis('equal')
   
    axs[1,i].contourf(xs,ys,W[i+k]/np.mean(W[i+k]),levels=100)
    
    #axs[1,i].axis('equal')

    axs[2,i].contourf(xs,ys,Wai[i+k]/np.mean(W[i+k]),levels=100)
    
    
    axs[3,i].contourf(xs,ys, reconstruction_fbp[i+k]/np.mean(W[i+k]),levels=100)
    axs[3,i].set_xlabel('q')

    #axs[2,i].axis('equal')
axs[0,2].set_title('Sinogram')
axs[1,2].set_title('Theoretical wigner distribution')
axs[2,2].set_title('MLP prediction of wigner distribution')
axs[3,2].set_title('FBP (skimage implementation)')

axs[0,0].set_ylabel('θ')
axs[1,0].set_ylabel('p')
axs[2,0].set_ylabel('p')
axs[3,0].set_ylabel('p')
plt.subplots_adjust(hspace=0.5, wspace=0.4)
plt.show()

#%%
fig2=plt.figure(2)
ax2=fig2.add_subplot(111,projection='3d')
ax2.plot_surface(xs,ys,Wai[k], rstride=1, cstride=1, cmap='viridis', edgecolor='none')

plt.show()

#%%
square=sk.io.imread('../images/square.png',as_gray=True)
square=sk.util.invert(square)
psquare=radon(square, theta=phispace, circle=True)


#%%
squareAI=ai.predict(np.array([np.concatenate(psquare)]))
inputV=None
outputV=None
squareAI=np.concatenate(squareAI)
squareAI=np.reshape(squareAI, (nxs,nxs))

fig3, axs3=plt.subplots(1,3)
axs3[0].contourf(px, py, psquare,levels=100)
axs3[1].contourf(xs,ys, square,levels=100)
axs3[2].contourf(xs,ys, squareAI ,levels=100)

    #axs[2,i].axis('equal')
axs3[0].set_title('Sinogram')
axs3[1].set_title('Ground truth')
axs3[2].set_title('NNE')

axs3[0].set_ylabel('θ')
axs3[1].set_ylabel('p')
axs3[2].set_ylabel('p')
axs3[0].set_xlabel('q')
axs3[1].set_xlabel('q')
axs3[2].set_xlabel('q')

plt.show()