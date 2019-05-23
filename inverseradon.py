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
#%%
N=-1 #dimension of rho
s=100000 #number of samples
nphi=20#45 #number of angleSteps

nxs=40
xmax=5
lxs=np.linspace(-xmax, xmax, nxs)
[xs, ys]=np.meshgrid(lxs,lxs);

phispace=np.linspace(0,180,nphi, endpoint=False)
[px, py]=np.meshgrid(lxs,phispace)
#%%
P=np.load('data/P100000_12_12.npy')
W=np.load('data/W100000_12_12.npy')
#%%
i=3
p=generatePofw(W[i],lxs, phispace)
reconstruction_fbp = iradon(np.transpose(p), theta=phispace)
#error = reconstruction_fbp - W[0]
#print('FBP rms reconstruction error: %.3g' % np.sqrt(np.mean(error**2)))

sinogram=radon(W[i], theta=phispace, circle=True)


contour=np.linspace(-0.3,0.4,50)
fig, axs = plt.subplots(4,4, sharex=True)
axs[0,0].contourf(px,py,p,levels=15)
axs[0,0].set_xlabel('r')
axs[0,0].set_ylabel('phi')
axs[0,0].axis('equal')
   
axs[1,0].contourf(xs,ys,W[i],levels=15)
axs[1,0].set_xlabel('X')
axs[1,0].set_ylabel('Y')
axs[1,0].axis('equal')

axs[2,0].contourf(px,py, np.transpose(sinogram),levels=15)
axs[2,0].set_xlabel('X')
axs[2,0].set_ylabel('Y')
axs[2,0].axis('equal')

axs[3,0].contourf(xs,ys, reconstruction_fbp,levels=15)
axs[3,0].set_xlabel('X')
axs[3,0].set_ylabel('Y')
axs[3,0].axis('equal')

