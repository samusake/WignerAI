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

from genSample import *
from model import *

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import model_from_json

N=-1 #dimension of rho
nphi=60#45 #number of angleSteps

nxs=60
xmax=5
lxs=np.linspace(-xmax, xmax, nxs)
[xs, ys]=np.meshgrid(lxs,lxs);

phispace=np.linspace(0,180,nphi, endpoint=False)
[px, py]=np.meshgrid(lxs,phispace)

#%%

json_file = open('models/60/ai_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
ai = keras.models.model_from_json(loaded_model_json)
# load weights into new model
ai.load_weights("models/60/ai_weights.h5")
print("Loaded model from disk")

#%%
#Gauss
w=np.exp(-xs**2/2.-ys**2/2.)
p=generatePofw(w,lxs,phispace)

p_test=p.flatten()
p_test=np.array([p_test])

wai_orig=ai.predict(p_test)
wai=np.concatenate(wai_orig)
wai=np.reshape(wai, (nxs,nxs))


fig, axs = plt.subplots(3,3, sharex=True)
axs[0,0].contourf(px,py,p/np.mean(p),levels=15)
axs[0,0].set_xlabel('r')
axs[0,0].set_ylabel('phi')
   
axs[1,0].contourf(xs,ys,w/np.mean(w),levels=15)
axs[1,0].set_xlabel('X')
axs[1,0].set_ylabel('Y')
axs[1,0].axis('equal')

axs[2,0].contourf(xs,ys,wai/np.mean(w),levels=15)
axs[2,0].set_xlabel('X')
axs[2,0].set_ylabel('Y')
axs[2,0].axis('equal')


#squeezed light
vx=0.2
vy=1/vx
w=np.exp(-xs**2/2./vx-ys**2/2./vy)
p=generatePofw(w,lxs,phispace)

p_test=p.flatten()
p_test=np.array([p_test])

wai_orig=ai.predict(p_test)
wai=np.concatenate(wai_orig)
wai=np.reshape(wai, (nxs,nxs))

axs[0,1].contourf(px,py,p/np.mean(p),levels=15)
axs[0,1].set_xlabel('r')
axs[0,1].set_ylabel('phi')
   
axs[1,1].contourf(xs,ys,w/np.mean(w),levels=15)
axs[1,1].set_xlabel('X')
axs[1,1].set_ylabel('Y')

axs[2,1].contourf(xs,ys,wai/np.mean(wai),levels=15)
axs[2,1].set_xlabel('X')
axs[2,1].set_ylabel('Y')

#squares
w=sk.io.imread('./images/square60x60.png',as_gray=True)
p=generatePofw(w,lxs,phispace)

p_test=p.flatten()
p_test=np.array([p_test])

wai_orig=ai.predict(p_test)
wai=np.concatenate(wai_orig)
wai=np.reshape(wai, (nxs,nxs))

axs[0,2].contourf(px,py,p/np.mean(p),levels=15)
axs[0,2].set_xlabel('r')
axs[0,2].set_ylabel('phi')
   
axs[1,2].contourf(xs,ys,w/np.mean(w),levels=15)
axs[1,2].set_xlabel('X')
axs[1,2].set_ylabel('Y')

axs[2,2].contourf(xs,ys,wai/np.mean(wai),levels=15)
axs[2,2].set_xlabel('X')
axs[2,2].set_ylabel('Y')


plt.show()
