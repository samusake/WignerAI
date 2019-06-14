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
import json

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint

from genSample import *
from model import *


def equispacer_interpol(xaxis, xdata, pdata):
    return(np.interp(xaxis,xdata,pdata))
    
#%%

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')


nxs=100
nphi=30
xmax=5
lxs=np.linspace(-xmax, xmax, nxs)
[xs, ys]=np.meshgrid(lxs,lxs)

phispace=np.linspace(0,180,nphi, endpoint=False)
[px, py]=np.meshgrid(lxs,phispace)

w=wnm(0,0,lxs)+wnm(1,0,lxs)+wnm(0,1,lxs)+wnm(1,1,lxs)
w=np.real(w)
#w=wnm(1,1,lxs)
ax.plot_surface(xs,ys,np.real(w), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
#%%
probdx=probdist(np.real(w),0)
xaxis2, probdx2=equispacedPoints(probdx,lxs,20)
plt.plot(lxs, probdx) 
plt.plot(xaxis2, probdx2)


plt.show()
#%%
cumdx=cumulant(probdx)
plt.plot(lxs, cumdx)
plt.show()

#%%
histsample=sample(cumdx,lxs,1000)

#%%
nxs2=31
lxs2=np.linspace(-xmax, xmax, nxs2)
[xs2, ys2]=np.meshgrid(lxs2,lxs2)

interpsample=equispacer_interpol(lxs2, lxs, histsample)

plt.plot(lxs,histsample,lxs,probdx, lxs2, interpsample)
plt.show()