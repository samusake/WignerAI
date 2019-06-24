#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:07:19 2019

@author: herbert
"""
# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import factorial
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from scipy.ndimage.interpolation import shift

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as sk
from matplotlib.animation import FuncAnimation,FFMpegFileWriter

import sys
sys.path.append("../")

import time
import json

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from genSample import *
from environment import *
from model import *

N=-1 #dimension of rho
nphi=20#45 #number of angleSteps

nxs=20
xmax=5
lxs=np.linspace(-xmax, xmax, nxs)
[xs, ys]=np.meshgrid(lxs,lxs);

phispace=np.linspace(0,180,nphi, endpoint=False)
[px, py]=np.meshgrid(lxs,phispace)

#%%

json_file = open('../models/20/ai_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
ai = keras.models.model_from_json(loaded_model_json)
# load weights into new model
ai.load_weights("../models/20/ai_weights.h5")
print("Loaded model from disk")

#%%

myenv=env(100,100)

Ndata=100000
Dataset=randomWalk(Ndata,myenv)

phistep=180./nphi
philist=[[] for x in range(0,2*nphi)]
for i in Dataset:
    philist[int(round(i[0]/phistep))%int(2*nphi)].append(i[1])

fig, axs = plt.subplots()
P=np.zeros((nphi,nxs))
for i in range(0,nphi):
    P[i]=np.histogram(philist[i]+philist[-(len(philist)-i)],  bins=np.linspace(lxs[0]-(lxs[1]-lxs[0])/2., lxs[-1]+(lxs[1]-lxs[0])/2., nxs+1))[0]
    #maybe buggy
    P[i]=P[i]/np.sum(P[i])
axs.contourf(px,py,P)
axs.set_ylabel('phi')

fig2, axs2 = plt.subplots()
w=wnm(0,0,lxs)+wnm(1,0,lxs)+wnm(0,1,lxs)+wnm(1,1,lxs)
w=np.real(w)
P_real=generatePofw(w, lxs, phispace)

axs2.contourf(px,py,P)
axs2.set_ylabel('phi')

#%%
ptest_real=np.array([P_real.flatten()])
ptest=np.array([P.flatten()])

w_rec_real=np.reshape(ai.predict(ptest_real), (nxs,nxs))
w_rec=np.reshape(ai.predict(ptest), (nxs,nxs))


fig, axs = plt.subplots(2,2, sharex=True)
axs[0,0].contourf(px,py,P_real/np.mean(P_real),levels=15,extend='both')
axs[0,0].set_ylabel('phi')
   
axs[1,0].contourf(xs,ys,w_rec_real/np.mean(w_rec_real),levels=15,extend='both')
axs[1,0].set_xlabel('X')
axs[1,0].set_ylabel('Y')
axs[1,0].axis('equal')

axs[0,1].contourf(px,py,P/np.mean(P_real),levels=15,extend='both')
axs[0,1].set_ylabel('Y')

axs[1,1].contourf(xs,ys,w_rec/np.mean(w_rec_real),levels=15,extend='both')
axs[1,1].set_xlabel('X')
axs[1,1].set_ylabel('Y')
axs[1,1].axis('equal')

axs[0,0].title.set_text('Theoretical')
axs[0,1].title.set_text('Generated experimental Data')

plt.show()