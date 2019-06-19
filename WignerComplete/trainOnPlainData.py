# -*- coding: utf-8 -*-

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

import time
from genSample import *
from environment import *
from model import *

N=-1 #dimension of rho
s=1000 #number of samples

nxs=12
xmax=5
lxs=np.linspace(-xmax, xmax, nxs)
[xs, ys]=np.meshgrid(lxs,lxs);


myenv=env(60,60)
Ndata=1000
#%%
#D,W = generatePlainDataset(N, s, lxs, myenv, Ndata)
#np.save('data/D_s1000_nxs12_env6060', D)
#np.save('data/W_s1000_nxs12_env6060', W)

D=np.load('data/D_s1000_nxs12_env6060.npy')
W=np.load('data/W_s1000_nxs12_env6060.npy')
#%%
Dinput=np.reshape(D,(s,Ndata))
outputV=np.zeros((s,nxs*nxs))
for i in range(0, len(W)):
    outputV[i]=W[i].flatten()
#%%

ai=pureDataNN(nxs, Ndata)

ai.model.compile(optimizer=keras.optimizers.RMSprop(0.001),#tf.train.GradientDescentOptimizer(0.005),#optimizer=tf.train.AdamOptimizer(0.001),
    loss='mean_squared_error')

checkpoint = ModelCheckpoint('models/ai_checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history=ai.model.fit(Dinput, outputV, epochs=10, batch_size=32, verbose=1, validation_split=0.1, callbacks=callbacks_list)
ai.model.count_params()
#%%
ai.model.load_weights('models/ai_checkpoint.h5')
#%%

with open('models/ai_model.json', 'w') as json_file:
    json_file.write(ai.model.to_json())
ai.model.save_weights('models/ai_weights.h5')
with open('models/ai_history.json', 'w') as json_file:
    json.dump(history.history, json_file)
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
Wai_orig=ai.model.predict(testIn)
Wai=np.concatenate(Wai_orig)
Wai=np.reshape(Wai, (30,nxs,nxs))

fig, axs = plt.subplots(3, 6, sharex='col', sharey='row')

contour=np.linspace(-0.5,1,50)
for i in range(0,6):
    axs[0,i].contourf(px,py,Ptest[i],contour)
    axs[0,i].set_ylabel('phi')
    #axs[0,i].axis('equal')
   
    axs[1,i].contourf(xs,ys,Wtest[i],contour)
    axs[1,i].set_ylabel('Y')
    #axs[1,i].axis('equal')

    axs[2,i].contourf(xs,ys,Wai[i],contour)
    axs[2,i].set_xlabel('X')
    axs[2,i].set_ylabel('Y')
    #axs[2,i].axis('equal')
axs[0,3].set_title('Probability distributions')
axs[1,3].set_title('Theoretical wigner distribution')
axs[2,3].set_title('AI prediction of wigner distribution')

plt.show()
'''

#%%
