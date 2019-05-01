#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 19:34:08 2019

@author: herbert
"""

import numpy as np
from scipy.special import factorial
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as sk

from numba import jit
import time

from genSample import *
#from model import *

N=4 #dimension of rho
s=4 #number of samples
nphi=180 #number of angleSteps

nxs=101
xmax=5
lxs=np.linspace(-xmax, xmax, nxs)
[xs, ys]=np.meshgrid(lxs,lxs);

phispace=np.linspace(0,180,nphi)
[px, py]=np.meshgrid(lxs,phispace)

P, W=generateDataset(N,s,nphi,lxs)

#%%
fig=plt.figure(1)
ax=fig.add_subplot(111)

ax.contourf(px,py,P[3],levels=15)
ax.set_xlabel('u')
ax.set_ylabel('phi')

fig=plt.figure(2)
ax=fig.add_subplot(111)
ax.contourf(xs,ys,np.real(W[3]),levels=15)
ax.set_xlabel('X')
ax.set_ylabel('Y')