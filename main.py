#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 19:34:08 2019

@author: herbert
"""

import numpy as np
from scipy.special import factorial
from scipy.misc import imrotate
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as sk

from numba import jit
import time

from genSample import *
#from model import *

N=2 #dimension of rho
s=1000 #number of samples
nphi=4 #number of angleSteps

nxs=51
xmax=5
lxs=np.linspace(-xmax, xmax, nxs)
[xs, ys]=np.meshgrid(lxs,lxs);

P, W=generateDataset(N,s,nphi,lxs)
