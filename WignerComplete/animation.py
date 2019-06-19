
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
from genSample import *
from environment import *

myenv=env(100,100)


meas=[]
for i in range(0,5000):
    meas.append(myenv.measure()[1])
hist=np.histogram(meas,  bins=np.linspace(myenv.lxs[0]-(myenv.lxs[1]-myenv.lxs[0])/2., myenv.lxs[-1]+(myenv.lxs[1]-myenv.lxs[0])/2., len(myenv.lxs)+1))[0]
hist=hist/np.sum(hist)

plt.plot(myenv.lxs,hist)
plt.show()


Ndata=1000
Dataset=randomWalk(Ndata,myenv)


fig, ax = plt.subplots()
ln, = plt.plot([], [], 'r')
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
xdata,ydata=[],[]

def init():
    return ln,

def anim(i):
    i=int(i)
    xdata=[0,(np.cos(Dataset[i][0]/180.*np.pi))]
    ydata=[0,(np.sin(Dataset[i][0]/180.*np.pi))]
    ln.set_data(xdata,ydata)

ani = FuncAnimation(fig, anim, frames=np.linspace(0,len(Dataset)-1,len(Dataset)), interval = len(Dataset)/10.,repeat=False,init_func=init, blit=False)
plt.show()
