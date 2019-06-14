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

class env:
    def __init__(self,nxs,nphi):
        self.nxs=nxs
        self.nphi=nphi
        self.xmax=5
        self.lxs=np.linspace(-self.xmax, self.xmax, nxs)
        self.phispace=np.linspace(0,180,nphi, endpoint=False)
        
        self.w=wnm(0,0,self.lxs)+wnm(1,0,self.lxs)+wnm(0,1,self.lxs)+wnm(1,1,self.lxs)
        self.w=np.real(self.w)
        
        
        self.it=0
        self.phi=0
        self.phistep=6
    def step_right(self):
        self.phi=self.phi+(np.random.random_sample()+0.5)*self.phistep
        self.it=self.it+1
    def step_left(self):
        self.phi=self.phi-(np.random.random_sample()+0.5)*self.phistep
        self.it=self.it+1
    def measure(self):
        dxd2=(self.lxs[1]-self.lxs[0])/2.
        probdx=probdist(self.w,self.phi)
        cumdx=cumulant(probdx)
        f=interp1d(cumdx, self.lxs)
        randsample=f(np.random.random_sample())+dxd2
        self.it=self.it+1
        return(self.phi,randsample)

def randomWalk(maxit, myenv):
    Dataset=[]
    while myenv.it < maxit:
        if np.random.randint(2)==0:
            myenv.step_right()
        else:
            myenv.step_left()
        Dataset.append(myenv.measure())
    return(Dataset)
        
myenv=env(100,100)
'''
meas=[]
for i in range(0,5000):
    meas.append(myenv.measure()[1])
hist=np.histogram(meas,  bins=np.linspace(myenv.lxs[0]-(myenv.lxs[1]-myenv.lxs[0])/2., myenv.lxs[-1]+(myenv.lxs[1]-myenv.lxs[0])/2., len(myenv.lxs)+1))[0]
hist=hist/np.sum(hist)

plt.plot(myenv.lxs,hist)
plt.show()
'''

Dataset=randomWalk(100000,myenv)

'''
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

ani = FuncAnimation(fig, anim, frames=np.linspace(0,len(Dataset)-1,len(Dataset)), interval = len(Dataset)/4.,repeat=False,init_func=init, blit=False)
plt.show()
'''
nphi=12
nxs=12
xmax=5
lxs=np.linspace(-xmax, xmax, nxs)
[xs, ys]=np.meshgrid(lxs,lxs);

phispace=np.linspace(0,180,nphi, endpoint=False)
[px, py]=np.meshgrid(lxs,phispace)


phistep=360./nphi
philist=[[] for x in range(0,nphi)]
for i in Dataset:
    philist[int(round(i[0]/phistep))%nphi].append(i[1])
'''
P=np.zeros((nphi,nxs))
for i in range(0,len(philist)):
    P[i]=np.histogram(philist[i],  bins=np.linspace(lxs[0]-(lxs[1]-lxs[0])/2., lxs[-1]+(lxs[1]-lxs[0])/2., nxs+1))[0]
    P[i]=P[i]/np.sum(P[i])
'''
fig, axs = plt.subplots()
w=wnm(0,0,lxs)+wnm(1,0,lxs)+wnm(0,1,lxs)+wnm(1,1,lxs)
w=np.real(w)
P=generatePofw(w, lxs, phispace)

axs.contourf(px,py,P)
axs.set_ylabel('phi')


