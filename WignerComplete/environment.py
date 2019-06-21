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
        
        self.dxd2=(self.lxs[1]-self.lxs[0])/2.
    def step_right(self):
        self.phi=self.phi+(np.random.random_sample()+0.5)*self.phistep
        self.it=self.it+1
    def step_left(self):
        self.phi=self.phi-(np.random.random_sample()+0.5)*self.phistep
        self.it=self.it+1
    def setW(self):
        #self.w=wnm(0,0,self.lxs)+wnm(1,0,self.lxs)+wnm(0,1,self.lxs)+wnm(1,1,self.lxs)
        #self.w=np.real(self.w)
        N=2
        rho=randomDensityMatrix(N)
        self.w=np.zeros((self.nxs,self.nxs))
        for i in range(0,N):
            for j in range(0,N):
                self.w=self.w+rho[i,j]*wnm(i,j,self.lxs)
        self.w=np.real(self.w)
        randrot=np.random.randint(0,360)
        self.w=sk.transform.rotate(self.w,randrot,resize=False);
    def setNewRandomW(self, N, lowscalexaxis):
        rho=randomDensityMatrix(N)
        self.w=np.zeros((self.nxs,self.nxs))
        for i in range(0,N):
            for j in range(0,N):
                self.w=self.w+rho[i,j]*wnm(i,j,self.lxs)
        self.w=np.real(self.w)
        randrot=np.random.randint(0,360)
        self.w=sk.transform.rotate(self.w,randrot,resize=False);
        
        lowscalenxs=len(lowscalexaxis)
        W=np.zeros((lowscalenxs,lowscalenxs))
        for i in range(0,N):
            for j in range(0,N):
                W=W+rho[i,j]*wnm(i,j,lowscalexaxis)
        W=np.real(W)
        W=sk.transform.rotate(W,randrot,resize=False);
        return(W)
 
    def setNewRandomWwithP(self, N, lowscalexaxis, lowscalephispace):
        rho=randomDensityMatrix(N)
        self.w=np.zeros((self.nxs,self.nxs))
        for i in range(0,N):
            for j in range(0,N):
                self.w=self.w+rho[i,j]*wnm(i,j,self.lxs)
        self.w=np.real(self.w)
        randrot=np.random.randint(0,360)
        self.w=sk.transform.rotate(self.w,randrot,resize=False);
        
        lowscalenxs=len(lowscalexaxis)
        W=np.zeros((lowscalenxs,lowscalenxs))
        for i in range(0,N):
            for j in range(0,N):
                W=W+rho[i,j]*wnm(i,j,lowscalexaxis)
        W=np.real(W)
        W=sk.transform.rotate(W,randrot,resize=False);
        P=generatePofw(W,lowscalexaxis,lowscalephispace)
        return(W,P) 
        
    def measure(self):
        probdx=probdist(self.w,self.phi)
        cumdx=cumulant(probdx)
        f=interp1d(cumdx, self.lxs)
        randsample=f(np.random.random_sample())+self.dxd2
        self.it=self.it+1
        return(self.phi,randsample)

def randomWalk(maxit, myenv):
    Dataset=[]
    myenv.it=0
    while myenv.it < maxit:
        if np.random.randint(2)==0:
            myenv.step_right()
        else:
            myenv.step_left()
        Dataset.append(myenv.measure())
    return(Dataset)
    
def generatePlainDataset(N, s, xaxis, myenv, Ndata):
    nxs=len(xaxis)   
    W=np.zeros((s,nxs,nxs))
    D=np.zeros((s, int(Ndata/2), 2))
    for i in range(0, s):
        if i%100==0:
            k=i/s*100
            print("{0} %".format(k))
        W[i]=myenv.setNewRandomW(np.random.randint(2,7),xaxis, withP)
        D[i]=randomWalk(Ndata,myenv)
    return((D,W))
    
def generatePlainDatasetwithP(N, s, xaxis, myenv, Ndata, phispace):
    nxs=len(xaxis)
    nphi=len(phispace)
    W=np.zeros((s,nxs,nxs))
    D=np.zeros((s, int(Ndata/2), 2))
    P=np.zeros((s,nphi,nxs))
    for i in range(0, s):
        if i%100==0:
            k=i/s*100
            print("{0} %".format(k))
        W[i], P[i]=myenv.setNewRandomWwithP(np.random.randint(2,7),xaxis, phispace)
        D[i]=randomWalk(Ndata,myenv)
    return((D,W,P))
#%%
'''
myenv=env(100,100)

nphi=12
nxs=12
xmax=5
lxs=np.linspace(-xmax, xmax, nxs)
[xs, ys]=np.meshgrid(lxs,lxs);

phispace=np.linspace(0,180,nphi, endpoint=False)
[px, py]=np.meshgrid(lxs,phispace)

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
P=generatePofw(w, lxs, phispace)

axs2.contourf(px,py,P)
axs2.set_ylabel('phi')

'''
