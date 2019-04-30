#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:09:10 2019

@author: samu
"""
import numpy as np
from scipy.special import factorial
from scipy.misc import imrotate
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as sk


def wnm(n,m,xs):
    xs.astype(complex)
    if np.size(xs[0])>np.size(xs[1]):
        xs=xs.transpose()

    #horner Method
    [qs,ps]=np.meshgrid(xs,xs)
    Lag=np.ones((len(xs),len(xs)),dtype=complex)
    tmp=np.array(1)
    qp2=np.power(qs,2)+np.power(ps,2)
    qjp=qs-1j*ps
    for cnt in range(n,0,-1): #n, n-1, .., 1.
        tmp=tmp*(m-n+cnt)/(n+1-cnt)
        Lag=tmp-2*qp2*Lag/cnt
    p1=((-1.)**float(n))/np.pi
    p2=np.sqrt(factorial(n)/factorial(m)*2**(m-n))
    p3=np.power(qjp,m-n)
    p41=np.transpose(np.exp(-np.power([xs],2)))
    p42=np.exp(-np.power([xs],2))
    p4=p41*p42
        
    ws=p1 * p2 * p3 * p4 * Lag
        
    if n==m:
        w0=1./np.pi*np.power(-1.,n)
    else:
        w0=0.;
    for i in range(0,np.shape(ws)[1]):
        for j in range(0,np.shape(ws)[0]):
            if np.isnan(ws[j,i]):
                ws[j,i]=w0
    return(np.real(ws))

def probdist(w,phi):#w-wigner-function-matrix, phi-rotationangle
    w2=sk.transform.rotate(w,phi,resize=False);
    probdist=np.sum(w2,1);
    probdist=probdist/np.sum(probdist)
    return(probdist)

def equispacedPoints(prob, xaxis, N ):
    f=interp1d(xaxis,prob)
    xaxis2=np.linspace(xaxis[0], xaxis[-1], N)
    prob2=f(xaxis2)
    return((xaxis2,prob2))
    
def cumulant(probd):
    return(np.cumsum(probd))

def sample(cum,xaxis,N):
    dxd2=(xaxis[1]-xaxis[0])/2.
    randn=np.random.random_sample(N)
    f=interp1d(cum, xaxis)
    randsample=f(randn)+dxd2
    hist=np.histogram(randsample,  bins=np.linspace(xaxis[0]-dxd2, xaxis[-1]+dxd2, len(xaxis)+1))[0]
    hist=hist/np.sum(hist)
    return(hist)

def prepareExpData(expdata, nDatapoints): 
    #ToDo:gaussian_kde(x)
    #use kerneldensity estimation to get probabillity distribution to get X equispaced Datapoints
    return(0)



    
#%%

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')


nxs=101
xmax=5
lxs=np.linspace(-xmax, xmax, nxs)
[xs, ys]=np.meshgrid(lxs,lxs);

w=wnm(0,0,lxs)+wnm(1,0,lxs)+wnm(0,1,lxs)+wnm(1,1,lxs)
#w=wnm(1,1,lxs)
ax.plot_surface(xs,ys,np.real(w), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
#%%
probdx=probdist(w,0)
xaxis2, probdx2=equispacedPoints(probdx,lxs,20)
plt.plot(lxs, probdx, xaxis2, probdx2)


plt.show()
#%%
cumdx=cumulant(probdx)
plt.plot(lxs, cumdx)
plt.show()

#%%
histsample=sample(cumdx,lxs,100000)
plt.plot(lxs,histsample,lxs,probdx)
plt.show()
#%%


































        