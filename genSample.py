#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:09:10 2019

@author: samu
"""
import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def wnm(n,m,xs):
    xs.astype(complex)
    if np.size(xs[0])>np.size(xs[1]):
        xs=xs.transpose()
    if len(xs)==2:
        qs=xs[0]
        ps=xs[1]
        Lag=np.array(1)
        tmp=np.array(1)
        qp2=np.power(qs,2)+np.power(ps,2)
        qjp=qs-1j*ps
        for cnt in range(n,0,-1): #n, n-1, .., 1.
            tmp=tmp*(m-n+cnt)/(n+1-cnt)
            Lag=tmp-2*qp2*Lag/cnt
        ws=np.power(-1.,n)/np.pi*np.sqrt(factorial(n)/factorial(m)*np.power(2,m-n))*np.power(qjp,m-n)*(np.transpose(np.exp(-np.power(qs,2)))*np.exp(-np.power(ps,2)))*Lag
    else:
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
        p3=np.power(qjp,(m-n))
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
    return(ws)

def sample()

#%%

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')


nxs=31
xmax=5
lxs=np.linspace(-xmax, xmax, nxs)
[xs, ys]=np.meshgrid(lxs,lxs);

w=wnm(0,0,lxs)+wnm(1,0,lxs)+wnm(0,1,lxs)+wnm(1,1,lxs)
#w=wnm(1,1,lxs)
ax.plot_surface(xs,ys,np.real(w), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
#%%








































        