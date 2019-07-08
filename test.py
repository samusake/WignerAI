# -*- coding: utf-8 -*-
import numpy as np
from WigToDensity import *
from genSample import *
import matplotlib as mpl

N=5
nphi=201#45 #number of angleSteps

nxs=201
xmax=5
lxs=np.linspace(-xmax, xmax, nxs)
[xs, ys]=np.meshgrid(lxs,lxs);

phispace=np.linspace(0,2*np.pi,nphi, endpoint=False)
[px, py]=np.meshgrid(lxs,phispace)

[xpcol,ypcol]=np.meshgrid(np.linspace(1,N+1,N+1),np.linspace(1,N+1,N+1))
'''
rho=randomDensityMatrix(N)


plt.figure(1)
plt.pcolor(xpcol, ypcol, np.real(rho), norm=mpl.colors.Normalize(vmin=-1.,vmax=1.))
plt.figure(2)
plt.pcolor(xpcol, ypcol, np.imag(rho), norm=mpl.colors.Normalize(vmin=-1.,vmax=1.))


w=np.zeros((nxs,nxs))
for i in range(0,N):
    for j in range(0,N):
        w=w+rho[i,j]*wnm(j,i,lxs)

fig=plt.figure(3)
ax=fig.add_subplot(111,projection='3d')
ax.plot_surface(xs,ys,np.real(w), rstride=1, cstride=1, cmap='viridis',edgecolor='none')


rho2=rho1modeps(w, phispace, lxs, N-1)
plt.figure(4)
plt.pcolor(xpcol, ypcol, np.real(rho2), norm=mpl.colors.Normalize(vmin=-1.,vmax=1.))
plt.figure(5)
plt.pcolor(xpcol, ypcol, np.imag(rho2), norm=mpl.colors.Normalize(vmin=-1.,vmax=1.))
'''
#%%
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')

w=wnm(3,2,lxs)
#w=wnm(1,1,lxs)
ax.plot_surface(xs,ys,np.real(w), rstride=1, cstride=1, cmap='viridis',edgecolor='none')

[xpcol,ypcol]=np.meshgrid(np.linspace(1,N+1,N+1),np.linspace(1,N+1,N+1))
fig,axs=plt.subplots()
rho=rho1modeps(w, phispace, lxs, N)
plt.pcolor(xpcol,ypcol, np.real(rho))

fig,axs=plt.subplots()
plt.pcolor(xpcol, ypcol, np.imag(rho))

