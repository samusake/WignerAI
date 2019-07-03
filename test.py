# -*- coding: utf-8 -*-

import numpy as np
from WigToDensity import *
from genSample import *

N=10
nphi=61#45 #number of angleSteps

nxs=61
xmax=5
lxs=np.linspace(-xmax, xmax, nxs)
[xs, ys]=np.meshgrid(lxs,lxs);

phispace=np.linspace(0,180,nphi, endpoint=False)
[px, py]=np.meshgrid(lxs,phispace)


rho=randomDensityMatrix(N)

w=np.zeros((nxs,nxs))
for i in range(0,N):
    for j in range(0,N):
        w=w+rho[i,j]*wnm(i,j,lxs)
        
rho2=rho1modeps(w, phispace, lxs, N-1)