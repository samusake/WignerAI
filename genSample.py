 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import factorial
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from scipy.ndimage.interpolation import shift

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as sk

import time
#%%

#https://arxiv.org/pdf/1811.06654.pdf
def randomDensityMatrix(dim): #http://www.qetlab.com/RandomDensityMatrix
    rho=np.random.rand(dim,dim) #https://arxiv.org/pdf/1010.3570.pdf
    rho.astype(complex)
    rho=rho + 1j*np.random.rand(dim,dim)
    rho=rho*np.transpose(rho) #hermitian + positiv semidefinite
    rho=rho/np.trace(rho) #trace one
    return(rho)


def wnm(n,m,xs):
    xs=xs.astype(complex)
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
    return(ws)

def randomWignerMatrix(N, xaxis):
    nxs=len(xaxis)
    rho=randomDensityMatrix(N)
    w=np.zeros((nxs,nxs))
    for i in range(0,N):
        for j in range(0,N):
            w=w+rho[i,j]*wnm(i,j,xaxis)
    w=np.real(w)
    return(w)

def randomSqueezedWigner(xs,ys):
    vx=np.random.random()*0.8+0.1
    vy=1./vx
    return(np.exp(-xs**2/2./vx-ys**2/2./vy))    
    
def probdist(w,phi):#w-wigner-function-matrix, phi-rotationangle
    w2=sk.transform.rotate(w,-phi,resize=False);
    probdist=np.sum(w2,0);
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

def generatePofw(w, xaxis, phispace):
    nxs=len(xaxis)
    nphi=len(phispace)
    P=np.zeros((nphi,nxs))
    k=0
    for i in phispace:
        P[k]=probdist(w,i)
        k=k+1
    return(P)

def generateP_radonofw(w,xaxis,phispace):
    nxs=len(xaxis)
    nphi=len(phispace)
    P=np.zeros((nphi,nxs))
    k=0
    for i in phispace:
        w2=sk.transform.rotate(w,-i,resize=False);
        P[k]=np.sum(w2,0)
        k=k+1
    return(np.transpose(P))    

def generateDataset(N,s,phispace,xaxis): #N-Dimension of rho, s-Number of samples, nphi-angular stepsize
    nxs=len(xaxis)   
    nphi=len(phispace)
    W=np.zeros((s,nxs,nxs))
    P=np.zeros((s,nphi,nxs))
    for i in range(0, s):
        if i%100==0:
            k=i/s*100
            print("{0} %".format(k))
        if N==-1:
            W[i]=randomWignerMatrix(np.random.randint(2,7),xaxis)
        else:
            W[i]=randomWignerMatrix(N,xaxis)
        W[i]=sk.transform.rotate(W[i],np.random.randint(0,360),resize=False);
        P[i]=generatePofw(W[i],xaxis,phispace)
    return((P,W))

def generateDatasetWithShift(N,s,phispace,xaxis): #N-Dimension of rho, s-Number of samples, nphi-angular stepsize
    nxs=len(xaxis)
    nphi=len(phispace)
    W=np.zeros((s,nxs,nxs))
    P=np.zeros((s,nphi,nxs))
    for i in range(0, s):
        if i%100==0:
            k=i/s*100
            print("{0} %".format(k))
        if N==-1:
            W[i]=randomWignerMatrix(np.random.randint(2,7),xaxis)
        else:
            W[i]=randomWignerMatrix(N,xaxis)
        W[i]=sk.transform.rotate(W[i],np.random.randint(0,360),resize=False);
        W[i]=shift(W[i],shift=(np.random.randint(-nxs/6,nxs/6),np.random.randint(-nxs/6,nxs/6)))
        P[i]=generatePofw(W[i],xaxis,phispace)
    return((P,W))
def generateDatasetWithShiftAndSqueezed(N,s,phispace,xaxis):
    nxs=len(xaxis)    
    nphi=len(phispace)
    W=np.zeros((s,nxs,nxs))
    P=np.zeros((s,nphi,nxs))
    [xs, ys]=np.meshgrid(xaxis,xaxis);
    for i in range(0, s):
        if i%100==0:
            k=i/s*100
            print("{0} %".format(k))
        if N==-1:
            if(np.random.randint(11)>3):
                W[i]=randomWignerMatrix(np.random.randint(2,7),xaxis)
            else:
                W[i]=randomSqueezedWigner(xs,ys)
        else:
            W[i]=randomWignerMatrix(N,xaxis)
        W[i]=sk.transform.rotate(W[i],np.random.randint(0,360),resize=False);
        W[i]=shift(W[i],shift=(np.random.randint(-nxs/6,nxs/6),np.random.randint(-nxs/6,nxs/6)))
        P[i]=generatePofw(W[i],xaxis,phispace)
    return((P,W))
    
def generateCompleteRandomDataset(s,phispace,xaxis):
    nxs=len(xaxis)    
    nphi=len(phispace)
    W=np.random.rand(s,nxs,nxs)*4-2
    P=np.zeros((s,nphi,nxs))
    for k in range(0,s):
        P[k]=generatePofw(W[k],xaxis,phispace)
    return((P,W))
#%%
#%%
'''
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')


nxs=51
nphi=51
xmax=5
lxs=np.linspace(-xmax, xmax, nxs)
[xs, ys]=np.meshgrid(lxs,lxs);

phispace=np.linspace(0,180,nphi, endpoint=False)
[px, py]=np.meshgrid(lxs,phispace)

w=wnm(0,0,lxs)+wnm(1,0,lxs)+wnm(0,1,lxs)+wnm(1,1,lxs)
w=np.real(w)
#w=wnm(1,1,lxs)
ax.plot_surface(xs,ys,np.real(w), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
#%%
probdx=probdist(np.real(w),0)
xaxis2, probdx2=equispacedPoints(probdx,lxs,20)
plt.plot(lxs, probdx) 
plt.plot(xaxis2, probdx2)


plt.show()
#%%
cumdx=cumulant(probdx)
plt.plot(lxs, cumdx)
plt.show()

#%%
histsample=sample(cumdx,lxs,100)
plt.plot(lxs,histsample,lxs,probdx)
plt.show()
#%%

w=wnm(0,0,lxs)+wnm(1,0,lxs)+wnm(0,1,lxs)+wnm(1,1,lxs)
w=np.real(w)
P=generatePofw(w, lxs, phispace)
plt.plot(lxs, P[0])
plt.show()
#%%

fig, axs = plt.subplots()
axs.contourf(px,py,P,levels=15)
'''
'''
#%%
N=2
w=randomWignerMatrix(N,lxs)

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.plot_surface(xs,ys,np.real(w), rstride=1, cstride=1, cmap='viridis', edgecolor='none')


#%%
P=generatePofw(w, lxs, phispace)
plt.plot(lxs, P[3])
plt.show()

#%%
N=2 #dimension of rho
s=10 #number of samples
nphi=4 #number of angleSteps

nxs=51
xmax=5
lxs=np.linspace(-xmax, xmax, nxs)
[xs, ys]=np.meshgrid(lxs,lxs);

P, W=generateDataset(N,s,phispace,lxs)

start=time.time()
P, W=generateDataset(N,s,phispace,lxs)
end=time.time()

print("elapsed with numba-nopython=True = %s" % (end-start))
#%%




'''











        
