# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


from skimage.transform import iradon, iradon_sart
from skimage.transform import radon, rescale
from scipy.special import erf

def genfnm(xs,nmax):
    Nsamp=len(xs)
    mmax=4*nmax
    a4nminone=np.sqrt(2*(mmax-1)+1)
    a4n=np.sqrt(2*mmax+1)
    xmax=a4n-np.power(2*a4n,-1/3)
    
    f=np.zeros((nmax,nmax));
    psi=np.zeros((nmax+1));
    phi=np.zeros((mmax+1))#(mmax-1));
    fnm=np.zeros((nmax,nmax,Nsamp));
    for cnts in range(0,Nsamp): #1:Nsamp
        x=xs[cnts]

        psi[0]=np.power(np.pi,-1/4)*np.exp(-(x**2)/2)
        psi[1]=np.sqrt(2)*x*psi[0]
        for cnt in range(2,nmax+1):#3:(nmax+1)
            n=cnt
            psi[cnt]=np.sqrt(1/n) * ( np.sqrt(2) * x * psi[cnt-1] - np.sqrt(n-1) * psi[cnt-2] )

        if np.abs(x)<xmax:
           t=np.arccos(x/a4n)
           phi[mmax]=np.sqrt( 2 /(np.pi*a4n*np.sin(t)) ) * np.sin( a4n**2/4 * ( np.sin(2*t) - 2*t ) + np.pi/4)
           t=np.arccos(x/a4nminone);
           phi[mmax-1] = np.sqrt( 2 /(np.pi*a4nminone*np.sin(t)) ) * np.sin( a4nminone**2/4 * ( np.sin(2*t) - 2*t ) + np.pi/4)
           for cnt in range(mmax-2,-1,-1):#(mmax-1):-1:1
               n=cnt #cnt-1
               phi[cnt]=1/np.sqrt(n+1)*(np.sqrt(2)*x*phi[cnt+1] -np.sqrt(n+2)*phi[cnt+2])

        else:
           phi[0]=np.power(np.pi,-3/4)/x*np.exp(x**2/2)
           for cnt in range(1,nmax+1):#2:(nmax+1)
               n=cnt
               phi[cnt]=sqrt(n/2)/x*phi[cnt-1]

        for cntn in range(0,nmax): #1:(nmax)
            n=cntn
            for cntm in range(cntn, nmax):#cntn:nmax
                m=cntm
                f[cntn,cntm]=2*x*psi[cntn]*phi[cntm] - np.sqrt(2*n+2)*psi[cntn+1]*phi[cntm] - np.sqrt(2*m+2)*psi[cntn]*phi[cntm+1]

        for cntn in range(1,nmax):# 2:(nmax)
            for cntm in range(0,cntn+1):#=1:cntn
                f[cntn,cntm]=f[cntm,cntn]

        fnm[:,:,cnts]=np.pi*f
    return(fnm)
def rho1modeps(ps, thetas, xs, nmax):
    nmp1=nmax+1
    Nangle=len(thetas)
    Noutofbounds=0
    # generate pattern functions
    fnm=genfnm(xs,nmp1)
    phase=np.zeros((2*nmax+1,Nangle))
    
    for cntt in range(0,Nangle):#1:Nangle
        phase[:,cntt]=np.exp(1j*(np.arange(-nmax,(nmax+1)))*thetas[cntt])
    
    F=np.zeros((Nangle,nmp1**2, Nangle))
    for m in range(0,nmp1):#=1:nmp1
        for n in range(0,nmp1):#=1:nmp1
            for cntt in range(0,Nangle):#1:Nangle
                #hääääh? hier weitermachens
                #F(cntt).fnm(n+(m-1)*nmp1,:)=fnm(n,m,:)*phase(nmp1+n-m,cntt);
                F[cntt,n+(m)*nmp1,:]=fnm[n,m,:]*phase[nmp1+n-m-1,cntt]
    #compute rho from joint probabilties and pattern functions see leonhardt's
    # papers
    tmp=np.zeros((1,nmp1**2))
    
    for cntt in range(0,Nangle):#1:Nangle
        ft=np.squeeze(F[cntt,:,:])#F(cntt).fnm);
        tmp=tmp+np.transpose(ps[:,cntt]).dot(np.transpose(ft))
    
    rholin=tmp/Nangle
    rho=np.reshape(rholin,(nmp1,nmp1))
    return(rho)

'''
xmax=5;
nxs=61;
nthetas=61;
 
xs =np.linspace(-xmax,xmax,nxs);
thetas=np.linspace(0,2*np.pi,nthetas, endpoint=False)

ps=np.zeros((nxs,nthetas));
r=1;
[mx,my]=np.meshgrid(xs,xs) 
[tx,ty]=np.meshgrid(thetas,thetas)

vs=np.sinh(r)*np.cos(2*thetas)+np.cosh(r);
plt.plot(vs)
#%%
for cnt in range(0,nthetas):   
    ps[:,cnt]=np.exp(-xs**2/vs[cnt])/(np.sum(np.exp(-xs**2/vs[cnt])));
fig, axs = plt.subplots()
axs.contourf(mx,my,ps,levels=15)
plt.show()
#%%

fig2, axs2 = plt.subplots()
irad=iradon(ps,theta=np.linspace(0,360,nthetas, endpoint=False))
axs2.contourf(irad,levels=15)
plt.show()
#%%
rho=rho1modeps(ps,thetas,xs,10);
plt.pcolor(np.real(rho))
plt.show()
'''