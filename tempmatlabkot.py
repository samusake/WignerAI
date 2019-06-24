# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 21:50:42 2019

@author: Sandrine
"""

xmax=5;
nxs=61;
nthetas=61;
 
xs =linspace(-xmax,xmax,nxs);
thetas=linspace(0,2*pi,nthetas+1);
thetas(end)=[];
ps=zeros(nxs,nthetas);
r=1;
 
vs=sinh(r)*cos(2*thetas)+cosh(r);
plot(vs)
%
for cnt = 1:nthetas
   
    ps(:,cnt)=exp(-xs.^2/vs(cnt))./(sum(exp(-xs.^2/vs(cnt))));
end
surf(ps)
 
contour(iradon(ps,360/nthetas))
rho=rho1modeps(ps,thetas,xs,10);
pcolor(real(rho))

#%%
function [rho,no]=rho1modeps(varargin)
if nargin == 4
    ps=varargin{1};
    thetas=varargin{2};
    xs=varargin{3};
    nmax=varargin{4};
    if size(ps,2)>size(ps,1)
        warning('ps must be 2D nthetas x nqs');
    end
else
    disp('rho1modeps(ps(qs,thetas),thetas,xs,nmax)');  
    return
end;
nmp1=nmax+1;
Nangle=length(thetas);
Noutofbounds=0;

% % Fock-ProbDensity
% H0=0*xs+1;
% H1=(2*xs)/sqrt(2^1*factorial(1));
% H2=(4*xs.^2-2)/sqrt(2^2*factorial(2));
% H3=(8*xs.^3-12*xs)/sqrt(2^3*factorial(3));
% H4=(16*xs.^4-48*xs.^2+12)/sqrt(2^4*factorial(4));
% pna=1/sqrt(pi)*(H2).^2.*exp(-xs.^2 )*(xs(2)-xs(1));
% % pnb=1/sqrt(pi)*(H2).^2.*exp(-xs.^2 )*(xs(2)-xs(1));
% %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tic


%% generate pattern functions
fnm=genfnm(xs,nmp1);
phase=zeros(2*nmax+1,Nangle);
for cntt=1:Nangle
phase(:,cntt)=exp(1i*(-nmax:nmax)*thetas(cntt));
end;
for m=1:nmp1
    for n=1:nmp1
        for cntt=1:Nangle
        F(cntt).fnm(n+(m-1)*nmp1,:)=fnm(n,m,:)*phase(nmp1+n-m,cntt);
        end;
    end;
end;
%% compute rho from joint probabilties and pattern functions see leonhardt's
%% papers
tmp=zeros(1,nmp1^2);

for cntt=1:Nangle
    ft=squeeze(F(cntt).fnm);
    tmp=tmp+ps(:,cntt)'*ft.';
end;

rholin=tmp/Nangle;
rho=reshape(rholin,[nmp1 nmp1]);
%imagesc(real(rho))

#%%
function fnm=genfnm(xs,nmax)
Nsamp=length(xs);
%an=sqrt(2*nmax+1);

mmax=4*nmax;
a4nminone=sqrt(2*(mmax-1)+1);
a4n=sqrt(2*mmax+1);
xmax=a4n-power(2*a4n,-1/3);

f=zeros(nmax);
psi=zeros(1,nmax+1);
phi=zeros(1,mmax-1);
fnm=zeros(nmax,nmax,Nsamp);
for cnts=1:Nsamp
   
x=xs(cnts);

psi(1)=power(pi,-1/4)*exp(-(x^2)/2);
psi(2)=sqrt(2)*x*psi(1);
for cnt=3:(nmax+1)
    n=cnt-1;
    psi(cnt)=sqrt(1/n) * ( sqrt(2) * x * psi(cnt-1) - sqrt(n-1) * psi(cnt-2) );
end;

if abs(x)<xmax
   t=acos(x/a4n);
   phi(mmax+1)=sqrt( 2 /(pi*a4n*sin(t)) ) * sin( a4n^2/4 * ( sin(2*t) - 2*t ) + pi/4);
   t=acos(x/a4nminone);
   phi(mmax) = sqrt( 2 /(pi*a4nminone*sin(t)) ) * sin( a4nminone^2/4 * ( sin(2*t) - 2*t ) + pi/4);
   for cnt=(mmax-1):-1:1
       n=cnt-1;
       phi(cnt)=1/sqrt(n+1)*(sqrt(2)*x*phi(cnt+1) -sqrt(n+2)*phi(cnt+2));
   end;
else
   phi(1)=power(pi,-3/4)/x*exp(x^2/2);
   for cnt=2:(nmax+1)
       n=cnt-1;
       phi(cnt)=sqrt(n/2)/x*phi(cnt-1);

   end;
end;

% for cntn = 1:(nmax)
%     n=cntn-1;
%     for cntm=1:nmax
%         m=cntm-1;
%         gcnt=gcnt+1;
%         fnm(gcnt)=pi*(2*x*psi(cntn)*phi(cntm) - sqrt(2*n+2)*psi(cntn+1)*phi(cntm) - sqrt(2*m+2)*psi(cntn)*phi(cntm+1));
%     end;
% end;

for cntn = 1:(nmax)
    n=cntn-1;
    for cntm=cntn:nmax
        m=cntm-1;
        f(cntn,cntm)=2*x*psi(cntn)*phi(cntm) - sqrt(2*n+2)*psi(cntn+1)*phi(cntm) - sqrt(2*m+2)*psi(cntn)*phi(cntm+1);
    end;
end;
for cntn = 2:(nmax)
    for cntm=1:cntn
        f(cntn,cntm)=f(cntm,cntn);
    end;
end;

fnm(:,:,cnts)=pi*f;
end;
