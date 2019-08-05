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
