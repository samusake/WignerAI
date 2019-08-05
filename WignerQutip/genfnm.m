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
