nxs=101;
xmax=5;
lxs=linspace(-xmax,xmax,nxs);
[xs,ys]=meshgrid(lxs);
w=wnm(0,0,lxs)+wnm(1,0,lxs)+wnm(0,1,lxs)+wnm(1,1,lxs);
w=imrotate(real(w),0,'bilinear','crop');
vx=0.2;
vy=1/vx;
wg=exp(-xs.^2/2/vx-ys.^2/2/vy);
wg=wg./sum(sum(wg));
mesh(xs,ys,wg)
axis square
%plot(lxs,sum(w,2))
%%
prdx=sum(w,2);
prdx=prdx/sum(prdx);
size(prdx)
cprdx=cumsum(prdx);
plot(lxs,cprdx);
%%
N=100000;
r=rand(N,1);
r2=interp1(cprdx,lxs,r,'linear')+(lxs(2)-lxs(1))/2;
hr2=hist(r2,lxs)
hr2=hr2/sum(hr2);
plot(lxs,hr2,lxs,prdx)
%hist(r,1001)


