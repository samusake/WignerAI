function ws=wnm(varargin)
if nargin==3
    [n,m,xs]=varargin{:};
else
    warning('MATLAB:AmbiguousInput','MOCK!!! wnm(n,m,xs)');
    n=4;
    m=4;
    xs=linspace(-5,5,3);
end;
if size(xs,1)>size(xs,2)
    xs=xs.';
end;


if length(xs)==2

    qs=xs(1);
    ps=xs(2);
    Lag=1;
    tmp=1;
    qp2=qs.^2+ps.^2;
    qjp=qs-j*ps;
    for cnt=n:-1:1
        tmp=tmp*(m-n+cnt)/(n+1-cnt);
        %    Lag=tmp-(2*qs.^2+2*ps.^2).*Lag/cnt;
        Lag=tmp-2*qp2.*Lag/cnt;
    end;
    ws=(-1)^n/pi*sqrt(factorial(n)/factorial(m)*2^(m-n))*...
        qjp.^(m-n).*(exp(-qs.^2)'*exp(-ps.^2)).*Lag;%*dx^2;
    %    (qs-j*ps).^(m-n).*exp(-qp2).*Lag*dx^2;

else
    % Horner method
    [qs,ps]=meshgrid(xs);
    Lag=ones(length(xs));
    tmp=1;
    qp2=qs.^2+ps.^2;
    qjp=qs-j*ps;
    for cnt=n:-1:1
        tmp=tmp*(m-n+cnt)/(n+1-cnt);
        %    Lag=tmp-(2*qs.^2+2*ps.^2).*Lag/cnt;
        Lag=tmp-2*qp2.*Lag/cnt;
    end;
    ws=(-1)^n/pi*sqrt(factorial(n)/factorial(m)*2^(m-n))*...
        qjp.^(m-n).*(exp(-xs.^2)'*exp(-xs.^2)).*Lag;%*dx^2;
    %    (qs-j*ps).^(m-n).*exp(-qp2).*Lag*dx^2;

end;

if n==m
    w0=1/pi*(-1)^n;
else
    w0=0;
end;
ws(isnan(ws))=w0;%*dx*dx;
%%%%%%%%%%%%%%%%%%%%%%
% symbolic method
% syms x p q
% qs2ps2=2*(qs.^2+ps.^2);
% ln=simplify(exp(x)*(x^(-m+n))/factorial(n)*diff(exp(-x)*x^(n+m-n),n));
% w=simplify(exp(-q^2-p^2) ...
%     *((q-i*p)^(m-n)) ...
%     *ln);
% f=1/pi*sqrt(factorial(n)/factorial(m)*2^(m-n))*(-1)^n *dx^2;
% ws=subs(w,{'q','p','x'},{qs,ps,qs2ps2})*f;
%%%%%%%%%%%%%%%%%%%%%%%%
% if length(xs)>1
%     dx=xs(2)-xs(1);
% else
%     dx=1;
% end;

