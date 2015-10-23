function Yout=smoothn_x(X,Y,Xeval,S,robust)
% smoothn_x(X,Y,Xeval,robust)
if X~=round(X), error('X should be integers'); end

Xbis=min(X):max(X);
Ybis=nan(1,length(Xbis));

Ybis(X)=Y;

if robust,robust='robust'; end
Yout=smoothn(Ybis,S,robust);
Yout=Yout(Xeval);
