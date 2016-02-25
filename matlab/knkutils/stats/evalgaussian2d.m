function f = evalgaussian2d(params,x,y)

% function f = evalgaussian2d(params,x,y)
%
% <params> is [mx my sx sy g d] where
%   <mx>,<my> is the mean
%   <sx>,<sy> is the standard deviation
%   <g> is the gain
%   <d> is the offset
% <x>,<y> are matrices containing x- and y-coordinates to evaluate at.
%   you can omit <y> in which case we assume the first row
%   of <x> contains x-coordinates and the second row contains
%   y-coordinates.
%
% evaluate the 2D Gaussian at <x> and <y>.  note that this function
% allows only 2D Gaussians whose axes are aligned with the coordinate axes.
%
% the FWHM is 2*sqrt(2*log(2))*<sx> along the x-dimension (and similarly for
% the y-dimension).  if you want a FWHM of X, you should set <sx>
% to X/(2*sqrt(2*log(2))).
%
% assuming <d> is 0, if you want the volume under the curve to be 1,
% you should set <g> to (1/(sx*sqrt(2*pi)))*(1/(sy*sqrt(2*pi)))
% 
% example:
% [xx,yy] = meshgrid(0:.01:1,0:.01:1);
% zz = evalgaussian2d([.5 .5 .1 .2 2 0],xx,yy);
% figure; contour(xx,yy,zz); colorbar;
% figure; plot(xx(51,:),zz(51,:));

% input
if ~exist('y','var')
  y = x(2,:);
  x = x(1,:);
end
mx = params(1);
my = params(2);
sx = params(3);
sy = params(4);
g = params(5);
d = params(6);

% handle equal std dev as a separate case for speed reasons
if sx==sy
  f = g*exp( ((x-mx).^2+(y-my).^2)/-(2*sx^2) ) + d;
else
  f = g*exp( (x-mx).^2/-(2*sx^2) + (y-my).^2/-(2*sy^2) ) + d;
end
