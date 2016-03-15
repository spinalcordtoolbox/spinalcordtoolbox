function f = evalorientedgaussian2d(params,x,y,ors)

% function f = evalorientedgaussian2d(params,x,y,ors)
%
% <params> is [mx my ang sd1 sd2 orsd g d] where
%   <mx>,<my> is the center
%   <ang> is the orientation in [0,2*pi).  0 means horizontal.
%   <sd1> is the standard deviation along the major axis (> 0)
%   <sd2> is the standard deviation along the minor axis (> 0)
%   <orsd> is the orientation width (>= 0) (i.e. the k in the von Mises distribution)
%   <g> is the gain (> 0)
%   <d> is the offset
% <x>,<y> are matrices containing x- and y-coordinates to evaluate at.
%   you can omit <y> (or pass it as []) in which case we assume the first row
%   of <x> contains x-coordinates and the second row contains
%   y-coordinates.
% <ors> (optional) is a vector of orientations to draw images at.
%   default: linspacecircular(0,pi,8).
%
% evaluate the oriented 2D Gaussian at <x> and <y>.  we concatenate the images
% for different orientations along the next dimension (beyond the dimensions of <x>).
%
% example:
% [xx,yy] = meshgrid(0:.01:1,0:.01:1);
% zz = evalorientedgaussian2d([.5 .5 pi/4 .3 .1 2 1 1],xx,yy);
% figure; imagesc(makeimagestack(zz,[],[],-1),[0 2]); axis equal tight; set(gca,'YDir','normal');

% input
if ~exist('y','var') || isempty(y)
  y = x(2,:);
  x = x(1,:);
end
if ~exist('ors','var') || isempty(ors)
  ors = linspacecircular(0,pi,8);
end

mx = params(1);
my = params(2);
ang = params(3);
sd1 = params(4);
sd2 = params(5);
orsd = params(6);
g = params(7);
d = params(8);

  % see evalgabor2d.m
coord = [cos(ang) sin(ang); -sin(ang) cos(ang)]*[flatten(x-mx); flatten(y-my)];
f = g * bsxfun(@times,exp(-1/2 * (coord(1,:).^2/(sd1^2) + coord(2,:).^2/(sd2^2))), ...
               vflatten(exp(orsd*cos(2*mod(ors-ang,pi)))/exp(orsd))) + d;  % could speed up the sd1==sd2 case...
f = reshape(f',[size(x) length(ors)]);
