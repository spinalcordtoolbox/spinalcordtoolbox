function f = evalrbf2d(params,x,y)

% function f = evalrbf2d(params,x,y)
%
% <params> is [cx cy b g d] where
%   <cx>,<cy> are the x- and y-coordinates
%   <b> is the bandwidth in [0,Inf)
%   <g> is the gain
%   <d> is the offset
% <x>,<y> are matrices containing x- and y-coordinates to evaluate at.
%   you can omit <y> in which case we assume the first row
%   of <x> contains x-coordinates and the second row contains
%   y-coordinates.
%
% evaluate the 2D RBF at <x> and <y>.
% 
% example:
% [xx,yy] = meshgrid(0:.01:1,0:.01:1);
% zz = evalrbf2d([.4 .6 1 1 0],xx,yy);
% figure; contour(xx,yy,zz); colorbar;

% input
if ~exist('y','var')
  y = x(2,:);
  x = x(1,:);
end
cx = params(1);
cy = params(2);
b = params(3);
g = params(4);
d = params(5);

% do it
f = g * exp(-b*((x-cx).^2+(y-cy).^2)) + d;
