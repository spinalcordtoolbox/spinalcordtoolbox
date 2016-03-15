function f = evaldog2d(params,x,y)

% function f = evaldog2d(params,x,y)
%
% <params> is [mx my sd sdratio volratio g d] where
%   <mx>,<my> is the center
%   <sd> is the SD of the center Gaussian
%   <sdratio> is the ratio of the surround SD to the center SD
%   <volratio> is the ratio of the surround volume to the center volume
%   <g> is the gain in [0,Inf) (1 means the intrinsic height given by makedog2d.m)
%   <d> is the offset
% <x>,<y> are matrices containing x- and y-coordinates to evaluate at.
%   you can omit <y> in which case we assume the first row
%   of <x> contains x-coordinates and the second row contains
%   y-coordinates.
%
% evaluate the 2D Difference-of-Gaussians at <x> and <y>.
% 
% example:
% [xx,yy] = meshgrid(0:.01:1,0:.01:1);
% zz = evaldog2d([.5 .5 .1 2 2 1 0],xx,yy);
% figure; contour(xx,yy,zz); colorbar;

% input
if ~exist('y','var')
  y = x(2,:);
  x = x(1,:);
end
mx = params(1);
my = params(2);
sd = params(3);
sdratio = params(4);
volratio = params(5);
g = params(6);
d = params(7);

% do it
temp = (x-mx).^2 + (y-my).^2;
f = g * (exp(temp/-(2*sd^2)) - ((1/sdratio^2)*sqrt(volratio)) * exp(temp/-(2*(sd*sdratio)^2))) + d;
  % THIS COULD BE SPED UP IF WE WANTED TO...
