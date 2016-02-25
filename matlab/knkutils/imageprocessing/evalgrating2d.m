function f = evalgrating2d(params,x,y)

% function f = evalgrating2d(params,x,y)
%
% <params> is [cpu ang phase g d] where
%   <cpu> is the number of cycles per unit
%   <ang> is the orientation in [0,2*pi).  0 means a horizontal grating.
%   <phase> is the phase in [0,2*pi)
%   <g> is the gain (1 means the total peak-to-trough distance is 2)
%   <d> is the offset
% <x>,<y> are matrices containing x- and y-coordinates to evaluate at.
%   you can omit <y> in which case we assume the first row
%   of <x> contains x-coordinates and the second row contains
%   y-coordinates.
%
% evaluate the 2D sinusoidal grating at <x> and <y>.
% 
% example:
% [xx,yy] = meshgrid(0:.01:1,0:.01:1);
% zz = evalgrating2d([5 pi/2 0 1 0],xx,yy);
% figure; contour(xx,yy,zz); colorbar;
% figure; plot(xx(51,:),zz(51,:));

% input
if ~exist('y','var')
  y = x(2,:);
  x = x(1,:);
end
cpu = params(1);
ang = params(2);
phase = params(3);
g = params(4);
d = params(5);

% do it
f = g*cos(2*pi*cpu*(-sin(ang)*x + cos(ang)*y) + phase) + d;
