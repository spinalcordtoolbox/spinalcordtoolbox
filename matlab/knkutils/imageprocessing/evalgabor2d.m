function f = evalgabor2d(params,x,y)

% function f = evalgabor2d(params,x,y)
%
% <params> is [cpu ang phase mx my sd g d] where
%   <cpu> is the number of cycles per unit
%   <ang> is the orientation in [0,2*pi).  0 means a horizontal grating.
%   <phase> is the phase in [0,2*pi)
%   <mx>,<my> is the center
%   <sd> is the standard deviation of the Gaussian envelope (assumed isotropic)
%   <g> is the gain in [0,Inf) (1 means the total peak-to-trough distance is 2)
%   <d> is the offset
%     OR [cpu ang phase mx my sd1 sd2 g d] where
%   <sd1> is the standard deviation along the major axis
%   <sd2> is the standard deviation along the minor axis
% <x>,<y> are matrices containing x- and y-coordinates to evaluate at.
%   you can omit <y> in which case we assume the first row
%   of <x> contains x-coordinates and the second row contains
%   y-coordinates.
%
% evaluate the 2D Gabor at <x> and <y>.
% 
% example:
% [xx,yy] = meshgrid(0:.01:1,0:.01:1);
% zz = evalgabor2d([4 pi/6 0 .5 .5 .1 1 0],xx,yy);
% figure; contour(xx,yy,zz); colorbar;

% input
if ~exist('y','var')
  y = x(2,:);
  x = x(1,:);
end
if length(params)==8
  cpu = params(1);
  ang = params(2);
  phase = params(3);
  mx = params(4);
  my = params(5);
  sd = params(6);
  g = params(7);
  d = params(8);
else
  cpu = params(1);
  ang = params(2);
  phase = params(3);
  mx = params(4);
  my = params(5);
  sd1 = params(6);
  sd2 = params(7);
  g = params(8);
  d = params(9);
end

% do it
if length(params)==8
  f = g * exp( ((x-mx).^2+(y-my).^2)/-(2*sd^2) ) .* ...
      cos(2*pi*cpu*(-sin(ang)*(x-mx) + cos(ang)*(y-my)) + phase) + d;
else
  % the base case is a Gaussian aligned with coordinate axes.
  % we stipulate that positive orientation means to rotate the Gaussian
  % CCW.  so, for a given Gaussian, to figure out the values, we first
  % undo the rotation and then sample from the base case.  to undo CCW,
  % we just have to rotate CW, which is like
  % [x' y']' = [cos ang  sin ang; -sin ang  cos ang] [x y]'.
  coord = [cos(ang) sin(ang); -sin(ang) cos(ang)]*[flatten(x-mx); flatten(y-my)];
  f = g * exp(-1/2 * (coord(1,:).^2/(sd1^2) + coord(2,:).^2/(sd2^2))) .* ...
      cos(2*pi*cpu*coord(2,:) + phase) + d;
  f = reshape(f,size(x));
end
