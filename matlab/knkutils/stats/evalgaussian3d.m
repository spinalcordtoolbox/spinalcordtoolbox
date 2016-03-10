function f = evalgaussian3d(params,x,y,z)

% function f = evalgaussian3d(params,x,y,z)
%
% <params> is [mx my mz sx sy sz g d n] where
%   <mx>,<my>,<mz> is the mean
%   <sx>,<sy>,<sz> is the standard deviation
%   <g> is the gain
%   <d> is the offset
%   <n> (optional) is an exponent (> 0).  default: 1.
% <x>,<y>,<z> are matrices containing x-, y-, and z-coordinates to evaluate at.
%   you can omit <y> and <z> in which case we assume the first row
%   of <x> contains x-coordinates, the second row contains
%   y-coordinates, and the third row contains z-coordinates.
%
% evaluate the 3D Gaussian at <x>, <y>, and <z>.  note that this function
% allows only 3D Gaussians whose axes are aligned with the coordinate axes.
%
% the exponent works by defining an S-shaped function that maps from values between
% 0 and 1 to values between 0 and 1.  for example, <n>==2 means that values between
% 0 and 0.5 will be squared while values between 0.5 and 1 will be square-rooted.
% this will be done in such a way that 0 maps to 0 and 1 maps to 1.  the exponent
% is applied right before the final gain and offset (see code for details).
% the point of the exponent is to change the shape of the Gaussian.
%
% the FWHM is 2*sqrt(2*log(2))*<sx> along the x-dimension (and similarly for
% the y- and z-dimensions).  if you want a FWHM of X, you should set <sx>
% to X/(2*sqrt(2*log(2))).
%
% assuming <d> is 0 and <n> is 1, if you want the volume under the curve to be 1,
% you should set <g> to (1/(sx*sqrt(2*pi)))*(1/(sy*sqrt(2*pi)))*(1/(sz*sqrt(2*pi)))
% 
% example:
% [xx,yy,zz] = meshgrid(0:.01:1,0:.01:1,0:.01:1);
% ff = evalgaussian3d([.2 .3 .5 .2 .2 .2 1 0],xx,yy,zz);
% figure; imagesc(makeimagestack(ff));
% ff = evalgaussian3d([.2 .3 .5 .2 .2 .2 1 0 2],xx,yy,zz);
% figure; imagesc(makeimagestack(ff));

% input
if ~exist('y','var')
  y = x(2,:);
  z = x(3,:);
  x = x(1,:);
end
mx = params(1);
my = params(2);
mz = params(3);
sx = params(4);
sy = params(5);
sz = params(6);
g = params(7);
d = params(8);
if length(params)==9
  n = params(9);
else
  n = 1;
end

% do it
temp = exp( (x-mx).^2/-(2*sx^2) + (y-my).^2/-(2*sy^2) + (z-mz).^2/-(2*sz^2) );
if n ~= 1
  temp(temp <= 0.5) = temp(temp <= 0.5) .^ n * (0.5/0.5^n);
  temp(temp > 0.5) = (temp(temp > 0.5) - 0.5) .^ (1/n) * (0.5/0.5^(1/n)) + 0.5;
end
f = g*temp + d;
