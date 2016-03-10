function [f,xx,yy,zz] = makegaussian3d(matrixsize,mn,sd,xx,yy,zz,n)

% function [f,xx,yy,zz] = makegaussian3d(matrixsize,mn,sd,xx,yy,zz,n)
%
% <matrixsize> is [X Y Z] with the number of voxels along the three dimensions
% <mn> is [A B C] with the peak of the Gaussian.  a value of 0 means center of
%   the first voxel.  a value of 1 means center of the last voxel.
%   for example, [.5 .5 .5] means to position the Gaussian exactly
%   at the center of the volume.
% <sd> is [D E F] with the standard deviation of the Gaussian
% <xx>,<yy>,<zz> (optional) are speed-ups (dependent on <matrixsize>)
% <n> (optional) is an exponent like in evalgaussian3d.m.  default: 1.
%
% return a volume where values are in [0,1].
% if the <matrixsize> is 1 along any dimension, we basically act as if that
%   dimension doesn't contribute to the Gaussian.
%
% example:
% figure; imagesc(makeimagestack(makegaussian3d([20 20 9],[.2 .4 .5],[.2 .2 .2])));

% construct coordinates
if ~exist('xx','var') || isempty(xx)
  [xx,yy,zz] = ndgrid(1:matrixsize(1),1:matrixsize(2),1:matrixsize(3));
end
if ~exist('n','var') || isempty(n)
  n = 1;
end

% prep
xmn = normalizerange(mn(1),1,matrixsize(1),0,1,0);
ymn = normalizerange(mn(2),1,matrixsize(2),0,1,0);
zmn = normalizerange(mn(3),1,matrixsize(3),0,1,0);
xsd = choose(matrixsize(1)==1,Inf,sd(1)*(matrixsize(1)-1));
ysd = choose(matrixsize(2)==1,Inf,sd(2)*(matrixsize(2)-1));
zsd = choose(matrixsize(3)==1,Inf,sd(3)*(matrixsize(3)-1));

% do it
f = exp(-((xx-xmn).^2 / (2*xsd^2) + (yy-ymn).^2 / (2*ysd^2) + (zz-zmn).^2 / (2*zsd^2)));
if n ~= 1
  f(f <= 0.5) = f(f <= 0.5) .^ n * (0.5/0.5^n);
  f(f > 0.5) = (f(f > 0.5) - 0.5) .^ (1/n) * (0.5/0.5^(1/n)) + 0.5;
end
