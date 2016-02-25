function [f,xx,yy] = makegaussian2d(res,r,c,sr,sc,xx,yy,ang,omitexp)

% function [f,xx,yy] = makegaussian2d(res,r,c,sr,sc,xx,yy,ang,omitexp)
%
% <res> is the number of pixels along one side
% <r> is the row associated with the peak of the Gaussian (can be a decimal).
%   if [], default to the exact center of the image along the vertical dimension.
% <c> is the column associated with the peak of the Gaussian (can be a decimal).
%   if [], default to the exact center of the image along the horizontal dimension.
% <sr> is the standard deviation in the vertical direction
% <sc> is the standard deviation in the horizontal direction
% <xx>,<yy> (optional) are speed-ups (dependent on <res>)
% <ang> (optional) is the CCW rotation to apply in [0,2*pi).  0 means no rotation.
%   it's okay for <ang> to go out of range.  default: 0.
% <omitexp> (optional) is whether to omit the final exp operation.  default: 0.
%
% return an image where values are in [0,1].
%
% if you want an L1-normalized image, divide the image by 2*pi*<sr>*<sc>.
% note that this is in reference to the ideal case where the Gaussian has 
% enough room to extend out.  so, if you are constructing a Gaussian that
% does not fit very well within the image, the actual L1 length of the image
% that is constructed will not be exactly 1.
%
% note that it doesn't matter if <sr> or <sc> are negative, since they 
% are always squared in function evaluation.
%
% history:
% - 2013/08/28 - implement speed-up
%
% example:
% figure; imagesc(makegaussian2d(32,8,8,4,2),[0 1]);

% input
if isempty(r)
  r = (1+res)/2;
end
if isempty(c)
  c = (1+res)/2;
end
if ~exist('ang','var') || isempty(ang)
  ang = 0;
end
if ~exist('omitexp','var') || isempty(omitexp)
  omitexp = 0;
end

% construct coordinates
if ~exist('xx','var') || isempty(xx)
  [xx,yy] = calcunitcoordinates(res);
end

% convert to the unit coordinate frame
  % r = normalizerange(r,.5,-.5,.5,res+.5,0,0,1);  % note the signs
  % c = normalizerange(c,-.5,.5,.5,res+.5,0,0,1);
r = (-1/res) * r + (.5 + .5/res);  % this is faster
c = (1/res) * c + (-.5 - .5/res);  % this is faster
sr = sr/res;
sc = sc/res;

% construct coordinates (see makegabor2d.m)
coord = [cos(ang) sin(ang); -sin(ang) cos(ang)]*[flatten(xx-c); flatten(yy-r)];

% handle equal std dev as a separate case for speed reasons
if sc==sr
  f = (coord(1,:).^2+coord(2,:).^2)/-(2*sc^2);
else
  f = coord(1,:).^2/-(2*sc^2) + coord(2,:).^2/-(2*sr^2);
end
if ~omitexp
  f = exp(f);
end
f = reshape(f,size(xx));
