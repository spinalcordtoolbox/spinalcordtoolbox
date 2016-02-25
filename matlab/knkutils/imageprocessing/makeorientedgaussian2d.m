function [f,xx,yy] = makeorientedgaussian2d(res,r,c,ang,sd1,sd2,orsd,ors,xx,yy)

% function [f,xx,yy] = makeorientedgaussian2d(res,r,c,ang,sd1,sd2,orsd,ors,xx,yy)
%
% <res> is the number of pixels along one side
% <r> is the row associated with the peak of the Gaussian (can be a decimal).
%   if [], default to the exact center of the image along the vertical dimension.
% <c> is the column associated with the peak of the Gaussian (can be a decimal).
%   if [], default to the exact center of the image along the horizontal dimension.
% <ang> is the orientation in [0,2*pi).  0 means horizontal.
% <sd1> is the std dev of the Gaussian along the major axis (parallel to the orientation)
% <sd2> is the std dev of the Gaussian along the minor axis (orthogonal to the orientation)
% <orsd> is the orientation width (i.e. the k in the von Mises distribution)
% <ors> (optional) is a vector of orientations to draw images at.
%   default: linspacecircular(0,pi,8).
% <xx>,<yy> (optional) are speed-ups (dependent on <res>)
%
% return images where values are in [0,1].
% the images have dimensions <res> x <res> x length(<ors>).
% the von Mises distribution is normalized such that the maximum possible value is exactly 1.
%
% example:
% figure; imagesc(makeimagestack(makeorientedgaussian2d(32,[],[],pi/4,10,5,2,[]),[],[],-1),[0 1]); axis equal tight;

% input
if isempty(r)
  r = (1+res)/2;
end
if isempty(c)
  c = (1+res)/2;
end
if ~exist('ors','var') || isempty(ors)
  ors = linspacecircular(0,pi,8);
end

% construct coordinates
if ~exist('xx','var') || isempty(xx)
  [xx,yy] = calcunitcoordinates(res);
end

% convert to the unit coordinate frame
r = normalizerange(r,.5,-.5,.5,res+.5,0,0,1);  % note the signs
c = normalizerange(c,-.5,.5,.5,res+.5,0,0,1);
sd1 = sd1/res;
sd2 = sd2/res;

% do it
  % see makegabor2d.m
coord = [cos(ang) sin(ang); -sin(ang) cos(ang)]*[flatten(xx-c); flatten(yy-r)];
if sd1==sd2
  f = exp((coord(1,:).^2 + coord(2,:).^2) / -(2*sd1^2));
else
  f = exp(-1/2 * (coord(1,:).^2/(sd1^2) + coord(2,:).^2/(sd2^2)));
end
gains = exp(orsd*cos(2*mod(ors-ang,pi)))/exp(orsd);
f = bsxfun(@times,reshape(gains,1,1,[]),f);
f = reshape(f,[size(xx) length(ors)]);
