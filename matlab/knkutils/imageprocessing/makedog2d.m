function [f,xx,yy] = makedog2d(res,r,c,sd,sdratio,volratio,xx,yy)

% function [f,xx,yy] = makedog2d(res,r,c,sd,sdratio,volratio,xx,yy)
%
% <res> is the number of pixels along one side
% <r> is the row associated with the dog center (can be a decimal).
%   if [], default to the exact center of the image along the vertical dimension.
% <c> is the column associated with the dog center (can be a decimal).
%   if [], default to the exact center of the image along the horizontal dimension.
% <sd> is the number of pixels in the SD of the center Gaussian
% <sdratio> is the ratio of the surround SD to the center SD
% <volratio> is the ratio of the surround volume to the center volume
% <xx>,<yy> (optional) are speed-ups (dependent on <res>)
%
% return <f>, an image where values are in [-1,1].
% we don't normalize the matrix for power or anything like that.
% 
% the center gaussian has a height of 1, and then comes the surround gaussian
% which is subtracted from the center gaussian.
%
% example:
% figure; imagesc(makedog2d(64,[],[],4,2,1),[-1 1]);

% input
if isempty(r)
  r = (1+res)/2;
end
if isempty(c)
  c = (1+res)/2;
end

% construct coordinates
if ~exist('xx','var') || isempty(xx)
  [xx,yy] = calcunitcoordinates(res);
end

% convert to the unit coordinate frame
r = normalizerange(r,.5,-.5,.5,res+.5,0,0,1);  % note the signs
c = normalizerange(c,-.5,.5,.5,res+.5,0,0,1);
sd = sd/res;

% do it
temp = (xx-c).^2 + (yy-r).^2;
center = exp(temp/-(2*sd^2));  % height is 1
surround = exp(temp/-(2*(sd*sdratio)^2)) * 1/sdratio^2;  % surround now has same volume as center
f = center - sqrt(volratio)*surround;
