function [f,xx,yy] = makesquareimage(res,r,c,sz,transzone,xx,yy)

% function [f,xx,yy] = makesquareimage(res,r,c,sz,transzone,xx,yy)
%
% <res> is the number of pixels along one side of the overall image
% <r> is the row associated with the center of the square (can be a decimal).
%   if [], default to the exact center of the image along the vertical dimension.
% <c> is the column associated with the center of the square (can be a decimal).
%   if [], default to the exact center of the image along the horizontal dimension.
% <sz> is the length of a side of the square in pixels
% <transzone> (optional) is size of transition zone in pixels.  default: 2.
% <xx>,<yy> (optional) are speed-ups (dependent on <res>)
%
% the image is a white square (1) on a black background (0).
% we gradually ramp between black and white using a cosine function.
%
% example:
% im = makesquareimage(100,40,70,20,4);
% figure; imagesc(im); axis equal tight;
% figure; plot(im(40,:),'ro-');

% construct coordinates
if ~exist('transzone','var') || isempty(transzone)
  transzone = 2;
end
if ~exist('xx','var') || isempty(xx)
  [xx,yy] = calcunitcoordinates(res);
end
if isempty(r)
  r = (1+res)/2;
end
if isempty(c)
  c = (1+res)/2;
end

% do it
tempL = 1-makecircleimage(res,-sz/2-transzone/2+(c-(1+res)/2),xx,yy,-sz/2+transzone/2+(c-(1+res)/2),3);
tempR = makecircleimage(res,sz/2-transzone/2+(c-(1+res)/2),xx,yy,sz/2+transzone/2+(c-(1+res)/2),3);
tempU = 1-makecircleimage(res,-sz/2-transzone/2+((1+res)/2-r),xx,yy,-sz/2+transzone/2+((1+res)/2-r),4);
tempD = makecircleimage(res,sz/2-transzone/2+((1+res)/2-r),xx,yy,sz/2+transzone/2+((1+res)/2-r),4);
f = tempL .* tempR .* tempU .* tempD;  % resxres, values in [0,1]
