function cmap = cmapangLVF(s)

% function cmap = cmapangLVF(s)
%
% <s> (optional) is the desired size of the colormap.
%   note that regardless of what <s> is, the returned colormap has 88 entries!
%
% return the colormap.  this colormap focuses colors on the left visual field and
% has centered circularity!
%
% example:
% figure; drawcolorbarcircular(cmapangLVF,1);

% deal with input
if ~exist('s','var') || isempty(s)
  s = 88;
end
if ~isequal(s,88)
  warning('we are using 88 colors in the cmapang.m colormap!');
  s = 88;
end

%%%%%%%%%%%%%%% GET THE BASE COLORMAP

cmap = rgb2hsv(cmapang);  % 64 colors right now

%%%%%%%%%%%%%%% ADD LIGHT GRAYS

% insert 24 light grays right before the yellow
cmap = [cmap(1:(1+8+8-1),:); repmat([1 0 .9],[24 1]); cmap(1+8+8:end,:)];

% the first light gray should repeat the yellow
cmap(1+8+8-1+1,:) = cmap(1+8+8-1+ 24 +1,:);
 
% then make the color map center the light gray on the right horizontal meridian
cmap = circshift(cmap,[-28 0]);
 
%%%%%%%%%%%%%%% CONVERT TO RGB

cmap = hsv2rgb(cmap);
