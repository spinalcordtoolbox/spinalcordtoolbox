function cmap = cmapangRVF(s)

% function cmap = cmapangRVF(s)
%
% <s> (optional) is the desired size of the colormap.
%   note that regardless of what <s> is, the returned colormap has 88 entries!
%
% return the colormap.  this colormap focuses colors on the right visual field and
% has centered circularity!
%
% example:
% figure; drawcolorbarcircular(cmapangRVF,1);

% deal with input
if ~exist('s','var') || isempty(s)
  s = 88;
end
if ~isequal(s,88)
  warning('we are using 88 colors in the cmapang.m colormap!');
  s = 88;
end

%%%%%%%%%%%%%%% GET THE BASE COLORMAP

cmap = rgb2hsv(cmapangLVF);  % 88 colors right now

%%%%%%%%%%%%%%% FLIP

cmap = flipud(circshift(cmap,[43 0]));

%%%%%%%%%%%%%%% CONVERT TO RGB

cmap = hsv2rgb(cmap);
