function f = cmaplookup(x,mn,mx,circulartype,cmap)

% function f = cmaplookup(x,mn,mx,circulartype,cmap)
%
% <x> is a matrix with values
% <mn>,<mx> are values to associate with the minimum and maximum of the colormap
% <circulartype> (optional) is
%   0 means normal colormap lookup (min is .5, max is n+.5)
%   1 means centered colormap lookup (min is 1, max is n+1 (repeat first color))
%   default: 0.
% <cmap> (optional) is the colormap.  default is current figure colormap.
%
% return a matrix of colors.  the dimensions are like this:
%   1 x 1 => 1 x 3
%   1 x N => N x 3
%   N x 1 => N x 3
%   M x N x ... => M x N x ... x 3
%
% note that we specifically map NaNs in <x> to the first color entry.
%
% example:
% isequal(cmaplookup(.34,0,1,[],gray(3)),[.5 .5 .5])

% inputs
if ~exist('circulartype','var') || isempty(circulartype)
  circulartype = 0;
end
if ~exist('cmap','var') || isempty(cmap)
  cmap = colormap;
end

% calc
n = size(cmap,1);

% calculate indices into colormap
switch circulartype
case 0
  f = round(normalizerange(x,.5,n+.5,mn,mx));
  f(f==n+1) = n;  % oops, outlier at top
  f(f==0) = 1;  % oops, rounding error
case 1
  f = round(normalizerange(x,1,n+1,mn,mx));
  f(f==n+1) = 1;  % the top one is actually the same as the first
end
f(isnan(f)) = 1;

% figure out size
if isvector(x)
  sz = [length(x) 3];
else
  sz = [size(x) 3];
end

% reshape
f = reshape(cmap(f,:),sz);
