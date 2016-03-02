function visualizemultiscalegaborfilters(info,vals,mn,mx,cmap,prefix)

% function visualizemultiscalegaborfilters(info,vals,mn,mx,cmap,prefix)
%
% <info> is 9 x channels, and is returned by applymultiscalegaborfilters.m
% <vals> is 1 x channels with values to visualize
% <mn> (optional) is the minimum value to associate with the colormap
%   default: -max(abs(vals))
% <mx> (optional) is the maximum value to associate with the colormap
%   default: max(abs(vals))
% <cmap> (optional) is the colormap to use
%   default: cmapsign
% <prefix> (optional) is the prefix for a filename (see figurewrite.m).
%   if not supplied, we make a figure window.
%   if supplied, we write to a file.
%
% draw some visualizations.
%
% see also applymultiscalegaborfilters.m.
% see also visualizemultiscalegaborfilters2.m.

% inputs
if ~exist('mn','var') || isempty(mn)
  mn = -max(abs(vals));
end
if ~exist('mx','var') || isempty(mx)
  mx = max(abs(vals));
end
if ~exist('cmap','var') || isempty(cmap)
  cmap = cmapsign;
end
if ~exist('prefix','var') || isempty(prefix)
  prefix = [];
end

% calc
numph = max(info(6,:));
res = info(7,1);
r = floor(sqrt(numph));
c = ceil(numph/r);

% do it
if isempty(prefix)
  drawnow; figure; setfigurepos([.1 .1 .8 .8]);
else
  figureprep([.1 .1 .8 .8]);
end
for p=1:numph
  subplot(r,c,p); hold on;
  set(gca,'YDir','reverse'); colormap(cmap);
  for zz=p:numph:size(info,2)
    h = plotorientedbar(info(1,zz),info(2,zz),-info(3,zz),info(4,zz),2*(log2(info(5,zz))-2),'r-');
    set(h,'Color',cmaplookup(vals(zz),mn,mx));
  end
  axis equal tight;
  axis([.5 res+.5 .5 res+.5]);
end
if isempty(prefix)
else
  figurewrite(prefix);
end
