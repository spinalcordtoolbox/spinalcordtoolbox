function f = visualizemultiscalegaborfilters2(info,vals,mn,mx,cmap,flag,flag2)

% function f = visualizemultiscalegaborfilters2(info,vals,mn,mx,cmap,flag,flag2)
%
% <info> is 9 x channels, and is returned by applymultiscalegaborfilters.m
% <vals> is 1 x channels with values to visualize
% <mn> (optional) is the minimum value to associate with the colormap
%   default: -max(abs(vals))
% <mx> (optional) is the maximum value to associate with the colormap
%   default: max(abs(vals))
% <cmap> (optional) is the colormap to use
%   default: cmapsign
% <flag> (optional) is
%   0 means average values across phase
%   1 means make separate figures for each phase
%   default: 0
% <flag2> (optional) is a 2D matrix size [A B].
%   if not supplied, we make a figure window (and <f> is returned as []).
%   if supplied, we return an image in <f> (range is [0 1]).
%     each spatial tiling is resampled via nearest neighbor
%     interpolation to dimensions [A B].  in this mode,
%     <cmap> is unused.
%
% draw some visualizations.
%
% see also applymultiscalegaborfilters.m.
% see also visualizemultiscalegaborfilters.m.

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
if ~exist('flag','var') || isempty(flag)
  flag = 0;
end
if ~exist('flag2','var') || isempty(flag2)
  flag2 = [];
end

% calc
f = [];
numsc = max(info(8,:));
numor = max(info(9,:));
numph = max(info(6,:));

% do it
for zz=1:choose(flag==0,1,numph)
  if isempty(flag2)
    drawnow; figure; setfigurepos([.1 .1 .8 .8]);
  end
  for p=1:numsc
    tt = reshape(vals(info(8,:)==p),numph,numor,[]);
    side = sqrt(size(tt,3));
    if flag==0
      tt = permute(mean(reshape(tt,numph,numor,side,side),1),[3 4 2 1]);
    else
      tt = permute(reshape(tt(zz,:,:),1,numor,side,side),[3 4 2 1]);
    end
    if isempty(flag2)
      subplot(numsc,1,p);
      imagesc(makeimagestack(tt,[mn mx],2,[1 numor]),[0 1]);
      axis equal tight;
      set(gca,'YDir','reverse');
      colormap(cmap);
    else
      f = cat(1,f,makeimagestack(processmulti(@imresize,tt,flag2,'nearest'),[mn mx],0,[1 numor]));
    end
  end
end
