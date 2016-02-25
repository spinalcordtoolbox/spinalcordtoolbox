function cmap = cmapang(s)

% function cmap = cmapang(s)
%
% <s> (optional) is the desired size of the colormap.
%   note that regardless of what <s> is, the returned colormap has 64 entries!
%
% return the colormap.  this colormap has centered circularity!
%
% example:
% figure; drawcolorbarcircular(cmapang,1);

% deal with input
if ~exist('s','var') || isempty(s)
  s = 64;
end
if ~isequal(s,64)
  warning('we are forcing the use of 64 colors in the cmapang.m colormap!');
  s = 64;
end

%%%%%%%%%%%%%%% DEFINE COLORS

% define
n = 8;
colors = [
     0/360  1 1     % red
     30/360 1 1     % orange
     60/360 1 .92  % yellow
     105/360 1 .9   % green
     180/360 1 .9  % light-blue
     240/360 1 .9    % blue
     280/360 1 .75    % purple
     305/360 1 .82   % magenta
     1       1 1    % red
  ];

%%%%%%%%%%%%%%% UPSAMPLE

cmap = colorinterpolate(colors,n,1);
cmap(end,:) = [];

%%%%%%%%%%%%%%% DO SOME FINE-SCALE ADJUSTMENTS

cmap(2:9,:) = colorinterpolate([cmap(3,:); cmap(9,:)],7,1);

% %1+8+8+8 = 25
cmap(25:33,:) = [cmap(25,:); cmap(26,:); colorinterpolate([cmap(29,:); cmap(33,:)],6,1)];

% % %1+8+8+8+8+8 = 41
cmap(41:49,:) = [cmap(41,:); colorinterpolate([mean(cmap(42:43,:),1); cmap(49,:)],7,1)];

% % %1+8+8+8+8+8+8 = 49
cmap(49:57,:) = [colorinterpolate([cmap(49,:); cmap(55,:)],7,1); cmap(57,:)];

%%%%%%%%%%%%%%% CONVERT TO RGB

cmap = hsv2rgb(cmap);
