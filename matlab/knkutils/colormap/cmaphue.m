function f = cmaphue(n)

% function f = cmaphue(n)
%
% <n> (optional) is desired number of entries.  default: 64.
%
% return a hue-based colormap.
% suitable for circular ranges.
%
% example:
% figure; imagesc((1:8)'); axis equal tight; colormap(cmaphue(8)); colorbar;

% inputs
if ~exist('n','var') || isempty(n)
  n = 64;
end

% constants
colors = [0/360 1 1; 30/360 1 1; 60/360 1 .9; 120/360 1 1; 180/360 1 .95; 230/360 1 1; 275/360 1 1; 300/360 1 .95];  % HSV
extra = [1 1 1];
nn = size(colors,1);

% do it (MAYBE CONSOLIDATE THIS CODE?)
f = [];
for p=1:size(colors,2)
  f(:,p) = interp1(1:nn+1,[colors(:,p)' extra(1,p)],linspacecircular(1,nn+1,n),'linear');
end
f = hsv2rgb(f);
