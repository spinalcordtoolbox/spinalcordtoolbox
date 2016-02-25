function f = cmapsign3(n)

% function f = cmapsign3(n)
%
% <n> (optional) is desired number of entries
%   default: 64
%
% return a blue-lightgray-red colormap.
% suitable for ranges like [-X X].
%
% example:
% figure; imagesc(randn(100,100)); axis equal tight; colormap(cmapsign3); colorbar;

% inputs
if ~exist('n','var') || isempty(n)
  n = 64;
end

% constants
colors = [
  0 0 1  % blue
  .85 .85 .85  % light gray
  1 0 0  % red
  ];

% do it (MAYBE CONSOLIDATE THIS CODE?)
f = [];
for p=1:size(colors,2)
  f(:,p) = interp1(linspace(0,1,size(colors,1)),colors(:,p)',linspace(0,1,n),'linear');
end


% OLD
% f = colorinterpolate(colors,31,1);
