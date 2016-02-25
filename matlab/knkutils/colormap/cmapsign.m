function f = cmapsign(n)

% function f = cmapsign(n)
%
% <n> (optional) is desired number of entries
%   default: 64
%
% return a blue-black-red colormap.
% suitable for ranges like [-X X].
%
% example:
% figure; imagesc(randn(100,100)); axis equal tight; colormap(cmapsign); colorbar;

% inputs
if ~exist('n','var') || isempty(n)
  n = 64;
end

% constants
colors = [
  0 0 1  % blue
  0 0 0  % black
  1 0 0  % red
  ];

% do it (MAYBE CONSOLIDATE THIS CODE?)
f = [];
for p=1:size(colors,2)
  f(:,p) = interp1(linspace(0,1,size(colors,1)),colors(:,p)',linspace(0,1,n),'linear');
end


% OLD
% f = colorinterpolate(colors,31,1);
