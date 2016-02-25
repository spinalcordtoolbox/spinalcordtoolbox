function h = axissquarify(wantunityline,style)

% function h = axissquarify(wantunityline,style)
%
% <wantunityline> (optional) is whether to also draw a unity line.
%   default: 1.
% <style> (optional) is the line style to use (only matters if <wantunityline>).
%   default: 'g-'.
%
% make the current axis square-shaped, equal aspect ratio, and centered at origin.
% we ensure that the bounds of the new axis enclose the bounds of the current axis.
% if we draw a unity line, return <h> as the handle to this line.
% otherwise, return <h> as [].
%
% example:
% figure; axissquarify;

% input
if ~exist('wantunityline','var') || isempty(wantunityline)
  wantunityline = 1;
end
if ~exist('style','var') || isempty(style)
  style = 'g-';
end

% do it
ax = axis;
mx = max(abs(ax(1:4)));
if wantunityline
  hold on; h = plot([-mx mx],[-mx mx],style);
else
  h = [];
end
axis square;
axis([-mx mx -mx mx]);
