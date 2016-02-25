function f = getfigurepos(fig,units)

% function f = getfigurepos(fig,units)
%
% <fig> (optional) is the figure handle.  default: gcf.
% <units> (optional) is 'normalized' (default) | 'points'
%
% return the position of <fig> in units specified by <units>.
%
% example:
% figure; getfigurepos(gcf,'points')

% input
if ~exist('fig','var') || isempty(fig)
  fig = gcf;
end
if ~exist('units','var') || isempty(units)
  units = 'normalized';
end

% do it
  prev = get(fig,'Units');
  set(fig,'Units',units);
f = get(fig,'Position');
  set(fig,'Units',prev);
