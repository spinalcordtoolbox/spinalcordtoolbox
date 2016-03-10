function fig = figureprep(pos,wantvisible)

% function fig = figureprep(pos,wantvisible)
%
% <pos> (optional) is a position like in setfigurepos.m
% <wantvisible> (optional) is whether to keep the window visible.  default: 0.
%
% make a new invisible figure window and set hold on.
% then, if <pos> is supplied, set the position of the window.
% return a handle to the figure window.
%
% use in conjunction with figurewrite.m.
%
% example:
% figureprep;
% scatter(randn(100,1),randn(100,1));
% figurewrite;

% input
if ~exist('pos','var') || isempty(pos)
  pos = [];
end
if ~exist('wantvisible','var') || isempty(wantvisible)
  wantvisible = 0;
end

% do it
fig = figure; hold on;
if ~wantvisible
  set(fig,'Visible','off');
end
if ~isempty(pos)
  setfigurepos(pos);
end
