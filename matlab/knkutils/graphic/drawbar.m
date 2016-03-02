function drawbar(res,x,y,ang,width,length,color,bg)

% function drawbar(res,x,y,ang,width,length,color,bg)
%
% <res> is
%   0 means standard figure coordinate frame
%  -1 means y-reversed coordinate frame
% <x> is x-position of bar center
% <y> is y-position of bar center
% <ang> is the orientation in [0,2*pi).  0 means a horizontal bar.
% <width> is the width of the bar
% <length> is the length of the bar
% <color> is a 3-element vector with the bar color
% <bg> is a 3-element vector with the background color.  [] means do not draw the background.
%
% draw an oriented bar on the current figure.
% <x>,<y>,<width>,<length> are interpreted with respect to the x-
% and y-axes being bounded by [-.5,.5].  we automatically set the
% axis bounds and also reverse the y-axis if necessary.
%
% example:
% figure; makebar(0,.2,.1,pi/6,.1,.4,[.5 .1 .3],[1 1 1]);

% prep figure
hold on;

% draw square for background
if ~isempty(bg)
  hbg = patch([-.5 -.5 .5 .5 -.5],[-.5 .5 .5 -.5 -.5],bg);
  set(hbg,'EdgeColor','none');
end

% draw bar (act as if at origin, rotate CCW, then translate)
xx = [-length/2 -length/2 length/2 length/2 -length/2];
yy = [-width/2 width/2 width/2 -width/2 -width/2];
ang0 = choose(res==0,-ang,ang);
newcoord = [cos(ang0) sin(ang0); -sin(ang0) cos(ang0)] * [xx; yy];  % want to rotate CCW (see makegrating.m)
hbar = patch(newcoord(1,:) + x,newcoord(2,:) + y,color);
set(hbar,'EdgeColor','none');

% prep figure
axis([-.5 .5 -.5 .5]);
if res ~= 0
  set(gca,'YDir','reverse');
end
