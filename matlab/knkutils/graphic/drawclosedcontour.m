function drawclosedcontour(res,x,y,sc,rot,color,edgewidth,edgecolor,bg,coord)

% function drawclosedcontour(res,x,y,sc,rot,color,edgewidth,edgecolor,bg,coord)
%
% <res> is
%   0 means standard figure coordinate frame
%  -1 means y-reversed coordinate frame
% <x> is x-position of object
% <y> is y-position of object
% <sc> is scale factor in (0,Inf)
% <rot> is a CCW rotation to apply in [0,2*pi)
% <color> is a 3-element vector with the color.  [] means no fill.
% <edgewidth> is the edge width in points.  0 means no edge.
%   negative values mean do not draw the closing edge; in this case,
%   <color> should be set to [].
% <edgecolor> is a 3-element vector with the edge color.  does not matter if <edgewidth> is 0.
% <bg> is a 3-element vector with the background color.  [] means do not draw the background.
% <coord> is 2 x N with the x- and y-coordinates of the contour.  N must be at least 2.
%
% draw a closed contour on the current figure.
% the field-of-view of <coord> is assumed to be [-.5,.5]
% (but parts of the contour can extend beyond the field-of-view,
% and these parts can show up if <sc> is less than 1).
% <x> and <y> are interpreted with respect to the x-
% and y-axes being bounded by [-.5,.5].  we automatically set the
% axis bounds and also reverse the y-axis if necessary.
%
% example:
% figure; drawclosedcontour(0,0,0,.5,0,[.5 1 1],2,[0 0 0],[1 1 1],coordpolygon(3));

% repeat first two (A BIT HACKY, BUT SEEMS TO BE NECESSARY TO MAKE SURE EDGES CLOSE ALL THE WAY)
coord = cat(2,coord,coord(:,1:2));

% prep figure
hold on;

% draw square for background
if ~isempty(bg)
  hbg = patch([-.5 -.5 .5 .5 -.5],[-.5 .5 .5 -.5 -.5],bg);
  set(hbg,'EdgeColor','none');
end

% draw contour (scale, rotate, translate)
rot0 = choose(res==0,-rot,rot);
newcoord = [cos(rot0) sin(rot0); -sin(rot0) cos(rot0)] * sc * coord;  % want to rotate CCW (see makegrating.m)
if edgewidth < 0  % in this case, we need to do a line object (without opengl we can't omit the last line)
  h = line(newcoord(1,1:end-2) + x,newcoord(2,1:end-2) + y);
  set(h,'LineWidth',-edgewidth);
  set(h,'Color',edgecolor);
else
  h = patch(newcoord(1,:) + x,newcoord(2,:) + y,'k');
  set(h,'FaceColor',choose(isempty(color),'none',color));
  if edgewidth==0
    set(h,'EdgeColor','none');
  else
    set(h,'LineWidth',edgewidth);
    set(h,'EdgeColor',edgecolor);
  end
end

% prep figure
axis([-.5 .5 -.5 .5]);
if res ~= 0
  set(gca,'YDir','reverse');
end
