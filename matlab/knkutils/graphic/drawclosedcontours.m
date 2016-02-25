function f = drawclosedcontours(res,x,y,sc,rot,color,edgewidth,edgecolor,bg,coord,mode)

% function f = drawclosedcontours(res,x,y,sc,rot,color,edgewidth,edgecolor,bg,coord,mode)
%
% arguments are the same as to drawclosedcontour.m except for:
%   <res> is the number of pixels along one side
%   <x>,<y>,<rot> can be vectors
%   <coord> can be a cell vector with different contours
%   <mode> (optional) is
%     0 means render a separate image for each contour
%     1 means render all contours on one image
%     default: 0.
% for <x> and <y>, assume the standard coordinate frame.
%
% return a series of 2D images where values are in [0,1].
% the dimensions of the returned matrix are res x res x images
% note that we explicitly convert to grayscale.
%
% example:
% figure; imagesc(makeimagestack(drawclosedcontours(100,0,0,.5,0,[.5 1 1],2,[0 0 0],[1 1 1],{coordpolygon(3) coordpolygon(100)}))); axis equal tight;

% NOTE: see also drawbars.m.

% input
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end
if ~iscell(coord)
  coord = {coord};
end
numcontours = length(coord);

% do it
fig = figure;
if mode == 0
  f = zeros(res,res,numcontours);
end
for p=1:numcontours
  if length(rot) > 1
    rot0 = rot(p);
  else
    rot0 = rot;
  end
  if length(x) > 1
    x0 = x(p);
  else
    x0 = x;
  end
  if length(y) > 1
    y0 = y(p);
  else
    y0 = y;
  end
  if mode == 0
    clf;
  end
  drawclosedcontour(0,x0,y0,sc,rot0,color,edgewidth,edgecolor,bg,coord{p});
  if mode == 0
    temp = renderfigure(res,2);
    if size(temp,3) > 1
      temp = rgb2gray(temp);
    end
    f(:,:,p) = temp;
  end
end  
if mode == 1
  temp = renderfigure(res,2);
  if size(temp,3) > 1
    temp = rgb2gray(temp);
  end
  f = temp;
end
close(fig);
