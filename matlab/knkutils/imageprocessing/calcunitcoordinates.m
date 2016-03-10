function [xx,yy] = calcunitcoordinates(res)

% function [xx,yy] = calcunitcoordinates(res)
%
% <res> is the number of pixels on a side
%  
% return <xx> and <yy> which contain x- and y-coordinates corresponding
% to equally spaced points within the space bounded by -.5 and .5.
% these points can be treated as centers of pixels.
%
% example:
% [xx,yy] = calcunitcoordinates(2);
% isequal(xx,[-.25 .25; -.25 .25]) & isequal(yy,[.25 .25; -.25 -.25])

% notice that the second argument proceeds from .5 to -.5.
% this ensures that the results match the usual coordinate axes 
% where the top is the positive y-axis.
[xx,yy] = meshgrid(linspacepixels(-.5,.5,res),linspacepixels(.5,-.5,res));
