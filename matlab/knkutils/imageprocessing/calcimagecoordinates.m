function [xx,yy] = calcimagecoordinates(res)

% function [xx,yy] = calcimagecoordinates(res)
%
% <res> is the number of pixels on a side
%  
% return <xx> and <yy> which contain x- and y-coordinates corresponding
% to the pixels of the image.  note that pixel (1,1) is at the lower left.
%
% example:
% [xx,yy] = calcimagecoordinates(2);
% isequal(xx,[1 2; 1 2]) & isequal(yy,[2 2; 1 1])

[xx,yy] = meshgrid(1:res,fliplr(1:res));
