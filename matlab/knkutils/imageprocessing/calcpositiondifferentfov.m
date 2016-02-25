function [xi,yi] = calcpositiondifferentfov(res1,size1,res2,size2)

% function [xi,yi] = calcpositiondifferentfov(res1,size1,res2,size2)
%
% <res1> is [R C] referring to image dimensions
% <size1> is [Rsz Csz] with the size of a pixel along each dimension
% <res2> is [R2 C2] referring to image dimensions
% <size2> is [R2sz C2sz] with the size of a pixel along each dimension
%
% return <xi> as a vector of x-positions and <yi> as a vector of y-positions.
% these indicate where the pixels of the second image reside, in terms of
% units that reflect the matrix space of the first image.  we assume that 
% the two images share the same center but may have different fields-of-view.  
%
% example:
% [xi,yi] = calcpositiondifferentfov([100 100],[1 1],[2 4],[50 25])

% calc
center0 = (1+res1)/2;  % the center of the image in the original matrix units

% do it
xi = linspace(center0(2) - (res2(2)/2 * size2(2) - size2(2)/2)/size1(2), ...
              center0(2) + (res2(2)/2 * size2(2) - size2(2)/2)/size1(2),res2(2));
yi = linspace(center0(1) - (res2(1)/2 * size2(1) - size2(1)/2)/size1(1), ...
              center0(1) + (res2(1)/2 * size2(1) - size2(1)/2)/size1(1),res2(1));
