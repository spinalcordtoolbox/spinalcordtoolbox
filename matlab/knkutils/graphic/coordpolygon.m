function [XY,X,Y] = coordpolygon(n)

% function [XY,X,Y] = coordpolygon(n)
%
% <n> is number of points (>= 1)
%
% return coordinates of an equilateral polygon with <n> sides.
% the coordinates are on a circle with radius 0.5 and proceed CCW from (.5,0).
% <XY> is 2 x <n> and <X> and <Y> are both 1 x <n>.
%
% example:
% [XY,X,Y] = coordpolygon(10);
% figure; plot(X,Y,'ro-');

[X,Y] = pol2cart(linspacecircular(0,2*pi,n),0.5);
XY = [X; Y];
