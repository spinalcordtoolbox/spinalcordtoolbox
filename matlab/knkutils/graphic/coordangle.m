function [XY,X,Y] = coordangle(ang1,ang2)

% function [XY,X,Y] = coordangle(ang1,ang2)
%
% <ang1>,<ang2> are angles in [0,2*pi)
%
% return coordinates of an angle defined by two line segments that
% emanate from the origin and have length 0.5.  the order of
% coordinates is "tip of angle 1", origin, and "tip of angle 2".
% <XY> is 2 x <n> and <X> and <Y> are both 1 x <n>.
%
% example:
% [XY,X,Y] = coordangle(0,pi/4);
% figure; plot(X,Y,'ro-');

[x1,y1] = pol2cart(ang1,0.5);
[x2,y2] = pol2cart(ang2,0.5);
X = [x1 0 x2];
Y = [y1 0 y2];
XY = [X; Y];
