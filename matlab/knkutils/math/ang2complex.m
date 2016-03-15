function [f,x,y] = ang2complex(m)

% function [f,x,y] = ang2complex(m)
%
% <m> is a matrix of angles in radians
%
% return <f> as the corresponding unit-magnitude complex numbers.
% return <x> and <y> as the x- and y-coordinates of these complex numbers.
%
% [f,x,y] = ang2complex(.2)

x = cos(m);
y = sin(m);
f = x + j*y;
