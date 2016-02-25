function f = mod4(x,y)

% function f = mod4(x,y)
%
% <x> is a matrix
% <y> is a number
%
% returns mod(x,y) except that values lie in [-y/2,y/2).
%
% example:
% mod4(3/2*pi,2*pi)==-pi/2

f = mod(x,y);
f(f>=y/2) = f(f>=y/2) - y;
