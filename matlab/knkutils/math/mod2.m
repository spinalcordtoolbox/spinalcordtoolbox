function f = mod2(x,y)

% function f = mod2(x,y)
%
% <x> is a matrix
% <y> is a number
%
% returns mod(x,y) except that 0 is returned as y.
%
% example:
% mod2(6,3)==3

f = mod(x,y);
f(f==0) = y;
