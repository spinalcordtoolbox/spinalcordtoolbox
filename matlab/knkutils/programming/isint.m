function f = isint(m)

% function f = isint(m)
%
% <m> is a matrix
%
% return a logical matrix the same size as <m>.
% an element is 1 iff it is a float and finite and exactly equal to an integer.
% specifically:
%   f = isfloat(m) & isfinite(m) & m==round(m);
%
% example:
% isequal(isint([1 1.5 NaN Inf]),[1 0 0 0])

% do it
f = isfloat(m) & isfinite(m) & m==round(m);
