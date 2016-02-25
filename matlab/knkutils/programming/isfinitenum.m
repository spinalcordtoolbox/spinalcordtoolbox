function f = isfinitenum(m)

% function f = isfinitenum(m)
%
% <m> is a matrix
%
% return a logical matrix the same size as <m>.
% an element is 1 iff it is finite and numeric.
% specifically:
%   f = isfinite(m) & isnumeric(m);
%
% example:
% isequal(isfinitenum([1 1.5 NaN Inf]),[1 1 0 0])

f = isfinite(m) & isnumeric(m);
