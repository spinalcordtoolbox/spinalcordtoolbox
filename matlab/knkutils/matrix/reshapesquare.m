function f = reshapesquare(x)

% function f = reshapesquare(x)
%
% <x> is a matrix with N*N x D1 x D2 x ... elements.
%   special case is a row vector which we simply
%   transpose to become a column vector.
%
% reshape <x> into an N x N x D1 x D2 x ... matrix.
%
% example:
% reshapesquare([1 2 3 4])

if isvector(x)
  x = x(:);
end
xsize = size(x);
n = sqrt(xsize(1));
f = reshape(x,[n n xsize(2:end)]);

%OLD:
%f = reshape(x,sqrt(length(x)),[]);
