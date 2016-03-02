function f = fillout(m,desiredsize)

% function f = fillout(m,desiredsize)
%
% <m> is a matrix
% <desiredsize> is the desired size.  the desired size for each dimension must be a multiple
%   of the size of the corresponding dimension of <m>.
%
% repmat <m> as necessary in order to have size <desiredsize>.
%
% example:
% isequal(fillout([1 2],[3 2]),[1 2; 1 2; 1 2])

msize = [size(m) ones(1,length(desiredsize)-ndims(m))];  % add ones if needed
f = repmat(m,desiredsize ./ msize);  % do it
