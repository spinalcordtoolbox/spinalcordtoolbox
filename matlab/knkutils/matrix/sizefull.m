function f = sizefull(m,numdims)

% function f = sizefull(m,numdims)
%
% <m> is a matrix
% <numdims> is the number of dimensions desired
%
% return the size of <m> with respect to the first <numdims> dimensions.
% the result is a vector of length <numdims>.
%
% example:
% isequal(sizefull([1 2 3],3),[1 3 1])

f = placematrix(ones(1,numdims),size(m),[1 1]);
