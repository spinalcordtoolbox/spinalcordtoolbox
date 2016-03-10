function f = squish(m,num)

% function f = squish(m,num)
%
% <m> is a matrix
% <num> is the positive number of initial dimensions to squish together
%
% return <m> squished.
%
% example:
% isequal(squish([1 2; 3 4],2),[1 3 2 4]')

% get the size of m
msize = [size(m) ones(1,num-ndims(m))];  % add ones to end if necessary

% calculate the new dimensions
newdim = [prod(msize(1:num)) msize(num+1:end)];

% do the reshape
f = reshape(m,[newdim 1]);  % tack on a 1 to handle the special case of squishing everything together
