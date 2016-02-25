function f = subsetfun(fun,x,n,dim)

% function f = subsetfun(fun,x,n,dim)
%
% <fun> is a function that accepts a vector and returns a matrix
% <x> is a vector
% <n> is a vector of counts such that sum(<n>) is equal to length(<x>)
% <dim> (optional) is the dimension along which to concatenate results.
%   default: 2.
%
% apply <fun> to successive subsets of <x> (as specified by <n>)
% and concatenate the results.
%
% example:
% isequal(subsetfun(@mean,[1 2 3 4],[1 3]),[1 3])

% input
if ~exist('dim','var') || isempty(dim)
  dim = 2;
end

% do it
f = [];
for p=1:length(n)
  f = cat(dim,f,feval(fun,x(sum(n(1:p-1)) + (1:n(p)))));
end
