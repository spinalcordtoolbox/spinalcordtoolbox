function f = normalizemax(m,dim)

% function f = normalizemax(m,dim)
%
% <m> is a matrix
% <dim> (optional) is the dimension of <m> to operate upon.
%   default to 2 if <m> is a row vector and to 1 otherwise.
%   special case is 0 which means operate globally.
%
% divide <m> by the max value along some dimension (or globally).
%
% example:
% isequal(normalizemax([1 2 3]),[1/3 2/3 1])

% input
if ~exist('dim','var') || isempty(dim)
  dim = choose(isrowvector(m),2,1);
end

% do it
if dim==0
  f = m / max(m(:));
else
  f = bsxfun(@rdivide,m,max(m,[],dim));
end
