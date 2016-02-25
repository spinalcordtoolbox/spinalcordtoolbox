function f = squeezedim(m,dim)

% function f = squeezedim(m,dim)
%
% <m> is a matrix
% <dim> is a dimension of <m> with only one element
%
% reshape <m> such that <dim> is removed.
% if size(m,dim) is not 1, we die.
%
% example:
% isequal(size(squeezedim(randn(5,1,4,1,3),2)),[5 4 1 3])

msize = size(m);
if dim <= length(msize)
  assert(msize(dim)==1);
  msize(dim) = [];
end
f = reshape(m,[msize 1]);  % tack on 1 for safety
