function m = posnegrect(m,dim)

% function m = posnegrect(m,dim)
%
% <m> is a matrix
% <dim> (optional) is the dimension along which to concatenate.  default: 2.
%
% positively- and negatively-rectify <m>.
% the negative rectification is flipped.
%
% example:
% isequal(posnegrect([2 3 -4],2),[2 3 0 0 0 4])

if ~exist('dim','var') || isempty(dim)
  dim = 2;
end
m = cat(dim,posrect(m),posrect(-m));
