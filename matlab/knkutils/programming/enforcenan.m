function m = enforcenan(m,dim)

% function m = enforcenan(m,dim)
%
% <m> is a matrix
% <dim> is the dimension to operate upon
%
% look for cases that are all 0 along <dim>.
% set these cases to NaN.
%
% example:
% enforcenan([0 0 0; 1 2 0],2)

bad = all(m==0,dim);
sz = ones(1,max(ndims(m),dim));
sz(dim) = size(m,dim);
bad = repmat(bad,sz);
m(bad) = NaN;
