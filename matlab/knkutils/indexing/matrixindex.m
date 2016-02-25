function f = matrixindex(m,midx,dim)

% function f = matrixindex(m,midx,dim)
%
% <m> is a matrix X x Y x ... x A x ... x Z
% <midx> is a matrix X x Y x ... x 1 x ... x Z where elements are in 1:A
% <dim> is the dimension of <m> corresponding to A
%
% index into <m> using <midx>.  return a matrix the same size as <midx>.
%
% example:
% x = randn(5,6,7);
% x2 = randn(5,6,7);
% [mx,ix] = max(x,[],3);
% matrixindex(x2,ix)

% reshape to 2D
midxsize = size(midx);
m = reshape2D(m,dim);  % A x X*Y*Z
midx = reshape2D(midx,dim);  % 1 x X*Y*Z

% use indexing to pull out
f = m(midx + (size(m,1)*(0:size(m,2)-1)));  % 1 x X*Y*Z

% undo the reshape
f = reshape2D_undo(f,dim,midxsize);
