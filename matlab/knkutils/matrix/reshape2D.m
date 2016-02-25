function f = reshape2D(m,dim)

% function f = reshape2D(m,dim)
%
% <m> is a matrix
% <dim> is a dimension of <m>
%
% shift dimension <dim> to the beginning,
% then reshape to be a 2D matrix.
% see also reshape2D_undo.m.
%
% example:
% a = randn(3,4,5);
% b = reshape2D(a,2);
% isequal(size(b),[4 15])

% what is the new permutation order? (be careful that <dim> might be larger than number of dimensions of <m>!)
dimorder = [dim setdiff(1:max(ndims(m),dim),dim)];

% permute and then squish into a 2D matrix
f = reshape(permute(m,dimorder),size(m,dim),[]);
