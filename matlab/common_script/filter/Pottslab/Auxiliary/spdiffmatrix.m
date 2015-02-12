function A = spdiffmatrix( n )
%spdiffmatrix Creates a sparse matrix that acts as differentiation on the input vector
%
% Au = diff(u)

% written by M. Storath
% $Date: 2012/10/29 01:19:08 $	$Revision: 0.1 $

A = spdiags(ones(n-1, 1) * [-1 1], [0, 1], n-1, n);

end

