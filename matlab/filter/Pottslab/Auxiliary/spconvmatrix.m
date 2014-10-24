function A = spconvmatrix( filt, n )
%SPCONVMATRIX Computes a sparse convolution matrix of size n x n from the given
%filter
% 
% The output matrix A is such that it acts on a column vector as convolution, i.e.,
% 
%    A v = filt * v

% written by M. Storath
% $Date: 2012/10/29 01:19:08 $	$Revision: 0.1 $

% convolution interchanges order
filt = flipud(filt(:));

% build toeplitz matrix
m = numel(filt);
idx = (1:m) - ceil(numel(filt)/2);
A = spdiags( ones(n, 1) * filt', idx, n, n);

end

