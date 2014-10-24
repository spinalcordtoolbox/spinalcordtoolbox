function [u, dataError, nSpikes, energy] = minL1iSpars( f, gamma, A, varargin )
%minL1SparsInv Minimizes the sparsity problem
%
%Description
% Minimizes the L^1-sparsity problem 
%
%  \gamma \| u \|_0 + \| A u - f \|_1 -> min
%
% using the inverse Potts functional
%
%
%Reference
% M. Storath, A. Weinmann, L. Demaret, 
% Jump-sparse and sparse recovery using Potts functionals,
% IEEE Transactions on Signal Processing, 2014
%
% See also: minL2iSpars, minL1iSpars, iSparsByPottsADMM

% written by M. Storath
% $Date: 2012/10/29 01:19:08 $	$Revision: 0.1 $

[u, dataError, nSpikes, energy] = iSparsByPottsADMM( f, gamma, A, 1, varargin{:} );

end

