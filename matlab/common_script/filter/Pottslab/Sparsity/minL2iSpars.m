function [u, dataError, nSpikes, energy] = minL2iSpars( f, gamma, A, method, varargin )
%minL2iSpars Minimizes the sparsity problem
%
%Description
% Minimizes the L^2 sparsity problem
%  \gamma \| u \|_0 + \| A u - f \|_2^2 -> min
%
% using inverse Potts functionals (see iPottsADMM)
%
%Reference
% M. Storath, A. Weinmann, L. Demaret, 
% Jump-sparse and sparse recovery using Potts functionals,
% IEEE Transactions on Signal Processing, 2014
%
% See also: minL1iSpars, iPottsADMM

% written by M. Storath
% $Date: 2012/10/29 01:19:08 $	$Revision: 0.1 $

if not(exist('method', 'var'))
    method = 'PottsADMM';
end


switch method
    case 'PottsADMM'
        [u, dataError, nSpikes, energy] = iSparsByPottsADMM( f, gamma, A, 2, varargin{:} );
    case 'ADMM'
        [u, dataError, nSpikes, energy] = iSparsADMM( f, gamma, A, 2, varargin{:} );
    otherwise
        error('This method does not exist.')
end


end

