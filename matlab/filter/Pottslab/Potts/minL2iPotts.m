function [u, dataError, nJumps, energy] = minL2iPotts( f, gamma, A, varargin )
%minL2iPotts Minimizes the inverse Potts problem
%
%Description
% Minimizes the inverse L^2-Potts problem
%  \gamma \| D u \|_0 + \| A u - f \|_2^2
% by ADMM
%
%Reference
% M. Storath, A. Weinmann, L. Demaret, 
% Jump-sparse and sparse recovery using Potts functionals,
% IEEE Transactions on Signal Processing, 2014
%
% See also: minL2Potts, minL1iPotts, iPottsADMM

% written by M. Storath
% $Revision: 5.26.4.17 $  $Date: 2012/03/01 02:21:56$

[u, dataError, nJumps, energy] = iPottsADMM(f, gamma, A, 2, varargin{:});

end

