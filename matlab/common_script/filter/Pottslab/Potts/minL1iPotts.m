function [u, dataError, nJumps, energy] = minL1iPotts( f, gamma, A, varargin )
%minL1iPotts Minimizes the inverse Potts problem
%
%Description
% Minimizes the inverse L^1-Potts functional
%  \gamma \| D u \|_0 + \| A u - f \|_1
% by an ADMM splitting approach
%
%Reference
% M. Storath, A. Weinmann, L. Demaret, 
% Jump-sparse and sparse recovery using Potts functionals,
% IEEE Transactions on Signal Processing, 2014
%
% See also: minL2iPotts, iPottsADMM

% written by M. Storath
% $Date: 2013-01-05 17:25:45 +0100 (Sat, 05 Jan 2013) $	$Revision: 63 $

[u, dataError, nJumps, energy] = iPottsADMM(f, gamma, A, 1, varargin{:});

end

