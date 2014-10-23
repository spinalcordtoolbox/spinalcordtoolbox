function u = minL2Potts2DADMM(f, gamma, varargin)
%minL2Potts2DADMM A ADMM splitting strategy for the two-dimensional vector valued L^2-Potts
%problem
%
%Description:
% Minimizes the (2D) Potts problem 
%
%  \gamma \| u \|_0 + \| A u - f \|_p^p -> min
%
% using ADMM splitting an dynamic programming
% 
%Reference:
% M. Storath, A. Weinmann
% Fast partitioning of vector-valued images",
% SIAM Journal on Imaging Sciences, 2014

% written by M. Storath
% $Date: 2014-06-30 11:26:34 +0200 (Mo, 30 Jun 2014) $	$Revision: 99 $

[m,n,~] = size(f);

% parse options
ip = inputParser;
addParamValue(ip, 'muInit', gamma*1e-2);
addParamValue(ip, 'muStep', 2);
addParamValue(ip, 'tol', 1e-10);
addParamValue(ip, 'isotropic', true);
addParamValue(ip, 'verbose', false);
addParamValue(ip, 'weights', ones(m,n));
addParamValue(ip, 'multiThreading', true);
addParamValue(ip, 'quantization', true);
addParamValue(ip, 'useADMM', true);
parse(ip, varargin{:});
par = ip.Results;

% check args
assert(par.muStep > 1, 'Variable muStep must be > 1.');
assert(all(par.weights(:) >= 0), 'Weights must be >= 0.');
assert(par.tol > 0, 'Stopping tolerance must be > 0.');
assert(par.muInit > 0, 'muInit must be > 0.');

% cast data to PLImage
plf = pottslab.PLImage(f);

% main program (calls Java routines)
if par.isotropic
    % near-isotropic discretization
    omega(1) = sqrt(2.0) - 1.0;
    omega(2) = 1.0 - sqrt(2.0)/2.0;
    % alternative neighborhood weights
    %omega(1) = (2 * sqrt(2.0) - 1.0)/3;
    %omega(2) = (2 - sqrt(2.0))/3;
    plu = pottslab.JavaTools.minL2PottsADMM8(plf, gamma, par.weights, par.muInit, par.muStep, par.tol, par.verbose, par.multiThreading, par.useADMM, omega);
    
else
    % anisotropic discretization
    plu = pottslab.JavaTools.minL2PottsADMM4(plf, gamma, par.weights, par.muInit, par.muStep, par.tol, par.verbose, par.multiThreading, par.useADMM);
end

% reshape the 1D array given by .toDouble()
u = reshape( plu.toDouble(), size(f) );

% to remove small remaining variations in result (algorithm works with floats)
if par.quantization
    u = round(u * 255)/255;
end

end
