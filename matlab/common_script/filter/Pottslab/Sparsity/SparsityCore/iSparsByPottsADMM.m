function [u, dataError, nSpikes, energy] = iSparsByPottsADMM(f, gamma, A, p, opts)
%iSparsPottsADMM Minimizes the sparsity problem using an Potts ADMM method
%
%Description:
% Minimizes the sparsity functional
%
%  \gamma \| u \|_0 + \| A u - f \|_p^p
%
% using the inverse Potts functional (see iPottsADMM) 

% written by M. Storath
% $Date: 2013-04-05 12:22:32 +0200 (Fr, 05 Apr 2013) $	$Revision: 73 $

% create differentiation matrix
if isa(A, 'linop')
    D = linop( @(x) diff(x), @(x) conv(x, [-1 1], 'full') );
elseif isa(A, 'convop')
    d(numel(f),1) = -1; d(1) = 1;
    D = convop(fft(d));
else
    D = spdiffmatrix(size(A, 2)+1);  
end

% create substitution matrix
B = A * D;

% parse parameters
if ~exist('opts', 'var')
    % standard parameters
    opts.muInit = gamma * 1e-6;
    %opts.muInit = 2 * gamma / norm(B' * f, 2)^2 ;
    opts.muStep = 1.05;
    opts.dispStatus = 1;
    opts.tol = 1e-6;
    opts.iMax = 1;
end

% call Potts problem with matrix A * D
v = iPottsADMM(f, gamma, B, p, opts);

% resubstitution
u = real(D * v);

% compute error
res = A * u - f;
dataError = sum(res(:).^p);

% count number of spikes
nSpikes = sum(u(:) ~= 0);

% total energy
energy = gamma * nSpikes + dataError;

end
