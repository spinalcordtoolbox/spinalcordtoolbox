function [u, dataError, nSpikes, energy] = iSparsADMM(f, gamma, A, p, varargin)
%iSparsPottsADMM Minimizes the sparsity problem
%
%Description:
% Minimizes the sparsity problem
%
%  \gamma \| u \|_0 + \| A u - f \|_p^p -> min
%
% using a direct ADMM method
%
%Reference
% M. Storath, A. Weinmann, L. Demaret, 
% Jump-sparse and sparse recovery using Potts functionals,
% IEEE Transactions on Signal Processing, 2014
%

% written by M. Storath
% $Date: 2014-06-30 13:19:43 +0200 (Mo, 30 Jun 2014) $	$Revision: 102 $

% precomputations for solving linear equations
if p == 2
    linopts.mat = A' * A;
elseif p ==1
    linopts.mat = A * A';
    m = size(linopts.mat, 1);
    linopts.L = spconvmatrix([-1, 2, -1], m);
else
    error('p must be equal to 1 or 2.')
end
% option for iterative linear solver
linopts.maxit = 500;

% init variables
v = A' * f;
lambda = 0;
w = zeros(size(v));
u = Inf;

% parse options
ip = inputParser;
addParamValue(ip, 'muInit', gamma*1e-6);
addParamValue(ip, 'muStep', 1.05);
addParamValue(ip, 'dispStatus', 1);
addParamValue(ip, 'tol', 1e-6);
addParamValue(ip, 'iMax', 1);
parse(ip, varargin{:});
par = ip.Results;

% init
mu = par.muInit;

% counts total number of iterations
iter = 0;

%%-------------------------------------------------------------------------
%the main loop
while sum(abs(u(:) - v(:)).^2) > par.tol
    
    % ADMM steps
    for i = 1:par.iMax
        % solve L^2 Potts problem
        [u, ~, nSpikes] = minL2Spars( v - lambda/mu, 2*gamma/mu );
        
        % prepare right hand side (substitution v = u - w + lambda)
        b =  A * (u  + lambda/mu) - f;
        
        % switch between L^1 and L^2 data fitting
        if p == 2
            % initial guess (iterative solvers only)
            linopts.init = w;
            % solve L^2-Tikhonov problem
            [w, linflag] = minL2Tikhonov(b, mu/2, A, linopts);
        else
            % solve L^1-Tikhonov problem
            [w, linflag] = minL1Tikhonov(b, mu/2, A, linopts);
            %w = ssn_l1(A, b, mu/2);
        end
        
        % if an eventual linear equation solver did not converge
        if linflag ~= 0
            warning('Linear equation solver did not converge. Results may be inaccurate.')
        end
        
        % resubstitute
        v = u - w + lambda/mu;
        
        % update multiplier
        lambda =  lambda + mu*(u - v);
    end
    
    % increase coupling
    mu = mu * par.muStep;
    
    % update counter
    iter = iter+1;
    
    % show output
    if par.dispStatus == 2
        if mod(iter, 10) == 0
            % output
            subplot(2,2,1)
            showdbl(real(u));
            title('u')
            subplot(2,2,2)
            showdbl(real(v));
            title('v')
            subplot(2,2,3)
            showdbl(real(u-v));
            title('u-v')
            subplot(2,2,4)
            showdbl(real(lambda));
            title('lambda')
            drawnow;
            
            % compute energy  values
            err = A * u - f;
            dataError = sum(abs(err(:)).^p);
            energy = gamma * nSpikes + dataError;
            disp(['Potts-Energy: ' num2str(energy)]);
        end
    elseif par.dispStatus == 1
        % show status
        if mod(iter, 5) == 0
            fprintf('*');
        end
        if mod(iter, 200) == 0
            fprintf('\n');
        end
    end
end

err = A * u - f;
dataError = sum(abs(err(:)).^p);
energy = gamma * nSpikes + dataError;

fprintf(' Done. \n');

end
