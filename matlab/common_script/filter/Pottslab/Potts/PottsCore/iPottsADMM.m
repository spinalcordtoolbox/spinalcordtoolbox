function [u, dataError, nJumps, energy] = iPottsADMM(f, gamma, A, p, varargin)
%iPottsADMM Computes a minimizer of the inverse Potts functional
%
% The function minimizes the inverse Potts functional
%
%    \gamma || D u ||_0 + || A u - f ||_p^p
%
% by an ADMM splitting approach
%
% Syntax:
% [u, dataError, nJumps, energy] = iPottsADMM(f, gamma, A, p)
%
% See also: minL1iPotts, minL2iPotts, minL1iSpars, minL2iSpars

% written by M. Storath
% $Date: 2012/10/29 01:19:08 $	$Revision: 0.1 $



% parse options
ip = inputParser;
%addParamValue(ip, 'muInit', gamma/(2*norm(v(:), 2)^2));
addParamValue(ip, 'muInit', gamma*1e-6);
addParamValue(ip, 'muStep', 1.05);
addParamValue(ip, 'dispStatus', 1);
addParamValue(ip, 'tol', 1e-6);
addParamValue(ip, 'iMax', 1);
addParamValue(ip, 'v', A' * f);
parse(ip, varargin{:});
par = ip.Results;


% init variables
lambda = 0;
v = par.v;
w = zeros(size(v));
u = Inf;

% precomputations for solving linear equations
if p == 2
    linopts.mat = A' * A;
elseif p ==1
    linopts.mat = A * A';
    if isnumeric(A)
        m = size(linopts.mat, 1);
        linopts.L = spconvmatrix([-1, 2, -1], m);
    end
else
    error('p must be equal to 1 or 2.')
end
% option for iterative linear solver
linopts.maxit = 500;

% init mu
mu = par.muInit;

% counts total number of iterations
iter = 0;

%%-------------------------------------------------------------------------
%the main loop
while  sum(abs(u(:) - v(:)).^2) > par.tol
    
    % ADMM steps
    for i = 1:par.iMax
        % solve L^2 Potts problem
        u = minL2Potts( v - lambda/mu, 2*gamma/mu );
        nJumps = countJumps(u);
        
        % prepare right hand side (substitution v = u - w + lambda)
        b = A * (u  + lambda/mu) - f;
        
        % switch between L^1 and L^2 data fitting
        if p == 2
            % initial guess
            linopts.init = w;
            % solve L^2-Tikhonov problem
            [w, linflag] = minL2Tikhonov(b, mu/2, A, linopts);
        else
            % solve L^1-Tikhonov problem
            [w, linflag] = minL1Tikhonov(b, mu/2, A, linopts);
        end
        
        % if an eventual linear equation solver did not converge
        if linflag ~= 0
            warning('Linear equation solver did not converge. Results may be inaccurate.')
        end
        
        % resubstitute
        v = u - w + lambda/mu;
        
        % update multiplier
        lambda =  lambda + mu * (u - v);
    end
    
    % increase coupling
    mu = mu * par.muStep;
    
    % update counter
    iter = iter+1;
    
    % show output
    switch par.dispStatus
        case 2
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
                energy = gamma * nJumps + dataError;
                disp(['Potts-Energy: ' num2str(energy)]);
            end
        case 1
            % show status
            if mod(iter, 5) == 0
                fprintf('*');
            end
            if mod(iter, 200) == 0
                fprintf('\n');
            end
    end
end

% final output
err = A * u - f;
dataError = sum(abs(err(:)).^p);
energy = gamma * nJumps + dataError;

if par.dispStatus > 0
    fprintf(' Done. \n');
end
    
end
