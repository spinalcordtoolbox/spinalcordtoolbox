function [u, flag] = minL2Tikhonov(f, gamma, A, varargin)
%MINL2TIKHONOV Solves the L^2 Tikhonov problem
%
% Computes a minimizer of the L^2 Tikhonov functional
%
%     \| A u - f \|_2^2 + \gamma \| u - u0 \|_2^2
%
% using the normal equation
%
% See also: minL1Tikhonov, minL2iPotts, minL2iSpars, iPottsADMM

% written by M. Storath
% $Date: 2012/10/29 01:19:08 $	$Revision: 0.1 $

% init
flag = 0; % flag for iterative solver

% parse input
ip = inputParser;
addParamValue(ip, 'mat', A'*A); % A'*A
addParamValue(ip, 'maxit', []); % max iterations
addParamValue(ip, 'init', 0); % initial guess
addParamValue(ip, 'u0', []); % offset
addParamValue(ip, 'tol', []); % stop tolerance
addParamValue(ip, 'verbose', false); % show output?
parse(ip, varargin{:});
par = ip.Results;


% right hand side
if isempty(par.u0)
    b = A' * f;
else
    b = A' * (f - A * par.u0);
    par.init = par.init - par.u0; % update initial guess
end

% switch between ordinary matrix and linear operator
if isa(A, 'radonop') && A.useFBP
    if isempty(par.u0)
        w = minL2TikhonovFBP(f, gamma, A);
    else
        w = minL2TikhonovFBP(f - A * par.u0, gamma, A);
    end
 
elseif isa(A, 'linop') % A is a linear operator (function handle of type 'linop')
    % set up system of normal equations
    bCol = b(:);
    M = @(x) reshape(A.normalOp(reshape(x, size(b))) + gamma * reshape(x, size(b)), size(bCol));
    
    if A.posdef
        % if M is positive definite we can use cgs
        method = 'CG';
        [w,flag,relres,iter] = pcg(M, bCol, par.tol, par.maxit, [], [], par.init(:));
    else
        % M is symmetric, but not necessarily positive definite, therefore
        % minimization using minres
        method = 'MINRES';
        [w,flag,relres,iter] = minres(M, bCol, par.tol, par.maxit, [], [], par.init(:));
    end
    
    % reshaping
    w = reshape(w, size(b));
    
    % show output
    if par.verbose
        fprintf('Number of %s iterations %i; Rel. residual %f; Flag %i \n', method, iter, relres, flag);
        
    end
    
elseif isa(A, 'convop') % operator of convolution type, solve using fft
    wHat = fftn(b) ./ (abs(A).^2 + gamma) ;
    w =  ifftn(wHat);
else % ordinary matrix (only for 1D)
    w = (A' * A + speye(size(par.mat)) * gamma) \ b;
end

% resubstitue
if isempty(par.u0)
    u = w;
else
    u = w + par.u0;
end

end

