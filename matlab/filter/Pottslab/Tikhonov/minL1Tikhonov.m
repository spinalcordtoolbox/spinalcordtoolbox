function [u, flag] = minL1Tikhonov(f, gamma, A, opts)
%minL1Tikhonov Tikhonov regularization with L^1 data term
%
% Solves the optimization problem
%  \gamma \| u \|_2^2 + \| A u - f\|_1 -> min
% by a semi-smooth Newton method
%
% u = minL1Tikhonov(f, gamma, A)
%
%
% Reference:
%  Clason, Jin, Kunisch
%  A semismooth Newton method for L^1 data fitting with automatic choice of regularization parameters and noise calibration
%  SIAM Journal on Scientific Computing, 2010
%

% written by M. Storath
% $Date: 2012/10/29 01:19:08 $	$Revision: 0.1 $

%%-------------------------------------------------------------------------
% init
m = numel(f);
maxit = [];
init = [];
flag = 0;
if exist('opts', 'var')
    if isfield(opts, 'mat')
        AAT = opts.mat;
    end
    if isfield(opts, 'maxit')
        maxit = opts.maxit;
    end
    if isfield(opts, 'init')
        init = opts.init;
    end
    if isfield(opts, 'L')
        L = opts.L;
    end
end
if ~exist('AAT', 'var')
    AAT = A * A';
end
if ~isa(A, 'linop') && ~exist('L', 'var')
    L = spconvmatrix([-1, 2, -1], m);
end

% diagonal idx 
diagidx = eye(m) == 1;

% standard parameters (see Clason, Jin, Kunisch 2010)
iMax = 20; %(iMax higher than standard)
c = 1e9;
beta = 1;
q = 0.2;

% init variables
p = zeros(m,1);
pos = p;
neg = p;
alpha = 2 * gamma;

%deactivate warnings temporaily
warning off;

%%-------------------------------------------------------------------------
% main program
while beta > 1e-16
    pOld = p;
    
    % precompute matrices
    if ~isa(A, 'linop')
        M = (1/alpha) * AAT + beta * L;
        diagM = M(diagidx);
    end
        
    % newton step
    for k = 1:iMax
        % compute active sets
        posNew = p > +1;
        negNew = p < -1;
        comb = posNew | negNew;
        
        % solve linear equation
       
        b = f(:) + c * (posNew - negNew);
        if isa(A, 'linop')
            M = @(x) reshape(AAT * (x / alpha), m, 1) + beta * conv(x(:), [-1; 2; -1], 'same') + c * comb .* x;
            [p, flag] = cgs(M, b, [], maxit, [], [], init);
        else
            M(diagidx) = diagM + c * comb;
            p = M \ b;
        end
         
        % if active sets did not change, break
        if ~(any(posNew - pos) || any(negNew - neg))
            break;
        end
        
        % set new active sets
        pos = posNew;
        neg = negNew;
    end
    
    if (k == iMax) && any(abs(p)>2)
        p = pOld;
        break;
    end
    
    % decrease beta
    beta = beta * q;
end

% restore primal variable
u = A' * (p / alpha);

% reactivate warnings
warning on;