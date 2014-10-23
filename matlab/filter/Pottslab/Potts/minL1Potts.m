function [u, dataError, nJumps, energy] = minL1Potts( f, gamma, varargin )
%minL1Potts Minimizes the (non-inverse) L^1-Potts problem
%
%Description
% Minimizes the L^1 Potts functional
%  \gamma \| D u \|_0 + \| u - f \|_1
%
% The computations are performed in O(n^2) time
% and O(n) space complexity.
%
%Syntax
% pottsEstimator = minL1Potts( f, gamma )
% pottsEstimator = minL1Potts( f, gamma, samples )
%
% samples: sampling points x_i of data f_i, in case of non-equidistant sampling
%
%
%Reference
% A. Weinmann, M. Storath, L. Demaret
% "The $L^1$-Potts functional for robust jump-sparse reconstruction",
% arXiv:1207.4642, http://arxiv.org/abs/1207.4642
%
% See also: minL2Potts, minL1iPotts

% written by M. Storath
% $Date: 2012/10/29 01:19:08 $	$Revision: 0.1 $

%--------------------------------------------------------------------------
%%check for real data
if not(isreal(f))
    error('Data must be real-valued.')
end

ip = inputParser;
addParamValue(ip, 'samples', []);
addParamValue(ip, 'weights', []);
parse(ip, varargin{:});
par = ip.Results;

if not(isempty(par.samples)) && not(isempty(par.weights))
    error('Cannot define samples and weights at the same time.')
end

%--------------------------------------------------------------------------
%%main computation
if isempty(par.samples) && isempty(par.weights)
    partition = findBestPartition(f, gamma, 'L1');
    [u, nJumps] = reconstructionFromPartition(f, partition, 'L1');
    dataError = sum( abs(u - f) );
else
    % convert samples to weights
    if isempty(par.weights)
        if numel(par.samples) ~= numel(f)
            error('Data vector and sample vector must be of equal length');
        end
        par.weights = samplesToWeights(par.samples);
    end
    partition = findBestPartition(f, gamma, 'L1', par.weights);
    [u, nJumps] = reconstructionFromPartition(f, partition, 'L1', par.weights);
    dataError = sum( abs(u(:) - f(:)) .* par.weights(:) );
end

energy = gamma * nJumps + dataError;

%--------------------------------------------------------------------------
%%show the result, if requested
if nargout == 0
    if ~exist('samples', 'var')
        samples = linspace(0,1,numel(f));
    end
    plot(samples, f, '.', 'MarkerSize', 10);
    hold on
    stairs(samples, u, 'r', 'LineWidth', 2);
    hold off
    legend({'Signal', 'L^1-Potts estimate'});
    grid on;
end

end

