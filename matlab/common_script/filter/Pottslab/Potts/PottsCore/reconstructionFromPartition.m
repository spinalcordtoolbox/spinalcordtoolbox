function [u, nJumps] = reconstructionFromPartition( f, partition, dataFidelityNorm, weights )
%reconstructionFromPartition Reconstructs the minimizer from the optimal
%partition

% written by M. Storath
% $Date: 2012/10/29 01:19:08 $	$Revision: 0.1 $

u = zeros(size(f)); % preallocation
% start at right bound rb = n
if isvector(f)
    rb = numel(f);
else
    rb = size(f, 1);
end
%init nJumps
nJumps = -1;
%check for weights
weighted =  exist('weights', 'var');

while rb > 0
    % partition(rb) stores corresponding optimal left bound lb
    lb = partition(rb);
    interval = (lb+1) : rb;
    switch dataFidelityNorm
        case 'L1'
            % best approximation on partition is the median
            if weighted
                % best approximation on partition is the weighted median
                muLR = medianw(f(interval), weights(interval));
            else
                muLR = median(f(interval));
            end
            
        case 'L2'
            if weighted
                % best approximation on partition is the weighted mean value
                muLR = sum(f(interval).*weights(interval)) / sum(weights(interval));
            else
                % best approximation on partition is the mean value
                muLR = mean(f(interval));
            end
            
        case 'L2v'
            % best approximation on partition is the mean value
            muLR = mean(f(interval, :),1);
    end
    % set values of current interval to optimal value
    u(interval, :) = ones(numel(interval), 1) * muLR;
    % continue with next right bound
    rb = lb;
    % update nJumps
    nJumps = nJumps + 1;
end
u = double(u);
end

