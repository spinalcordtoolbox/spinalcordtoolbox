function [partition, devMatrix] = findBestPartition( f, gamma, dataFidelityNorm, weights )
%findBestPartition Finds the optimal Potts partition

% written by M. Storath
% $Date: 2012/10/29 01:19:08 $	$Revision: 0.1 $

%--------------------------------------------------------------------------
%%Initialization

% n: size of data f
n = numel(f);
% create gamma array
if isscalar(gamma)
    gamma = gamma * ones(n, 1);
else
    gamma = [gamma(1); gamma(:)];
end
% B: Bellmann values (indices start at 1!)
B = zeros(n + 1, 1);
B(1) = -gamma(1);
% partition: stores the optimal partition
partition = zeros(n, 1);

% deviations are only stored if requested, since it needs O(n^2)
% memory
storeDeviations = nargout > 1;
if storeDeviations
    devMatrix = zeros(n);
end

% check input signal
if any(~isfinite(f))
    error('Input vector must consist of finite values.')
end

%--------------------------------------------------------------------------
%%Pre-computations

switch dataFidelityNorm
    case 'L1' % fast \ell^1 computation needs an IndexedLinkedList
        % create an IndexedLinkedList (java object)
        % if IndexedLinkedList is not in the classpath
        if exist('weights', 'var')
            list = javaObject('pottslab.IndexedLinkedHistogram', weights);
        else
            list = javaObject('pottslab.IndexedLinkedHistogramUnweighted', n);
            
        end
        
        
    case 'L2' % \ell^2 computation gets faster by precomputing the moments
        if exist('weights', 'var')
            error('Weighted Potts implemented for L1 data term only')
        end
        m = [0; cumsum(real(f(:)) ) ]; % first moments
        s = [0; cumsum(real(f(:)).^2 ) ]; % second moments
        if ~isreal(f)
            mc = [0; cumsum(imag(f(:)) ) ]; % first moments
            sc = [0; cumsum(imag(f(:)).^2 ) ]; % second moments
        end
end

%--------------------------------------------------------------------------
%%The main loop

%rb: right bound
for rb = 1:n
    switch dataFidelityNorm
        case 'L1' % fast L1 penalty computation, O(n^2) time
            % insert element r to list (sorted)
            % -needs O(n) time
            % -list is a Java object!
            list.insertSorted(f(rb));
            % compute the L1-distances d^*_[lb,rb] for l = 1:rb;
            % -needs O(n) time
            devL = list.computeDeviations();
            
        case 'L1naive' % naive L1 penalty computation, very slow, O(n^3 * log n) time
            devL = zeros(rb,1);
            for lb = 1:rb
                % build signal
                arr = f(lb:rb);
                w = weights(lb:rb);
                % compute median
                m = medianw(arr,w);
                devL(lb) = sum(abs(arr - m) .* w);
            end
            
        case 'L2'
            % compute the L2-deviation d^*_[lb,rb] for lb = 1:rb (vectorized)
            lb = (1:(rb))';
            devL = s(rb+1) - s(lb) - (m(rb+1) - m(lb)).^2 ./ (rb - lb + 1);
            % in case of complex data
            if ~isreal(f)
                devLc = sc(rb+1) - sc(lb) - (mc(rb+1) - mc(lb)).^2 ./ (rb - lb + 1);
                devL = devL.^2 + devLc.^2;
            end
    end
    % due to round-off errors, deviations might be negative. This is
    % corrected here.
    devL( devL < 0 ) = 0;
    devL( rb ) = 0;
    %bv: best value at right bound r
    %blb: best left bound at right bound rb
    [bv, blb] = min( B(1:rb) + gamma(1:rb) + devL );
    partition( rb ) = blb-1;
    B( rb+1 ) = bv;
    
    % store the deviations, if requested
    if storeDeviations
        devMatrix(1:rb,rb) = devL;
    end
end

end

