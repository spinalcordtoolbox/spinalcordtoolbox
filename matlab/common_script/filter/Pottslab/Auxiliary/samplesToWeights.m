function weights = samplesToWeights( samples )
%samplesToWeights Computes weights from a set of sample points

% written by M. Storath
% $Date: 2012/10/29 01:19:08 $	$Revision: 0.1 $

%init
weights = zeros(size(samples));

% first and last weight
weights(1) = samples(2) - samples(1);
weights(end) = samples(end) - samples(end-1);

% central weights
subs = 2:numel(samples)-1;
weights(subs) = 0.5 * ( abs(samples(subs) - samples(subs-1)) ...
    + abs(samples(subs) - samples(subs+1)) );

end

