function mu = medianw( data, weight )
% medianw Computes a weighted median of the data

% written by M. Storath
% $Date: 2012/10/29 01:19:08 $	$Revision: 0.1 $

if numel(data) == 1
    mu = data;
else
    [dataSorted, idx] = sort(data);
    weightSorted = weight(idx);
    weightSum = sum(weight);
    cumWeight = cumsum(weightSorted)/weightSum;
    % find first median index 
    idx = find(cumWeight < 0.5, 1, 'last' ) + 1;
    if isempty(idx)
       idx = 1; 
    end
    mu = dataSorted(idx);
end

