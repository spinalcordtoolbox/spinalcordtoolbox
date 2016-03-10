function [f,cs] = calcintrinsicdim(m,cutoff)

% function [f,cs] = calcintrinsicdim(m,cutoff)
%
% <m> is points x dimensions
% <cutoff> (optional) is a fraction in [0,1].  default: 0.99.
%
% return <f> as minimum number of eigenvectors to achieve at least <cutoff>
% of the variance in <m>.  also return <cs> as a vector with the cumulative
% sum of the squared eigenvalues.
%
% example:
% X = randn(1000,2);
% X = [X X*randn(2,1)];
% isequal(2,calcintrinsicdim(X))

% input
if ~exist('cutoff','var') || isempty(cutoff)
  cutoff = 0.99;
end

% do it
[u,s,v] = svd(m'*m,0);
cs = cumsum(diag(s)/sum(diag(s)));
f = firstel(find(cs >= cutoff));
